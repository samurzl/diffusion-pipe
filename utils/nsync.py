import torch
try:
    from deepspeed import comm as dist
except ImportError:  # Allows the gradient algebra unit test to run without the training stack.
    dist = None


NSYNC_POSITIVE = 0
NSYNC_NEGATIVE = 1
NSYNC_ANCHOR = 2
NSYNC_ROLE_NAMES = {
    NSYNC_POSITIVE: 'positive',
    NSYNC_NEGATIVE: 'negative',
    NSYNC_ANCHOR: 'anchor',
}


class NSYNCGradientController:
    """Accumulate NSYNC role gradients and apply the paper's projection update.

    Positive, negative, and anchor examples are forwarded as separate microbatches.
    Parameter hooks retain the two auxiliary gradients while allowing only the
    positive gradient into ``param.grad``. Immediately before DeepSpeed clips and
    steps the optimizer, :meth:`apply_gradient_surgery` replaces it with

        g_pos - proj_{g_neg}(g_pos) + proj_{g_anchor}(g_pos).

    This avoids keeping three MiniMax H3 activation graphs alive at once.
    """

    def __init__(self, eps=1e-8, gradient_scale=3.0):
        self.eps = float(eps)
        self.gradient_scale = float(gradient_scale)
        self.current_role = None
        self.parameters = []
        self.role_grads = {
            NSYNC_NEGATIVE: {},
            NSYNC_ANCHOR: {},
        }
        self._handles = []

    def register_parameters(self, parameters):
        if self._handles:
            raise RuntimeError('NSYNC gradient hooks were already registered')
        self.parameters = list(parameters)
        if not self.parameters:
            raise RuntimeError('NSYNC did not find any trainable LoRA parameters')
        for parameter in self.parameters:
            self._handles.append(parameter.register_hook(self._make_parameter_hook(parameter)))

    def _make_parameter_hook(self, parameter):
        def hook(grad):
            role = self.current_role
            if role == NSYNC_POSITIVE:
                return grad
            if role not in (NSYNC_NEGATIVE, NSYNC_ANCHOR):
                # Normal/eval datasets do not participate in NSYNC surgery.
                return grad

            role_grads = self.role_grads[role]
            if parameter in role_grads:
                role_grads[parameter].add_(grad.detach())
            else:
                role_grads[parameter] = grad.detach().clone()
            return torch.zeros_like(grad)

        return hook

    def tag_output(self, output, roles):
        """Tag a graph edge so parameter hooks see the role of its microbatch.

        Pipeline schedules can interleave forwards and backwards. Capturing the
        role in an output hook associates it with the actual backward graph rather
        than mutable forward-time state.
        """
        if not output.requires_grad:
            return output
        role_values = roles.detach().reshape(-1)
        if role_values.numel() == 0:
            return output
        role = int(role_values[0].item())
        if not torch.all(role_values == role):
            raise RuntimeError('Every NSYNC microbatch must contain exactly one role')

        def set_role(grad):
            self.current_role = role
            return grad

        output.register_hook(set_role)
        return output

    @staticmethod
    def _local_projection_stats(parameters, negative_grads, anchor_grads, device):
        stats = torch.zeros(4, dtype=torch.float64, device=device)
        for parameter in parameters:
            positive = parameter.grad
            if positive is None:
                continue
            negative = negative_grads.get(parameter)
            anchor = anchor_grads.get(parameter)
            if negative is not None:
                stats[0] += torch.sum(positive.detach().float() * negative.float()).to(device, torch.float64)
                stats[1] += torch.sum(negative.float().square()).to(device, torch.float64)
            if anchor is not None:
                stats[2] += torch.sum(positive.detach().float() * anchor.float()).to(device, torch.float64)
                stats[3] += torch.sum(anchor.float().square()).to(device, torch.float64)
        return stats

    @torch.no_grad()
    def apply_gradient_surgery(self):
        negative_grads = self.role_grads[NSYNC_NEGATIVE]
        anchor_grads = self.role_grads[NSYNC_ANCHOR]
        if not negative_grads or not anchor_grads:
            found = {
                NSYNC_ROLE_NAMES[role]: len(grads)
                for role, grads in self.role_grads.items()
            }
            raise RuntimeError(f'NSYNC step did not receive every required role: {found}')

        first_grad = next((p.grad for p in self.parameters if p.grad is not None), None)
        if first_grad is None:
            raise RuntimeError('NSYNC step did not receive a positive gradient')

        stats = self._local_projection_stats(
            self.parameters,
            negative_grads,
            anchor_grads,
            first_grad.device,
        )
        if dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(stats)

        neg_coefficient = stats[0] / stats[1].clamp_min(self.eps)
        anchor_coefficient = stats[2] / stats[3].clamp_min(self.eps)
        neg_coefficient_value = neg_coefficient.item()
        anchor_coefficient_value = anchor_coefficient.item()

        for parameter in self.parameters:
            positive = parameter.grad
            if positive is None:
                continue
            negative = negative_grads.get(parameter)
            anchor = anchor_grads.get(parameter)
            if negative is not None:
                positive.add_(negative, alpha=-neg_coefficient_value)
            if anchor is not None:
                positive.add_(anchor, alpha=anchor_coefficient_value)
            positive.mul_(self.gradient_scale)

        self.clear()
        return {
            'negative_projection_coefficient': neg_coefficient_value,
            'anchor_projection_coefficient': anchor_coefficient_value,
        }

    def clear(self):
        for grads in self.role_grads.values():
            grads.clear()
        self.current_role = None
