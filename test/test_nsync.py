import torch

from utils.nsync import (
    NSYNC_ANCHOR,
    NSYNC_NEGATIVE,
    NSYNC_POSITIVE,
    NSYNCGradientController,
    build_anchor_iteration_order,
)


def test_anchor_iteration_order_is_deterministic_across_groups():
    first = build_anchor_iteration_order([2, 3], target_length=8, seed=1234)
    second = build_anchor_iteration_order([2, 3], target_length=8, seed=1234)

    assert first == second
    assert len(first) == 8
    assert set(first[:5]) == {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)}
    assert all(source_index in (0, 1) for source_index, _ in first)


def _backward_role(controller, parameters, gradients, role):
    output = sum((parameter * gradient).sum() for parameter, gradient in zip(parameters, gradients))
    output = controller.tag_output(output, torch.tensor([role]))
    output.backward()


def test_nsync_uses_global_paper_projection():
    parameters = [
        torch.tensor([0.2, -0.4], requires_grad=True),
        torch.tensor([0.7], requires_grad=True),
    ]
    positive = [torch.tensor([2.0, -1.0]), torch.tensor([3.0])]
    negative = [torch.tensor([1.0, 2.0]), torch.tensor([-2.0])]
    anchor = [torch.tensor([-1.0, 1.0]), torch.tensor([4.0])]

    controller = NSYNCGradientController(gradient_scale=1.0)
    controller.register_parameters(parameters)
    _backward_role(controller, parameters, positive, NSYNC_POSITIVE)
    _backward_role(controller, parameters, negative, NSYNC_NEGATIVE)
    _backward_role(controller, parameters, anchor, NSYNC_ANCHOR)
    controller.apply_gradient_surgery()

    positive_flat = torch.cat(positive)
    negative_flat = torch.cat(negative)
    anchor_flat = torch.cat(anchor)
    expected = (
        positive_flat
        - positive_flat.dot(negative_flat) / negative_flat.square().sum() * negative_flat
        + positive_flat.dot(anchor_flat) / anchor_flat.square().sum() * anchor_flat
    )
    actual = torch.cat([parameter.grad for parameter in parameters])
    torch.testing.assert_close(actual, expected)


def test_nsync_restores_three_microbatch_gradient_scale_and_clears_state():
    parameter = torch.tensor([1.0, 2.0], requires_grad=True)
    controller = NSYNCGradientController(gradient_scale=3.0)
    controller.register_parameters([parameter])
    _backward_role(controller, [parameter], [torch.tensor([1.0, 0.0])], NSYNC_POSITIVE)
    _backward_role(controller, [parameter], [torch.tensor([0.0, 1.0])], NSYNC_NEGATIVE)
    _backward_role(controller, [parameter], [torch.tensor([1.0, 0.0])], NSYNC_ANCHOR)
    controller.apply_gradient_surgery()

    torch.testing.assert_close(parameter.grad, torch.tensor([6.0, 0.0]))
    assert controller.current_role is None
    assert all(not gradients for gradients in controller.role_grads.values())
