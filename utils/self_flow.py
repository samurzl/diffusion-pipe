import torch
import torch.nn.functional as F


def sample_bernoulli_mask(shape, ratio, device):
    """Sample the paper's iid uniform-threshold token mask."""
    return torch.rand(shape, device=device) < float(ratio)


def mask_to_runs(mask, start, stop, false_row, true_row):
    """Convert a flat boolean token mask to contiguous timestep-row runs."""
    values = mask.reshape(-1).tolist()
    if len(values) != stop - start:
        raise RuntimeError(
            f'Self-Flow token mask has {len(values)} entries for a {stop - start}-token segment'
        )
    if not values:
        return [(start, stop, false_row)]

    runs = []
    run_start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[run_start]:
            row = true_row if values[run_start] else false_row
            runs.append((start + run_start, start + index, row))
            run_start = index
    return runs


def representation_cosine_loss(projected_student, teacher):
    """Negative cosine similarity used by the Self-Flow representation loss."""
    return -F.cosine_similarity(
        projected_student.float(), teacher.detach().float(), dim=-1
    ).mean()
