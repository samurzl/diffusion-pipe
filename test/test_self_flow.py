import torch

from utils.self_flow import mask_to_runs, representation_cosine_loss, sample_bernoulli_mask


def test_bernoulli_mask_tracks_the_configured_ratio():
    torch.manual_seed(7)
    mask = sample_bernoulli_mask((100_000,), 0.1, torch.device('cpu'))
    assert mask.dtype == torch.bool
    assert abs(mask.float().mean().item() - 0.1) < 0.01


def test_mask_to_runs_preserves_packed_token_order():
    mask = torch.tensor([False, False, True, True, False, True])
    assert mask_to_runs(mask, 10, 16, false_row=2, true_row=3) == [
        (10, 12, 2),
        (12, 14, 3),
        (14, 15, 2),
        (15, 16, 3),
    ]


def test_mask_to_runs_supports_an_empty_audio_stream():
    assert mask_to_runs(torch.empty(0, dtype=torch.bool), 7, 7, 2, 3) == [(7, 7, 2)]


def test_representation_loss_stops_teacher_gradients():
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    loss = representation_cosine_loss(student, teacher)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(-0.5))
    assert student.grad is not None
    assert teacher.grad is None
