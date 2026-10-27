import torch

from src.models.losses import NucleiDiceLoss


def test_nuclei_dice_loss_has_gradient():
    """Regression: the old hard-threshold mask had zero gradient."""
    fake = torch.rand(2, 1, 32, 32, requires_grad=True)
    real = torch.rand(2, 1, 32, 32)
    loss = NucleiDiceLoss(thresh=0.1, temperature=0.05)(fake, real)
    loss.backward()
    assert fake.grad is not None
    assert torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0


def test_nuclei_dice_loss_prefers_matching_masks():
    real = torch.zeros(1, 1, 16, 16)
    real[:, :, 4:12, 4:12] = 0.5
    good = real.clone().requires_grad_(True)
    bad = torch.zeros(1, 1, 16, 16, requires_grad=True)
    loss = NucleiDiceLoss()
    assert loss(good, real) < loss(bad, real)


def test_nuclei_dice_loss_supports_hard_eval():
    real = torch.zeros(1, 1, 16, 16)
    real[:, :, 4:12, 4:12] = 0.5
    matching = real.clone()
    empty = torch.zeros_like(real)
    loss = NucleiDiceLoss(soft=False)
    assert loss(matching, real) == 0
    assert torch.isclose(loss(empty, real), torch.tensor(1.0), atol=1e-6)
