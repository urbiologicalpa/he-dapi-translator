import torch
import torch.nn as nn

from src.models.train_paired import validate


class TinyGenerator(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x).expand(-1, 3, -1, -1)


def test_validate_reports_metrics():
    torch.manual_seed(0)
    loader = [{
        "he": torch.rand(1, 3, 16, 16) * 2 - 1,
        "dapi": torch.rand(1, 1, 16, 16),
    }]
    metrics = validate(TinyGenerator(), loader, torch.device("cpu"), "dapi2he")
    assert metrics["n"] == 1
    assert metrics["l1"] >= 0
    assert 0 <= metrics["ssim"] <= 1
    assert metrics["psnr"] > 0
    assert metrics["score"] == metrics["l1"] + 0.25 * (1 - metrics["ssim"])


class ConstantGenerator(nn.Module):
    def forward(self, x):
        return torch.full_like(x[:, :1], 0.5)


def test_validate_uses_real_ssim_not_ssim_loss():
    """Perfect prediction -> SSIM 1 and score ~0.

    Regression for the Stage 2.5 bug where ``SSIMLoss`` (1-SSIM / cosine proxy)
    was reported as SSIM and inverted best-checkpoint selection.
    """
    he = torch.zeros(1, 3, 32, 32)
    dapi = torch.full((1, 1, 32, 32), 0.5)
    loader = [{"he": he, "dapi": dapi}]
    metrics = validate(ConstantGenerator(), loader, torch.device("cpu"), "he2dapi")
    assert metrics["l1"] == 0
    assert metrics["ssim"] > 0.999
    assert metrics["score"] < 1e-3
