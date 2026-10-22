"""Colour/contrast and gradient losses used by the H&E quality pass."""
import torch

from src.models.losses import ColorStatsLoss, GradientLoss


def test_color_stats_zero_for_identical():
    x = torch.rand(1, 3, 16, 16)
    assert ColorStatsLoss()(x, x).item() < 1e-6


def test_gradient_zero_for_identical():
    x = torch.rand(1, 3, 16, 16)
    assert GradientLoss()(x, x).item() < 1e-6


def test_quality_losses_positive_for_flat_vs_textured():
    flat = torch.zeros(1, 3, 16, 16)
    textured = torch.rand(1, 3, 16, 16)
    assert ColorStatsLoss()(flat, textured).item() > 0
    assert GradientLoss()(flat, textured).item() > 0
