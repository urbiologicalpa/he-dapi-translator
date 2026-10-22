"""Reverse inference must use the trained paired U-Net, not the CycleGAN stub."""
from pathlib import Path

import numpy as np
import torch

from src.models.infer import run_torch_model
from src.models.train_paired import UNetGenerator


def test_reverse_infer_uses_unet_checkpoint(tmp_path: Path):
    G = UNetGenerator(in_ch=1, out_ch=3, ngf=64).eval()
    ckpt = tmp_path / "g.pt"
    torch.save({"G": G.state_dict()}, ckpt)
    tile = np.random.default_rng(0).integers(0, 255, (32, 32), dtype=np.uint8)
    out = run_torch_model(tile, "dapi2he", str(ckpt))
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8

    inp = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float() / 255.0
    expected = (G(inp * 2 - 1)[0].permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    assert np.allclose(out, expected, atol=1)
