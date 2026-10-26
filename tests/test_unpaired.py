"""CycleGAN generator, H5-only unpaired dataset, and reverse checkpoint loading."""
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.train_unpaired import ResnetGenerator


def test_resnet_generator_shape_and_range():
    G = ResnetGenerator(in_ch=1, out_ch=3, ngf=8, n_blocks=1).eval()
    with torch.no_grad():
        out = G(torch.randn(1, 1, 32, 32) * 10)
    assert out.shape == (1, 3, 32, 32)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_unpaired_h5_only_dataset(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    he_p, dapi_p = tmp_path / "he.h5", tmp_path / "dapi.h5"
    with h5py.File(he_p, "w") as f:
        f.create_dataset("he", data=np.zeros((6, 32, 32, 3), dtype=np.uint8))
    with h5py.File(dapi_p, "w") as f:
        f.create_dataset("dapi", data=np.zeros((4, 32, 32), dtype=np.uint8))

    from src.models.datasets import UnpairedDataset

    ds = UnpairedDataset(he_root=str(tmp_path / "none"), dapi_root=str(tmp_path / "none"),
                         patch_size=32, he_h5=str(he_p), dapi_h5=str(dapi_p))
    assert len(ds) == 6
    item = ds[0]
    assert item["he"].shape == (3, 32, 32)
    assert item["dapi"].shape == (1, 32, 32)


def test_reverse_infer_loads_cyclegan_checkpoint(tmp_path: Path):
    from src.models.infer import run_torch_model

    G = ResnetGenerator(in_ch=1, out_ch=3, ngf=8, n_blocks=1).eval()
    ckpt = tmp_path / "cyclegan.pt"
    torch.save({"G": G.state_dict()}, ckpt)
    tile = np.random.default_rng(0).integers(0, 255, (32, 32), dtype=np.uint8)
    out = run_torch_model(tile, "dapi2he", str(ckpt))
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8
