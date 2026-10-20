"""Regression tests for HEMIT channel order (DAPI=ch0, CD3=ch1, panCK=ch2)."""
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.data.pack_hemit_1024 import CHANNEL_ORDER, DAPI_CHANNEL, extract_dapi
from src.models.datasets import PairedHEMITDataset


def test_channel_constants():
    assert CHANNEL_ORDER == ("DAPI", "CD3", "panCK")
    assert DAPI_CHANNEL == 0


def test_extract_dapi_uses_ch0():
    mihc = np.zeros((2, 2, 3), dtype=np.uint8)
    mihc[..., 0] = 10
    mihc[..., 1] = 20
    mihc[..., 2] = 30
    assert np.array_equal(extract_dapi(mihc), mihc[..., 0])


def test_extract_dapi_rejects_non_rgb():
    with pytest.raises(ValueError):
        extract_dapi(np.zeros((2, 2), dtype=np.uint8))


def _write_h5(path: Path, with_dapi_key: bool) -> None:
    mihc = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    mihc[..., 0] = 10
    mihc[..., 1] = 20
    mihc[..., 2] = 30
    with h5py.File(path, "w") as f:
        f.create_dataset("he", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        f.create_dataset("mihc", data=mihc)
        if with_dapi_key:
            f.create_dataset("dapi", data=mihc[..., 0])
        f.attrs["patch_size"] = 4


def test_loader_prefers_dapi_key(tmp_path):
    path = tmp_path / "with_dapi.h5"
    _write_h5(path, with_dapi_key=True)
    ds = PairedHEMITDataset(h5_path=str(path), patch_size=4, augment=False, indices=range(1))
    out = ds[0]["dapi"]
    value = out[0, 0, 0].item() if hasattr(out, "item") else float(out[0, 0])
    assert value == pytest.approx(10 / 255)


def test_loader_mihc_fallback_uses_ch0(tmp_path):
    path = tmp_path / "without_dapi.h5"
    _write_h5(path, with_dapi_key=False)
    ds = PairedHEMITDataset(h5_path=str(path), patch_size=4, augment=False, indices=range(1))
    out = ds[0]["dapi"]
    value = out[0, 0, 0].item() if hasattr(out, "item") else float(out[0, 0])
    assert value == pytest.approx(10 / 255)
