"""Regression tests for HEMIT raw RGB channel order.

Ground truth (2026-10-23 audit): ch0=panCK (red), ch1=CD3 (green),
ch2=DAPI (blue). The loader must derive DAPI from ``mihc[...,2]`` even when a
pre-audit H5 carries a wrong ``dapi`` cache built from ch0.
"""
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.data.pack_hemit_1024 import CHANNEL_ORDER, DAPI_CHANNEL, extract_dapi
from src.models.datasets import PairedHEMITDataset


def test_channel_constants():
    assert CHANNEL_ORDER == ("panCK", "CD3", "DAPI")
    assert DAPI_CHANNEL == 2


def test_extract_dapi_uses_ch2():
    mihc = np.zeros((2, 2, 3), dtype=np.uint8)
    mihc[..., 0] = 10
    mihc[..., 1] = 20
    mihc[..., 2] = 30
    assert np.array_equal(extract_dapi(mihc), mihc[..., 2])


def test_extract_dapi_rejects_non_rgb():
    with pytest.raises(ValueError):
        extract_dapi(np.zeros((2, 2), dtype=np.uint8))


def _write_h5(path: Path, *, with_mihc: bool, dapi_cache: int) -> None:
    mihc = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    mihc[..., 0] = 10
    mihc[..., 1] = 20
    mihc[..., 2] = 30
    with h5py.File(path, "w") as f:
        f.create_dataset("he", data=np.zeros((1, 4, 4, 3), dtype=np.uint8))
        if with_mihc:
            f.create_dataset("mihc", data=mihc)
        f.create_dataset("dapi", data=np.full((1, 4, 4), dapi_cache, dtype=np.uint8))
        f.attrs["patch_size"] = 4


def _dapi_value(ds) -> float:
    out = ds[0]["dapi"]
    return out[0, 0, 0].item() if hasattr(out, "item") else float(out[0, 0])


def test_loader_prefers_mihc_ch2_over_wrong_dapi_cache(tmp_path):
    path = tmp_path / "wrong_cache.h5"
    _write_h5(path, with_mihc=True, dapi_cache=10)  # cache is ch0/panCK
    ds = PairedHEMITDataset(h5_path=str(path), patch_size=4, augment=False,
                            indices=range(1))
    assert _dapi_value(ds) == pytest.approx(30 / 255)


def test_loader_uses_dapi_key_without_mihc(tmp_path):
    path = tmp_path / "dapi_only.h5"
    _write_h5(path, with_mihc=False, dapi_cache=30)  # already ch2 after rebuild
    ds = PairedHEMITDataset(h5_path=str(path), patch_size=4, augment=False,
                            indices=range(1))
    assert _dapi_value(ds) == pytest.approx(30 / 255)


def test_unpaired_loader_prefers_mihc_ch2(tmp_path):
    from src.models.datasets import UnpairedDataset

    path = tmp_path / "unpaired_wrong_cache.h5"
    _write_h5(path, with_mihc=True, dapi_cache=10)  # cache is ch0/panCK
    ds = UnpairedDataset(
        he_root=str(tmp_path), dapi_root=str(tmp_path),
        patch_size=4, augment=False, he_h5=str(path), dapi_h5=str(path),
    )
    out = ds[0]["dapi"]
    value = out[0, 0, 0].item() if hasattr(out, "item") else float(out[0, 0])
    assert value == pytest.approx(30 / 255)
