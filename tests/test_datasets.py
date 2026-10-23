"""Dataset fallback regression tests (fast, no torch required).

Guards the synthetic-data path used by ``--dummy`` training smoke tests: when no
HDF5/root data is present, a requested ``indices`` subset must be honoured so
``__len__`` and ``__getitem__`` stay in sync. Regression for the CI failure
``IndexError: range object index out of range``.
"""
from pathlib import Path

import pytest

from src.models import datasets as datasets_mod
from src.models.datasets import PairedHEMITDataset


@pytest.mark.parametrize("n_indices", [1, 4, 8])
def test_synthetic_fallback_honours_indices(n_indices):
    ds = PairedHEMITDataset(
        h5_path=str(Path("/nonexistent/missing.h5")),
        root_dir=str(Path("/nonexistent/missing_root")),
        patch_size=32,
        augment=False,
        indices=range(n_indices),
    )
    assert len(ds) == n_indices
    for i in range(len(ds)):
        sample = ds[i]
        # torch -> CHW (3,32,32) / (1,32,32); no-torch -> numpy HWC (32,32,3) / (32,32)
        assert set(sample["he"].shape) == {3, 32, 32}
        assert set(sample["dapi"].shape) in ({1, 32, 32}, {32, 32})


def test_synthetic_fallback_without_indices():
    ds = PairedHEMITDataset(
        h5_path=str(Path("/nonexistent/missing.h5")),
        root_dir=str(Path("/nonexistent/missing_root")),
        patch_size=32,
        augment=False,
    )
    assert len(ds) == 8
    assert set(ds[7]["he"].shape) == {3, 32, 32}


def test_unpaired_h5_only_domains_use_h5_not_synthetic_noise(tmp_path, monkeypatch):
    """Kaggle bundles have no data/examples PNGs: an H5-backed domain must be
    sampled from the H5 even when the 50/50 PNG branch is not taken.

    Regression for the 2026-10-22 CycleGAN runs where ``[None]*8`` placeholders
    made 50% of H&E (and some DAPI) samples uniform random noise.
    """
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    from src.models.datasets import UnpairedDataset

    # Simulate the Kaggle bundle, which has no data/examples PNGs.
    real_glob = Path.glob

    def fake_glob(self, pattern):
        if pattern in {"he_*.png", "dapi_*.png"}:
            return iter(())
        return real_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", fake_glob)

    n, h = 4, 16
    h5_path = tmp_path / "domains.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("he", data=np.full((n, h, h, 3), 127, dtype=np.uint8))
        f.create_dataset("dapi", data=np.full((n, h, h), 63, dtype=np.uint8))

    ds = UnpairedDataset(
        he_root=str(tmp_path / "missing_he"),
        dapi_root=str(tmp_path / "missing_dapi"),
        patch_size=h,
        he_h5=str(h5_path),
        dapi_h5=str(h5_path),
    )
    assert len(ds) == n
    assert ds._he_real_files == []
    assert ds._dapi_real_files == []

    # Force the "load from files" branch; with no real files the H5 must win
    # instead of falling through to the noise fallback.
    monkeypatch.setattr(np.random, "rand", lambda *a, **k: 0.99)
    sample = ds[0]
    assert float(sample["he"].std()) == pytest.approx(0.0, abs=1e-6)
    assert float(sample["dapi"].std()) == pytest.approx(0.0, abs=1e-6)
    if datasets_mod.torch is None:
        # CI resource-guards job installs numpy/h5py without torch. The loader
        # returns uint8 arrays in that mode; normalisation is torch-only.
        assert float(sample["he"].mean()) == pytest.approx(127.0, abs=1e-6)
        assert float(sample["dapi"].mean()) == pytest.approx(63.0, abs=1e-6)
    else:
        assert float(sample["he"].mean()) == pytest.approx(127 / 127.5 - 1.0, abs=1e-6)
        assert float(sample["dapi"].mean()) == pytest.approx(63 / 255.0, abs=1e-6)


def test_explicit_h5_is_authoritative_over_examples(tmp_path):
    """An explicit H5 must not be mixed with data/examples PNGs.

    The repository has data/examples/he_*.png, so the old fallback order made
    an "H5-only" config silently sample example images 50% of the time. This
    keeps val/test/label TIFFs out of the unpaired pool.
    """
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    from src.models.datasets import UnpairedDataset

    n, h = 2, 16
    h5_path = tmp_path / "domains.h5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("he", data=np.full((n, h, h, 3), 100, dtype=np.uint8))
        f.create_dataset("dapi", data=np.full((n, h, h), 50, dtype=np.uint8))

    ds = UnpairedDataset(
        he_root=str(tmp_path / "missing_he"),
        dapi_root=str(tmp_path / "missing_dapi"),
        patch_size=h,
        he_h5=str(h5_path),
        dapi_h5=str(h5_path),
    )
    assert ds._he_real_files == []
    assert ds._dapi_real_files == []
    assert len(ds) == n
