"""Dataset fallback regression tests (fast, no torch required).

Guards the synthetic-data path used by ``--dummy`` training smoke tests: when no
HDF5/root data is present, a requested ``indices`` subset must be honoured so
``__len__`` and ``__getitem__`` stay in sync. Regression for the CI failure
``IndexError: range object index out of range``.
"""
from pathlib import Path

import pytest

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
        assert sample["he"].shape == (3, 32, 32)
        assert sample["dapi"].shape == (1, 32, 32)


def test_synthetic_fallback_without_indices():
    ds = PairedHEMITDataset(
        h5_path=str(Path("/nonexistent/missing.h5")),
        root_dir=str(Path("/nonexistent/missing_root")),
        patch_size=32,
        augment=False,
    )
    assert len(ds) == 8
    assert ds[7]["he"].shape == (3, 32, 32)
