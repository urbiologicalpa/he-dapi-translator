"""Tiny end-to-end smoke tests for both trainers.

Runs each trainer in its ``--dummy`` mode (self-contained, no data needed) and
asserts a clean exit. Marked ``slow`` because it builds the models and, for the
paired run, exercises the ONNX export. Skip when torch is unavailable.

Run locally:
    PYTHONPATH=. python -m pytest tests/test_training_dummy.py -v -m slow
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_dummy(module: str, config: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, "--config", config, "--dummy"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
    )


@pytest.mark.slow
def test_train_paired_dummy():
    pytest.importorskip("torch")
    result = _run_dummy("src.models.train_paired", "configs/hemit_dual_branch.yaml")
    assert result.returncode == 0, (result.stdout[-2000:] + result.stderr[-2000:])


@pytest.mark.slow
def test_train_paired_reverse_dummy():
    pytest.importorskip("torch")
    result = _run_dummy("src.models.train_paired", "configs/hemit_dapi2he_512_pretrain.yaml")
    assert result.returncode == 0, (result.stdout[-2000:] + result.stderr[-2000:])


@pytest.mark.slow
def test_train_unpaired_dummy():
    pytest.importorskip("torch")
    result = _run_dummy("src.models.train_unpaired", "configs/cyclegan_dapi2he.yaml")
    assert result.returncode == 0, (result.stdout[-2000:] + result.stderr[-2000:])
