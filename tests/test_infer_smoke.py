
"""CPU smoke test for the inference command."""
import subprocess
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize(
    ("direction", "expected_shape"),
    [
        ("he2dapi", "(64, 64)"),
        ("dapi2he", "(64, 64, 3)"),
    ],
)
def test_infer_dummy_smoke(direction, expected_shape):
    result = subprocess.run(
        [sys.executable, "-m", "src.models.infer", "--dummy","--direction", direction],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        result.stdout[-2000:] + result.stderr[-2000:]
    )
    assert "[SMOKE OK]" in result.stdout
    assert f"shape={expected_shape}" in result.stdout
    assert "dtype=float32" in result.stdout
    assert "range=[" in result.stdout
