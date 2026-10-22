#!/usr/bin/env python3
"""CPU-only Kaggle evaluator for a checkpoint produced by another kernel.

Environment overrides:
  EVAL_DIRECTION  he2dapi | dapi2he        (default dapi2he)
  EVAL_PATCH      patch size                (default 256)
  EVAL_NUM        images evaluated          (default 64)
  EVAL_METRIC_NUM images used for FID/LPIPS (default 50)
  EVAL_CKPT_GLOB  checkpoint filename       (default cyclegan_G_last.pt)

Attach the training kernel via `kernel_sources`; its output is searched for the
checkpoint. Attach `bikasash/hemit-dapi-1024-ch0-macenko` for the val HDF5.
Internet must be enabled so torchvision/lpips can fetch their weights.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DIRECTION = os.environ.get("EVAL_DIRECTION", "dapi2he")
PATCH = os.environ.get("EVAL_PATCH", "256")
NUM = os.environ.get("EVAL_NUM", "64")
METRIC_NUM = os.environ.get("EVAL_METRIC_NUM", "50")
CKPT_GLOB = os.environ.get("EVAL_CKPT_GLOB", "cyclegan_G_last.pt")
ROOT = Path("/kaggle/working/he-dapi")
INPUT = Path("/kaggle/input")


def find_code() -> Path | None:
    print("INPUT top:", [p.name for p in sorted(INPUT.glob("*"))])
    for needle in ("src/models/eval_offline.py", "src/models/train_paired.py"):
        matches = list(INPUT.rglob(needle))
        if matches:
            return matches[0].parents[2]
    bundles = list(INPUT.rglob("code_bundle.zip"))
    if bundles:
        return bundles[0].parent
    if Path("src/models/eval_offline.py").exists():
        return Path.cwd()
    return None


def install_code(code: Path) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    bundle = code / "code_bundle.zip"
    if bundle.exists():
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(ROOT)
        return
    source = code / "code_bundle"
    if not (source / "src/models/eval_offline.py").exists():
        source = code
    for item in source.iterdir():
        if item.name in {".git", "data", "resources", "logs", "code_bundle"}:
            continue
        dst = ROOT / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)


def main() -> None:
    code = find_code()
    if code is None:
        raise SystemExit("FATAL no code dataset found")
    install_code(code)

    ckpts = sorted(INPUT.rglob(CKPT_GLOB))
    if not ckpts:
        raise SystemExit(f"FATAL no {CKPT_GLOB} under /kaggle/input")
    ckpt = ckpts[0]
    h5s = sorted(INPUT.rglob("hemit_val_1024.h5"))
    if not h5s:
        raise SystemExit("FATAL no hemit_val_1024.h5 under /kaggle/input")
    print("CKPT", ckpt, f"{ckpt.stat().st_size / 1e6:.1f} MB")
    print("VAL ", h5s[0])

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lpips"], check=False)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    out = Path("/kaggle/working/metrics_eval.json")
    cmd = [
        sys.executable, "-u", "-m", "src.models.eval_offline",
        "--checkpoint", str(ckpt), "--direction", DIRECTION,
        "--h5", str(h5s[0]), "--patch-size", PATCH, "--num", NUM,
        "--metrics", "fid,lpips", "--metric-num", METRIC_NUM,
        "--label", os.environ.get("EVAL_LABEL", f"{DIRECTION}-quality-1024"),
        "--out", str(out),
    ]
    print("CMD", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if out.exists():
        print(out.read_text())
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
