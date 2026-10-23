#!/usr/bin/env python3
"""Kaggle GPU eval for paired H&E <-> DAPI at native 1024px.

Set DAPI_DIRECTION=he2dapi (default) or dapi2he. The direction is also read
from the checkpoint config, so a matching checkpoint is sufficient.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working/he-dapi")
INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
DIRECTION = os.environ.get("DAPI_DIRECTION", "he2dapi").lower()
if DIRECTION not in {"he2dapi", "dapi2he"}:
    raise SystemExit(f"unknown DAPI_DIRECTION={DIRECTION}")
TAG = "dapi" if DIRECTION == "he2dapi" else "dapi2he"


def run(cmd, cwd=ROOT):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd))


def find_code() -> Path | None:
    print("INPUT top:", [p.name for p in sorted(INPUT.glob("*"))])
    matches = list(INPUT.rglob("src/models/train_paired.py"))
    if matches:
        root = matches[0].parents[2]
        print("found code root", root)
        return root
    bundles = list(INPUT.rglob("code_bundle.zip"))
    if bundles:
        return bundles[0].parent
    return None


def install_code(code: Path) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    bundle = code / "code_bundle.zip"
    if bundle.exists():
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(ROOT)
        print(f"extracted {bundle} -> {ROOT}")
        return
    source = code if (code / "src/models/train_paired.py").exists() else code / "code_bundle"
    if (source / "src/models/train_paired.py").exists():
        for item in source.iterdir():
            dst = ROOT / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        print(f"copied {source} -> {ROOT}")
        return
    raise SystemExit(f"code root has no source: {code}")


def find_one(pattern: str) -> Path:
    matches = sorted(INPUT.rglob(pattern))
    if not matches:
        raise SystemExit(f"missing {pattern} under /kaggle/input")
    return matches[0]


def main() -> None:
    print(subprocess.getoutput("nvidia-smi | head -20"))
    code = find_code()
    if code is None:
        raise SystemExit("no code dataset found")
    install_code(code)

    ckpt = find_one("checkpoint_last.pt")
    print("checkpoint", ckpt, f"{ckpt.stat().st_size / 1e6:.1f} MB")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    results = {}
    for split, num in (("val", 630), ("test", 945)):
        h5 = find_one(f"hemit_{split}_1024.h5")
        out = OUT / f"metrics_{TAG}_1024_{split}.json"
        cmd = [
            sys.executable, "-u", "-m", "src.models.eval_offline",
            "--checkpoint", str(ckpt), "--h5", str(h5), "--split", split,
            "--num", str(num), "--patch-size", "1024",
            "--target", "DAPI ch2" if DIRECTION == "he2dapi" else "H&E RGB",
            "--label", f"{TAG}-1024-unet",
            "--direction", DIRECTION,
            "--out", str(out),
        ]
        result = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env)
        if result.returncode:
            raise SystemExit(f"eval {split} failed: {result.returncode}")
        results[split] = json.loads(out.read_text())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
