#!/usr/bin/env python3
"""Kaggle T4 runner for the corrected HEMIT DAPI -> H&E CycleGAN.

Attach:
  bikasash/hemit-dapi-1024-ch0-macenko  (1024 H5; cropped to 256 in-kernel)
  bikasash/he-dapi-code-dapi-1024       (code bundle, updated 2026-10-22)

Why crop in-kernel: the 1024 gzip H5 is ~6 s/step on CPU; a 256 LZF H5 is
~15x fewer bytes. Cropping once costs a few minutes, then 10 T4 epochs are
fast. The cropped data is identical to the local
data/processed/hemit_{train,val}_256.h5.

Checkpoints go to /kaggle/working/checkpoints/hemit_cyclegan_dapi2he_256/.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working/he-dapi")
CKPT_DIR = Path("/kaggle/working/checkpoints/hemit_cyclegan_dapi2he_256")
INPUT = Path("/kaggle/input")
PROCESSED = ROOT / "data/processed"


def find_code() -> Path | None:
    print("INPUT top:", [p.name for p in sorted(INPUT.glob("*"))])
    matches = list(INPUT.rglob("src/models/train_unpaired.py"))
    if matches:
        return matches[0].parents[2]
    bundles = list(INPUT.rglob("code_bundle.zip"))
    if bundles:
        return bundles[0].parent
    nested = list(INPUT.rglob("code_bundle/src/models/train_unpaired.py"))
    if nested:
        return nested[0].parents[3]
    if Path("src/models/train_unpaired.py").exists():
        return Path.cwd()
    return None


def install_code(code: Path) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    bundle = code / "code_bundle.zip"
    if bundle.exists():
        with zipfile.ZipFile(bundle) as zf:
            zf.extractall(ROOT)
        print(f"extracted {bundle} -> {ROOT}")
        return
    src = code / "code_bundle"
    source = src if (src / "src/models/train_unpaired.py").exists() else code
    if not (source / "src/models/train_unpaired.py").exists():
        raise SystemExit(f"no source under {code}")
    for item in source.iterdir():
        if item.name in {".git", "data", "resources", "logs", "code_bundle"}:
            continue
        dst = ROOT / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    print(f"copied {source} -> {ROOT}")


def link_inputs() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    wanted = {
        "hemit_train_1024.h5": "train",
        "hemit_val_1024.h5": "val",
    }
    for name, split in wanted.items():
        matches = sorted(INPUT.rglob(name))
        if not matches:
            raise SystemExit(f"FATAL missing {name} under /kaggle/input")
        dst = PROCESSED / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(matches[0])
        print(f"{name}: {matches[0].stat().st_size / 1e9:.2f} GB ({split})")


def crop_256(env: dict[str, str]) -> None:
    for split in ("train", "val"):
        src = PROCESSED / f"hemit_{split}_1024.h5"
        out = PROCESSED / f"hemit_{split}_256.h5"
        if out.exists() and out.stat().st_size > 0:
            print(f"[SKIP] {out} ({out.stat().st_size / 1e9:.2f} GB)")
            continue
        cmd = [
            sys.executable, "-u", str(ROOT / "scripts/data/crop_hemit_h5.py"),
            "--src", str(src), "--out", str(out), "--size", "256",
            "--compression", "lzf",
        ]
        print("CROP", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def main() -> None:
    print(subprocess.getoutput("nvidia-smi | head -20"))
    code = find_code()
    if code is None:
        raise SystemExit("FATAL no code dataset found")
    install_code(code)
    link_inputs()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    crop_256(env)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", "-m", "src.models.train_unpaired",
        "--config", str(ROOT / "configs/cyclegan_hemit_dapi2he_kaggle.yaml"),
        "--optimize-for", "full", "--resume", "auto",
        "--epochs", "10",
        "--checkpoint-dir", str(CKPT_DIR),
    ]
    print("CMD", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print("exit", result.returncode, flush=True)

    ckpt = CKPT_DIR / "cyclegan_G_last.pt"
    if result.returncode == 0 and ckpt.exists():
        eval_cmd = [
            sys.executable, "-u", "-m", "src.models.eval_offline",
            "--checkpoint", str(ckpt), "--direction", "dapi2he",
            "--h5", str(PROCESSED / "hemit_val_256.h5"),
            "--patch-size", "256", "--num", "630",
            "--label", "cyclegan-hemit-dapi2he-256",
            "--target", "H&E (HEMIT unpaired)",
            "--out", str(CKPT_DIR / "eval_cyclegan.json"),
        ]
        print("EVAL", " ".join(eval_cmd), flush=True)
        subprocess.run(eval_cmd, cwd=str(ROOT), env=env)

    for p in sorted(CKPT_DIR.glob("*")):
        print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")
    if result.returncode:
        raise SystemExit(result.returncode)
