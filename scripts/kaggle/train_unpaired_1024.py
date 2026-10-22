#!/usr/bin/env python3
"""Kaggle T4 runner for unpaired DAPI -> H&E CycleGAN.

Attach:
  bikasash/hemit-dapi-1024-ch0-macenko  (DAPI domain, train split)
  bikasash/mhist-he-512                 (external H&E domain)
  bikasash/he-dapi-code-dapi-1024       (code bundle)

Checkpoints go to /kaggle/working/checkpoints/dapi2he_unpaired/.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working/he-dapi")
CKPT_DIR = Path("/kaggle/working/checkpoints/dapi2he_unpaired")
INPUT = Path("/kaggle/input")


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
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    wanted = {
        "hemit_train_1024.h5": "dapi_h5_fallback",
        "mhist_256.h5": "he_h5_fallback",
        "hemit_val_1024.h5": "val",
    }
    for name in wanted:
        matches = sorted(INPUT.rglob(name))
        if not matches:
            raise SystemExit(f"FATAL missing {name} under /kaggle/input")
        dst = processed / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(matches[0])
        print(f"{name}: {matches[0].stat().st_size / 1e9:.2f} GB")


def main() -> None:
    print(subprocess.getoutput("nvidia-smi | head -20"))
    code = find_code()
    if code is None:
        raise SystemExit("FATAL no code dataset found")
    install_code(code)
    link_inputs()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-u", "-m", "src.models.train_unpaired",
        "--config", str(ROOT / "configs/cyclegan_dapi2he.yaml"),
        "--optimize-for", "full", "--resume", "auto",
        "--batch-size", "4", "--grad-accum", "2",
        "--checkpoint-dir", str(CKPT_DIR),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    print("CMD", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print("exit", result.returncode)

    ckpt = CKPT_DIR / "cyclegan_G_last.pt"
    if result.returncode == 0 and ckpt.exists():
        eval_cmd = [
            sys.executable, "-u", "-m", "src.models.eval_offline",
            "--checkpoint", str(ckpt), "--direction", "dapi2he",
            "--h5", str(ROOT / "data/processed/hemit_val_1024.h5"),
            "--patch-size", "256", "--num", "64",
            "--label", "cyclegan-dapi2he-1024",
            "--target", "H&E (MHIST external unpaired)",
            "--out", str(CKPT_DIR / "eval_cyclegan.json"),
        ]
        print("EVAL", " ".join(eval_cmd), flush=True)
        subprocess.run(eval_cmd, cwd=str(ROOT), env=env)

    for p in sorted(CKPT_DIR.glob("*")):
        print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
