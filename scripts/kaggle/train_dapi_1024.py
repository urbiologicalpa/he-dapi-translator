#!/usr/bin/env python3
"""Kaggle T4 runner for corrected H&E -> DAPI (ch0) training.

Stages (set DAPI_STAGE):
  smoke     - 1 epoch / 20 steps at 512, timing only
  pretrain  - 40 epochs at 512 from the 1024px Macenko HDF5
  finetune  - 20 epochs at 1024 from the 512 checkpoint

Attach:
  bikasash/hemit-dapi-1024-ch0-macenko
  bikasash/he-dapi-code-dapi-1024
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

STAGE = os.environ.get("DAPI_STAGE", "smoke").lower()
ROOT = Path("/kaggle/working/he-dapi")
CKPT_ROOT = Path("/kaggle/working/checkpoints")
INPUT = Path("/kaggle/input")


def run(cmd, cwd=ROOT):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd))


def find_code() -> Path | None:
    print("INPUT dirs:", [p.name for p in sorted(INPUT.glob("*"))])
    for base in sorted(INPUT.glob("*")):
        if (base / "src/models/train_paired.py").exists():
            return base
        if (base / "code_bundle.zip").exists():
            return base
        if (base / "code_bundle/src/models/train_paired.py").exists():
            return base
    if Path("src/models/train_paired.py").exists():
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
    source = code / "code_bundle"
    if (source / "src/models/train_paired.py").exists():
        for item in source.iterdir():
            dst = ROOT / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        print(f"copied extracted code {source} -> {ROOT}")
        return
    for item in code.iterdir():
        if item.name in {".git", "data", "resources", "logs", "code_bundle"}:
            continue
        dst = ROOT / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)
    print(f"copied code {code} -> {ROOT}")


def link_data() -> Path:
    matches = list(INPUT.rglob("hemit_train_1024.h5"))
    if not matches:
        print("FATAL: hemit_train_1024.h5 not found under /kaggle/input")
        sys.exit(2)
    src_dir = matches[0].parent
    dst_dir = ROOT / "data/processed"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("hemit_train_1024.h5", "hemit_val_1024.h5", "hemit_test_1024.h5"):
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            print(f"WARN missing {src}")
            continue
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
        print(f"{name}: {src.stat().st_size / 1e9:.2f} GB")
    return dst_dir / "hemit_train_1024.h5"


def find_input_checkpoint() -> Path | None:
    for p in sorted(INPUT.rglob("checkpoint_epoch*.pt")):
        return p
    for p in sorted(INPUT.rglob("checkpoint_last.pt")):
        return p
    return None


def main() -> None:
    if STAGE not in {"smoke", "pretrain", "finetune"}:
        print(f"FATAL unknown DAPI_STAGE={STAGE}")
        sys.exit(2)

    print("nvidia-smi:")
    print(subprocess.getoutput("nvidia-smi | head -20"))
    code = find_code()
    if code is None:
        print("FATAL no code dataset/repo found")
        sys.exit(2)
    install_code(code)
    h5 = link_data()

    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    ckpt_dir = CKPT_ROOT / ("dapi_smoke" if STAGE == "smoke" else f"dapi_{STAGE}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if STAGE == "smoke":
        config = ROOT / "configs/hemit_dapi_512_pretrain.yaml"
        extra = ["--epochs", "1", "--max-steps", "20", "--batch-size", "2"]
    elif STAGE == "pretrain":
        config = ROOT / "configs/hemit_dapi_512_pretrain.yaml"
        incoming = find_input_checkpoint()
        if incoming and not any(ckpt_dir.glob("checkpoint_*.pt")):
            shutil.copy2(incoming, ckpt_dir / "checkpoint_last.pt")
            print(f"resume checkpoint {incoming} -> {ckpt_dir}")
        extra = ["--resume", "auto"]
    else:
        config = ROOT / "configs/hemit_dapi_1024.yaml"
        seed = CKPT_ROOT / "dapi_pretrain/checkpoint_last.pt"
        restart = seed.exists()
        if not restart:
            incoming = find_input_checkpoint()
            if incoming:
                seed = incoming
        if not any(ckpt_dir.glob("checkpoint_*.pt")) and seed.exists():
            shutil.copy2(seed, ckpt_dir / "checkpoint_last.pt")
            print(f"initialized 1024 from {seed}")
            if restart:
                extra = ["--resume", str(ckpt_dir / "checkpoint_last.pt"), "--resume-epoch", "0"]
            else:
                extra = ["--resume", "auto"]
        else:
            extra = ["--resume", "auto"]

    cmd = [
        sys.executable, "-u", "-m", "src.models.train_paired",
        "--config", str(config), "--optimize-for", "full",
        "--h5", str(h5), "--checkpoint-dir", str(ckpt_dir),
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    print("STAGE", STAGE, "CMD", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print("exit", result.returncode)
    for p in sorted(ckpt_dir.glob("*")):
        print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")
    if result.returncode:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
