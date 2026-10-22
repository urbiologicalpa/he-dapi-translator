#!/usr/bin/env python3
"""Kaggle T4 runner for paired H&E <-> DAPI training.

Stages (set DAPI_STAGE):
  smoke     - 1 epoch / 20 steps at 512, timing only
  pretrain  - 20 epochs at 512 from the 1024px Macenko HDF5
  finetune  - 2 x 500 steps at 1024 from the 512 checkpoint
  all       - pretrain then finetune in one kernel (no checkpoint dataset needed)

Direction (set DAPI_DIRECTION):
  he2dapi   - default, H&E input -> DAPI ch0 output
  dapi2he   - reverse, DAPI ch0 input -> H&E output

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
DIRECTION = os.environ.get("DAPI_DIRECTION", "he2dapi").lower()
if DIRECTION not in {"he2dapi", "dapi2he"}:
    raise SystemExit(f"unknown DAPI_DIRECTION={DIRECTION}")
TAG = "dapi" if DIRECTION == "he2dapi" else "dapi2he"
ROOT = Path("/kaggle/working/he-dapi")
CKPT_ROOT = Path("/kaggle/working/checkpoints")
INPUT = Path("/kaggle/input")


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
        print("found bundle", bundles[0])
        return bundles[0].parent
    nested = list(INPUT.rglob("code_bundle/src/models/train_paired.py"))
    if nested:
        root = nested[0].parents[3]
        print("found nested code root", root)
        return root
    if Path("src/models/train_paired.py").exists():
        return Path.cwd()
    print("INPUT tree sample:", [str(p) for p in list(INPUT.rglob("*"))[:40]])
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
    last = sorted(INPUT.rglob("checkpoint_last.pt"))
    if last:
        return last[0]
    epochs: list[tuple[int, Path]] = []
    for p in INPUT.rglob("checkpoint_epoch*.pt"):
        try:
            n = int(p.stem.rsplit("epoch", 1)[-1])
        except ValueError:
            n = -1
        epochs.append((n, p))
    return max(epochs)[1] if epochs else None


def train_stage(stage: str, h5: Path) -> None:
    ckpt_dir = CKPT_ROOT / f"{TAG}_{stage}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if stage == "smoke":
        config = ROOT / f"configs/hemit_{TAG}_512_pretrain.yaml"
        extra = ["--epochs", "1", "--max-steps", "20", "--batch-size", "2"]
    elif stage == "pretrain":
        config = ROOT / f"configs/hemit_{TAG}_512_pretrain.yaml"
        incoming = find_input_checkpoint()
        if incoming and not any(ckpt_dir.glob("checkpoint_*.pt")):
            shutil.copy2(incoming, ckpt_dir / "checkpoint_last.pt")
            print(f"resume checkpoint {incoming} -> {ckpt_dir}")
        extra = ["--resume", "auto", "--epochs", "20"]
    elif stage == "finetune":
        config = ROOT / f"configs/hemit_{TAG}_1024.yaml"
        seed = CKPT_ROOT / f"{TAG}_pretrain/checkpoint_last.pt"
        if not seed.exists():
            incoming = find_input_checkpoint()
            if incoming:
                seed = incoming
        starting_fresh = not any(ckpt_dir.glob("checkpoint_*.pt"))
        if starting_fresh:
            if not seed.exists():
                print("FATAL fine-tune needs a pretrain checkpoint")
                sys.exit(2)
            shutil.copy2(seed, ckpt_dir / "checkpoint_last.pt")
            print(f"initialized 1024 from {seed}")
            # Pretrain checkpoint has a high epoch counter; reset it so fine-tune
            # actually runs epochs 1..2 at the lower 1024 LR.
            extra = ["--resume", str(ckpt_dir / "checkpoint_last.pt"), "--resume-epoch", "0", "--epochs", "2", "--max-steps", "500"]
        else:
            extra = ["--resume", "auto", "--epochs", "2", "--max-steps", "500"]
    else:
        raise SystemExit(f"unknown stage={stage}")

    cmd = [
        sys.executable, "-u", "-m", "src.models.train_paired",
        "--config", str(config), "--optimize-for", "full",
        "--h5", str(h5), "--checkpoint-dir", str(ckpt_dir),
        "--direction", DIRECTION,
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    print("STAGE", stage, "DIRECTION", DIRECTION, "CMD", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(ROOT), env=env)
    print("exit", result.returncode)
    for p in sorted(ckpt_dir.glob("*")):
        print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")
    if result.returncode:
        raise SystemExit(result.returncode)


def main() -> None:
    if STAGE not in {"smoke", "pretrain", "finetune", "all"}:
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

    stages = ["pretrain", "finetune"] if STAGE == "all" else [STAGE]
    for stage in stages:
        train_stage(stage, h5)


if __name__ == "__main__":
    main()
