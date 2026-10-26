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


def find_one_optional(pattern: str) -> Path | None:
    matches = sorted(INPUT.rglob(pattern))
    return matches[0] if matches else None


def find_one(pattern: str) -> Path:
    path = find_one_optional(pattern)
    if path is None:
        raise SystemExit(f"missing {pattern} under /kaggle/input")
    return path


def _checkpoint_epoch(path: Path) -> int | None:
    try:
        import torch

        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch")
        return int(epoch) if epoch is not None else None
    except Exception:
        return None


def find_checkpoints() -> list[tuple[str, Path]]:
    """Prefer best-selected checkpoints; fall back to last.

    ``checkpoint_best.pt`` and ``checkpoint_last.pt`` are often the same epoch
    (best == last). Evaluating both doubles GPU time, so skip ``last`` when it
    holds the same epoch.
    """
    found: list[tuple[str, Path]] = []
    best = sorted(INPUT.rglob("checkpoint_best.pt"))
    last = sorted(INPUT.rglob("checkpoint_last.pt"))
    if best:
        found.append(("best", best[0]))
    if last:
        best_epoch = _checkpoint_epoch(best[0]) if best else None
        last_epoch = _checkpoint_epoch(last[0])
        same_epoch = best_epoch is not None and best_epoch == last_epoch
        if not same_epoch:
            found.append(("last", last[0]))
        else:
            print(f"checkpoint last == best epoch ({last_epoch}); skipping duplicate eval")
    if not found:
        raise SystemExit("missing checkpoint_best.pt / checkpoint_last.pt under /kaggle/input")
    return found


def main() -> None:
    print(subprocess.getoutput("nvidia-smi | head -20"))
    code = find_code()
    if code is None:
        raise SystemExit("no code dataset found")
    install_code(code)

    checkpoints = find_checkpoints()
    for tag, ckpt in checkpoints:
        print("checkpoint", tag, ckpt, f"{ckpt.stat().st_size / 1e6:.1f} MB")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    results = {}
    for ckpt_tag, ckpt in checkpoints:
        # Run test only for the selected checkpoint (or last when no best).
        for split, num in (("val", 630), ("test", 945)):
            if split == "test" and ckpt_tag != checkpoints[0][0]:
                continue
            h5 = find_one_optional(f"hemit_{split}_1024.h5")
            if h5 is None:
                print(f"[EVAL] skip {split}: hemit_{split}_1024.h5 not attached")
                continue
            out = OUT / f"metrics_{TAG}_1024_{split}_{ckpt_tag}.json"
            cmd = [
                sys.executable, "-u", "-m", "src.models.eval_offline",
                "--checkpoint", str(ckpt), "--h5", str(h5), "--split", split,
                "--num", str(num), "--patch-size", "1024",
                "--metric-num", "50", "--metrics", "fid,lpips",
                "--target", "DAPI ch2" if DIRECTION == "he2dapi" else "H&E RGB",
                "--label", f"{TAG}-1024-unet-{ckpt_tag}",
                "--direction", DIRECTION,
                "--out", str(out),
            ]
            result = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), env=env)
            if result.returncode:
                raise SystemExit(f"eval {ckpt_tag} {split} failed: {result.returncode}")
            results[f"{ckpt_tag}_{split}"] = json.loads(out.read_text())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
