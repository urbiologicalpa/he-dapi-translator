"""
eval_offline.py - Low-RAM read-only eval, NEVER disturbs running training.
Computes on any checkpoint: L1, PSNR, SSIM-loss(proxy), Dice(thresh 0.1), D-loss.
Usage:
  PYTHONPATH=. .venv/bin/python -m src.models.eval_offline --checkpoint resources/models/checkpoints/he2dapi/checkpoint_epoch53.pt --split val --num 30
Saves JSON to docs/metrics_eval.json when --save, or one labeled file with --out.
Low-RAM: batch 1, CPU, no_grad, SWMR, gc every 10, max_cache 4MB.
"""
import argparse
import json
import math
import gc
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity


def evaluate(checkpoint_path: str, split: str = "val", num: int = 30,
             h5_path: str = None, patch_size: int = 512, ngf: int = 64,
             data_root: str = "data/raw/hemit"):
    from src.models.train_paired import UNetGenerator, PatchGANDiscriminator
    from src.models.datasets import PairedHEMITDataset
    from src.models.losses import SSIMLoss, NucleiDiceLoss

    h5_map = {
        "val": "data/processed/hemit_val_512.h5",
        "train": "data/processed/hemit_train_512.h5",
        "test": "data/processed/hemit_test_512.h5",
    }
    h5_path = h5_path or h5_map.get(split, h5_map["val"])
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    file_epoch = ckpt.get("epoch", 0) + 1 if isinstance(ckpt, dict) and "epoch" in ckpt else 0
    train_gloss = float(ckpt.get("loss", 0)) if isinstance(ckpt, dict) else 0.0

    G = UNetGenerator(ngf=ngf)
    G.load_state_dict(ckpt.get("G", ckpt.get("generator", ckpt)))
    G.eval()
    D = PatchGANDiscriminator()
    if "D" in ckpt:
        D.load_state_dict(ckpt["D"])
    D.eval()

    ds = PairedHEMITDataset(
        h5_path=h5_path, root_dir=data_root, patch_size=patch_size,
        augment=False, indices=range(num), max_cache_mb=4,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    ssim_fn = SSIMLoss()
    dice_fn = NucleiDiceLoss(thresh=0.1)  # fixed 2026-10-03 (0.5 was vacuous)
    l1_fn = nn.L1Loss()
    mse_fn = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    L1 = SE_PSNR = SS = DICE = DL = 0.0
    SSIM_REAL = PEARSON = 0.0
    PEARSON_N = 0
    n = 0
    with torch.no_grad():
        for b in loader:
            he = b["he"]
            real = b["dapi"]
            fake = G(he)
            he01 = (he + 1) / 2
            L1 += l1_fn(fake, real).item()
            mse = mse_fn(fake, real).item()
            SE_PSNR += 10 * math.log10(1 / max(mse, 1e-12))
            SS += ssim_fn(fake, real).item()
            fake_np = fake[0, 0].detach().cpu().numpy()
            real_np = real[0, 0].detach().cpu().numpy()
            SSIM_REAL += structural_similarity(real_np, fake_np, data_range=1.0)
            corr = float(np.corrcoef(real_np.ravel(), fake_np.ravel())[0, 1])
            if np.isfinite(corr):
                PEARSON += corr
                PEARSON_N += 1
            DICE += (1 - dice_fn(fake, real).item())
            rp = torch.cat([he01, real], 1)
            fp = torch.cat([he01, fake], 1)
            pr = D(rp)
            pf = D(fp)
            DL += ((bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf))) * 0.5).item()
            n += 1
            if n % 10 == 0:
                gc.collect()
            del he, real, fake
    gc.collect()
    return {
        "checkpoint": str(checkpoint_path),
        "file_epoch": file_epoch,
        "split": split,
        "resolution": patch_size,
        "n": n,
        "train_G_loss": round(train_gloss, 4),
        "val_L1": round(L1 / n, 4),
        "val_PSNR_dB": round(SE_PSNR / n, 2),
        "val_SSIM": round(SSIM_REAL / n, 4),
        "val_Pearson": round(PEARSON / PEARSON_N, 4) if PEARSON_N else None,
        "val_SSIM_loss_proxy": round(SS / n, 4),
        "val_Dice_thresh01": round(DICE / n, 4),
        "val_D_loss": round(DL / n, 4),
        "note": "SSIM/Pearson are per-image skimage/numpy; Dice thresh 0.1; D-loss ~0.69=balanced",
        "evaluated": datetime.now(timezone.utc).isoformat(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--save", action="store_true", help="append to docs/metrics_eval.json")
    ap.add_argument("--out", type=str, default=None, help="write a single labeled result JSON to this path")
    ap.add_argument("--label", type=str, default=None, help="experiment label, e.g. panck-ch2-ablation")
    ap.add_argument("--target", type=str, default=None, help="target definition, e.g. 'DAPI ch0' or 'panCK ch2'")
    ap.add_argument("--h5", type=str, default=None, help="override HDF5 path")
    ap.add_argument("--patch-size", type=int, default=512)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--data-root", type=str, default="data/raw/hemit")
    args = ap.parse_args()
    res = evaluate(args.checkpoint, args.split, args.num, h5_path=args.h5,
                   patch_size=args.patch_size, ngf=args.ngf, data_root=args.data_root)
    if args.label:
        res["label"] = args.label
    if args.target:
        res["target"] = args.target
    print(json.dumps(res, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"[SAVED] {out}")
    elif args.save:
        out = Path("docs/metrics_eval.json")
        hist = []
        if out.exists():
            try:
                hist = json.loads(out.read_text())
            except Exception:
                hist = []
        hist = [h for h in hist if not (h.get("checkpoint") == res["checkpoint"] and h.get("split") == res["split"] and h.get("n") == res["n"])]
        hist.append(res)
        hist.sort(key=lambda h: h.get("file_epoch", 0))
        out.write_text(json.dumps(hist, indent=2))
        print(f"[SAVED] {out} ({len(hist)} entries)")


if __name__ == "__main__":
    main()
