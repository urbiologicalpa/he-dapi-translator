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


def _as_3ch(x: torch.Tensor) -> torch.Tensor:
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


class _InceptionFID:
    """Inception pool3 (2048-d) features for Frechet distance, torchvision only."""

    def __init__(self):
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                  transform_input=False, aux_logits=True)
        self.model.fc = nn.Identity()
        self.model.eval()

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> np.ndarray:
        x = _as_3ch(x.float())
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode="bilinear",
                                            align_corners=False)
        x = x * 2 - 1  # [0,1] -> [-1,1], the standard FID input range
        return self.model(x).cpu().numpy()


def _frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    mu1, mu2 = a.mean(0), b.mean(0)
    sigma1 = np.cov(a, rowvar=False)
    sigma2 = np.cov(b, rowvar=False)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(sigma1 + sigma2 - 2 * covmean))


def evaluate(checkpoint_path: str, split: str = "val", num: int = 30,
             h5_path: str = None, patch_size: int = 512, ngf: int = 64,
             data_root: str = "data/raw/hemit", direction: str = None,
             metrics: list[str] | None = None, metric_num: int = 50):
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
    if direction is None:
        direction = ckpt.get("config", {}).get("data", {}).get("direction", "he2dapi")
    if direction not in {"he2dapi", "dapi2he"}:
        raise ValueError(f"unknown direction: {direction}")
    in_ch, out_ch = (3, 1) if direction == "he2dapi" else (1, 3)

    state = ckpt.get("G", ckpt.get("generator", ckpt))
    if any(str(k).startswith("net.") for k in state):
        from src.models.train_unpaired import ResnetGenerator
        n_blocks = ckpt.get("config", {}).get("model", {}).get("generator", {}).get("n_blocks", 9)
        G = ResnetGenerator(in_ch=in_ch, out_ch=out_ch, ngf=ngf, n_blocks=n_blocks)
    else:
        G = UNetGenerator(in_ch=in_ch, out_ch=out_ch, ngf=ngf)
    G.load_state_dict(state)
    G.eval()
    D = PatchGANDiscriminator(in_ch=4)
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

    wanted = {m.strip().lower() for m in (metrics or []) if m}
    fid = None
    if "fid" in wanted:
        try:
            fid = _InceptionFID()
        except Exception as exc:  # offline kernel without pretrained weights
            print(f"[METRICS] FID unavailable ({exc}); skipping")
            wanted.discard("fid")
    lpips_fn = None
    if "lpips" in wanted:
        try:
            import lpips  # noqa: F401
            lpips_fn = lpips.LPIPS(net="alex")
        except Exception as exc:  # optional dependency / offline weights
            print(f"[METRICS] LPIPS unavailable ({exc}); skipping")
            wanted.discard("lpips")
    fid_real: list[np.ndarray] = []
    fid_fake: list[np.ndarray] = []
    lpips_vals: list[float] = []
    metric_seen = 0

    L1 = SE_PSNR = SS = DICE = DL = 0.0
    SSIM_REAL = PEARSON = 0.0
    PEARSON_N = 0
    n = 0
    with torch.no_grad():
        for b in loader:
            he = b["he"]
            dapi = b["dapi"]
            if direction == "he2dapi":
                src, src01, real = he, (he + 1) / 2, dapi
            else:
                src, src01, real = dapi * 2 - 1, dapi, (he + 1) / 2
            fake = G(src)
            L1 += l1_fn(fake, real).item()
            mse = mse_fn(fake, real).item()
            SE_PSNR += 10 * math.log10(1 / max(mse, 1e-12))
            SS += ssim_fn(fake, real).item()
            fake_np = fake[0].detach().cpu().numpy()
            real_np = real[0].detach().cpu().numpy()
            if metric_seen < metric_num:
                if fid is not None:
                    fid_real.append(fid.features(real))
                    fid_fake.append(fid.features(fake))
                if lpips_fn is not None:
                    lp = lpips_fn(_as_3ch(fake * 2 - 1), _as_3ch(real * 2 - 1))
                    lpips_vals.append(float(lp.item()))
                metric_seen += 1
            if fake_np.shape[0] == 1:
                SSIM_REAL += float(structural_similarity(real_np[0], fake_np[0], data_range=1.0))
            else:
                SSIM_REAL += float(structural_similarity(real_np, fake_np, data_range=1.0, channel_axis=0))
            corr = float(np.corrcoef(real_np.ravel(), fake_np.ravel())[0, 1])
            if np.isfinite(corr):
                PEARSON += corr
                PEARSON_N += 1
            if direction == "he2dapi":
                DICE += (1 - dice_fn(fake, real).item())
            rp = torch.cat([src01, real], 1)
            fp = torch.cat([src01, fake], 1)
            pr = D(rp)
            pf = D(fp)
            DL += ((bce(pr, torch.ones_like(pr)) + bce(pf, torch.zeros_like(pf))) * 0.5).item()
            n += 1
            if n % 10 == 0:
                gc.collect()
            del he, dapi, src, src01, real, fake
    gc.collect()
    return {
        "checkpoint": str(checkpoint_path),
        "file_epoch": file_epoch,
        "split": split,
        "direction": direction,
        "resolution": patch_size,
        "n": n,
        "train_G_loss": round(train_gloss, 4),
        "val_L1": round(L1 / n, 4),
        "val_PSNR_dB": round(SE_PSNR / n, 2),
        "val_SSIM": round(SSIM_REAL / n, 4),
        "val_Pearson": round(PEARSON / PEARSON_N, 4) if PEARSON_N else None,
        "val_SSIM_loss_proxy": round(SS / n, 4),
        "val_Dice_thresh01": round(DICE / n, 4) if direction == "he2dapi" else None,
        "val_D_loss": round(DL / n, 4),
        "val_FID": (round(_frechet_distance(np.concatenate(fid_real),
                                           np.concatenate(fid_fake)), 2)
                    if len(fid_real) >= 2 else None),
        "val_LPIPS": round(float(np.mean(lpips_vals)), 4) if lpips_vals else None,
        "metric_n": metric_seen,
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
    ap.add_argument("--target", type=str, default=None, help="target definition, e.g. 'DAPI ch2' or 'panCK ch0'")
    ap.add_argument("--h5", type=str, default=None, help="override HDF5 path")
    ap.add_argument("--patch-size", type=int, default=512)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--data-root", type=str, default="data/raw/hemit")
    ap.add_argument("--direction", type=str, default=None, choices=["he2dapi", "dapi2he"])
    ap.add_argument("--metrics", type=str, default="", help="extra metrics, e.g. fid,lpips")
    ap.add_argument("--metric-num", type=int, default=50, help="images used for FID/LPIPS")
    args = ap.parse_args()
    res = evaluate(args.checkpoint, args.split, args.num, h5_path=args.h5,
                   patch_size=args.patch_size, ngf=args.ngf, data_root=args.data_root,
                   direction=args.direction,
                   metrics=[m for m in args.metrics.split(",") if m],
                   metric_num=args.metric_num)
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
