#!/usr/bin/env python3
"""Save DAPI -> H&E qualitative comparisons from a trained paired checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity

from src.models.train_paired import UNetGenerator

THUMB = 384


def to_u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def gray_pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray(to_u8(x), mode="L").convert("RGB")


def rgb_pil(x: np.ndarray) -> Image.Image:
    return Image.fromarray(to_u8(np.transpose(x, (1, 2, 0))), mode="RGB")


def label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 24), "white")
    out.paste(img, (0, 24))
    ImageDraw.Draw(out).text((6, 6), text, fill="black")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--h5", default="data/processed/hemit_val_1024.h5")
    ap.add_argument("--out", default="docs/qualitative_dapi2he")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    direction = ckpt.get("config", {}).get("data", {}).get("direction", "he2dapi")
    if direction != "dapi2he":
        raise SystemExit(f"expected dapi2he checkpoint, got {direction}")

    G = UNetGenerator(in_ch=1, out_ch=3, ngf=64)
    G.load_state_dict(ckpt.get("G", ckpt.get("generator", ckpt)))
    G.eval()

    with h5py.File(args.h5, "r") as f:
        n = len(f["dapi"])
        rng = np.random.default_rng(args.seed)
        idxs = sorted(rng.choice(n, size=min(args.num, n), replace=False).tolist())
        rows = []
        summary = []
        for idx in idxs:
            dapi = np.asarray(f["dapi"][idx], dtype=np.float32) / 255.0
            he = np.asarray(f["he"][idx], dtype=np.float32).transpose(2, 0, 1) / 255.0
            with torch.no_grad():
                fake = G(torch.from_numpy(dapi[None, None]).float() * 2 - 1)[0].numpy()
            fake = np.clip(fake, 0, 1)

            l1 = float(np.mean(np.abs(fake - he)))
            mse = float(np.mean((fake - he) ** 2))
            psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
            ssim = float(structural_similarity(he, fake, data_range=1.0, channel_axis=0))
            pearson = float(np.corrcoef(he.ravel(), fake.ravel())[0, 1])
            row = {
                "index": int(idx),
                "l1": round(l1, 4),
                "psnr": round(psnr, 2),
                "ssim": round(ssim, 4),
                "pearson": round(pearson, 4),
                "real_mean_rgb": [round(float(v), 4) for v in he.mean(axis=(1, 2))],
                "fake_mean_rgb": [round(float(v), 4) for v in fake.mean(axis=(1, 2))],
                "real_std_rgb": [round(float(v), 4) for v in he.std(axis=(1, 2))],
                "fake_std_rgb": [round(float(v), 4) for v in fake.std(axis=(1, 2))],
            }
            summary.append(row)

            dp = gray_pil(dapi)
            rp = rgb_pil(he)
            fp = rgb_pil(fake)
            stem = f"val_{idx:04d}"
            dp.save(out / f"{stem}_dapi.png")
            rp.save(out / f"{stem}_real_he.png")
            fp.save(out / f"{stem}_fake_he.png")
            thumb = [label(im.resize((THUMB, THUMB)), t) for im, t in
                     [(dp, "DAPI input"), (rp, "real H&E"), (fp, "generated H&E")]]
            tile = Image.new("RGB", (THUMB * 3, THUMB + 24), "white")
            for j, im in enumerate(thumb):
                tile.paste(im, (j * THUMB, 0))
            rows.append(tile)

    grid = Image.new("RGB", (THUMB * 3, (THUMB + 24) * len(rows)), "white")
    for i, row in enumerate(rows):
        grid.paste(row, (0, i * (THUMB + 24)))
    grid.save(out / "overview.png")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[SAVED] {out / 'overview.png'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
