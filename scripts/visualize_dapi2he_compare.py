#!/usr/bin/env python3
"""Build a DAPI / real H&E / paired / CycleGAN comparison grid (val crops)."""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def to_img(t):
    a = t.detach().cpu().float().numpy()
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = a.transpose(1, 2, 0)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def font(size=20):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cyclegan", required=True)
    ap.add_argument("--paired", required=True)
    ap.add_argument("--h5", default="data/processed/hemit_val_256.h5")
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--out", default="docs/qualitative_dapi2he/hemit_val_compare.png")
    a = ap.parse_args()

    from src.models.datasets import PairedHEMITDataset
    from src.models.train_paired import UNetGenerator
    from src.models.train_unpaired import ResnetGenerator

    ds = PairedHEMITDataset(h5_path=a.h5, patch_size=256, augment=False,
                            indices=range(a.num), max_cache_mb=4)

    cyc_state = torch.load(a.cyclegan, map_location="cpu")
    cyc_state = cyc_state.get("G", cyc_state)
    cyc = ResnetGenerator(1, 3, ngf=64, n_blocks=9)
    cyc.load_state_dict(cyc_state)
    cyc.eval()

    pck = torch.load(a.paired, map_location="cpu")
    p_state = pck.get("G", pck.get("generator", pck))
    if any(str(k).startswith("net.") for k in p_state):
        paired = ResnetGenerator(1, 3, ngf=64, n_blocks=9)
    else:
        paired = UNetGenerator(1, 3, ngf=64)
    paired.load_state_dict(p_state)
    paired.eval()

    tiles = []
    with torch.no_grad():
        for i in range(a.num):
            b = ds[i]
            dapi = b["dapi"].unsqueeze(0)
            real = (b["he"].unsqueeze(0) + 1) / 2
            src = dapi * 2 - 1
            cyc_out = cyc(src)
            pair_out = paired(src)
            tiles.append([
                ("DAPI", to_img(dapi)),
                ("real H&E", to_img(real)),
                ("paired U-Net", to_img(pair_out)),
                ("CycleGAN", to_img(cyc_out)),
            ])

    f = font(20)
    pad = 4
    w = h = 256
    header = 26
    cols, rows = 4, len(tiles)
    canvas = Image.new("RGB", (cols * (w + pad) + pad, rows * (h + pad + header) + pad),
                       (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for r, row in enumerate(tiles):
        for c, (label, im) in enumerate(row):
            x = pad + c * (w + pad)
            y = pad + r * (h + pad + header)
            d.text((x + 4, y + 3), f"{label} #{r}", font=f, fill=(0, 0, 0))
            canvas.paste(im.resize((w, h)), (x, y + header))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out, canvas.size)


if __name__ == "__main__":
    main()
