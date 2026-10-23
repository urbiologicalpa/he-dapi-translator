#!/usr/bin/env python3
"""Crop HEMIT HDF5 splits to a smaller training size with the correct DAPI.

DAPI is the blue channel: ``mihc[...,2]``. The script deliberately ignores a
stored ``dapi`` cache if it is wrong (pre-audit 1024 H5 used ch0/panCK). It
streams each source image once and writes 256px ``he``/``dapi`` patches with
fast LZF compression, so training reads ~15x fewer bytes per sample.

Usage:
  python -m scripts.data.crop_hemit_h5 --src data/processed/hemit_train_1024.h5 \
      --out data/processed/hemit_train_256.h5 --size 256
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import h5py
import numpy as np


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image {arr.shape} smaller than crop {size}")
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


def is_complete(path: Path, n: int, size: int) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return (
                int(f.attrs.get("patch_size", -1)) == size
                and "he" in f and "dapi" in f
                and int(f.attrs.get("dapi_channel", -1)) == 2
                and f["he"].shape == (n, size, size, 3)
                and f["dapi"].shape == (n, size, size)
            )
    except Exception:
        return False


def build(src: Path, out: Path, size: int, compression: str, overwrite: bool) -> Path:
    with h5py.File(src, "r", libver="latest", swmr=True, rdcc_nbytes=64 * 1024 * 1024) as f:
        src_attrs = dict(f.attrs)
        n = int(f["he"].shape[0])
        if is_complete(out, n, size) and not overwrite:
            print(f"[SKIP] {out} already complete")
            return out

        tmp = out.with_name(out.name + ".tmp")
        with h5py.File(tmp, "w") as d:
            opts = {"compression": compression, "shuffle": True} if compression else {}
            d.create_dataset("he", (n, size, size, 3), dtype="uint8",
                             chunks=(1, size, size, 3), **opts)
            d.create_dataset("dapi", (n, size, size), dtype="uint8",
                             chunks=(1, size, size), **opts)
            has_mihc = "mihc" in f
            for i in range(n):
                he = center_crop(f["he"][i], size)
                if has_mihc:
                    dapi = center_crop(f["mihc"][i][..., 2], size)
                else:
                    dapi = center_crop(f["dapi"][i], size)
                d["he"][i] = he
                d["dapi"][i] = dapi
                if (i + 1) % 200 == 0 or i + 1 == n:
                    d.flush()
                    gc.collect()
                    print(f"  [{src.name}] {i + 1}/{n}", flush=True)
            for k, v in src_attrs.items():
                if k not in {"num_pairs"}:
                    d.attrs[k] = v
            d.attrs["patch_size"] = size
            d.attrs["num_pairs"] = n
            d.attrs["dapi_channel"] = 2
            d.attrs["channel_order"] = "panCK,CD3,DAPI"
            d.attrs["derived_from"] = src.name
            d.attrs["compression"] = compression
        os.replace(tmp, out)
    print(f"[OK] {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    build(args.src, args.out, args.size, args.compression, args.overwrite)


if __name__ == "__main__":
    main()
