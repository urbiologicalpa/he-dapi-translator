#!/usr/bin/env python3
"""Build HEMIT HDF5 at native resolution with correct DAPI channel and Macenko H&E.

HEMIT label TIFF channels are ch0=DAPI, ch1=CD3, ch2=panCK. This script stores
``mihc`` (raw 3-channel) and ``dapi`` (ch0), and optionally applies Macenko
normalization to H&E only. It streams one image at a time so it can run on the
3.7 GB CPU box.

Usage:
  uv run --extra data python -m scripts.data.pack_hemit_1024 --splits train val test
  uv run --extra data python -m scripts.data.pack_hemit_1024 --splits val --limit 2 --out-dir /tmp
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

import h5py
import numpy as np
import tifffile

CHANNEL_ORDER = ("DAPI", "CD3", "panCK")
DAPI_CHANNEL = 0


def extract_dapi(mihc: np.ndarray) -> np.ndarray:
    if mihc.ndim != 3 or mihc.shape[-1] != 3:
        raise ValueError(f"mIHC must be HxWx3, got {mihc.shape}")
    return mihc[..., DAPI_CHANNEL]


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image {arr.shape} smaller than crop {size}")
    y0, x0 = (h - size) // 2, (w - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


def get_normalizer(enabled: bool):
    if not enabled:
        return None
    try:
        from torchstain.numpy.normalizers import NumpyMacenkoNormalizer
    except ImportError as exc:
        raise SystemExit(
            "Macenko requires torchstain. Install with: uv pip install torchstain "
            "or run with --no-macenko."
        ) from exc
    return NumpyMacenkoNormalizer()


def normalize_he(he: np.ndarray, normalizer) -> np.ndarray:
    if normalizer is None:
        return he
    result = normalizer.normalize(he, stains=False)
    return result[0] if isinstance(result, tuple) else result


def pair_files(root: Path, split: str) -> list[tuple[Path, Path]]:
    inp = root / split / "input"
    lab = root / split / "label"
    labels = {p.name: p for p in sorted(lab.glob("*.tif"))}
    pairs = []
    for he_p in sorted(inp.glob("*.tif")):
        lab_p = labels.get(he_p.name)
        if lab_p is None:
            raise FileNotFoundError(f"missing label for {he_p.name}")
        pairs.append((he_p, lab_p))
    if not pairs:
        raise FileNotFoundError(f"no pairs found in {inp}")
    return pairs


def build_split(
    root: Path,
    out_dir: Path,
    split: str,
    patch: int,
    normalizer,
    limit: int | None,
    overwrite: bool,
    compression: str,
    compression_level: int,
) -> Path:
    pairs = pair_files(root, split)
    if limit:
        pairs = pairs[:limit]
    out = out_dir / f"hemit_{split}_{patch}.h5"
    if out.exists() and not overwrite:
        with h5py.File(out, "r") as f:
            complete = (
                f.attrs.get("patch_size") == patch
                and int(f.attrs.get("dapi_channel", -1)) == DAPI_CHANNEL
                and f["he"].shape[0] == len(pairs)
            )
        if complete:
            print(f"[SKIP] {out} already complete")
            return out

    tmp = out.with_name(out.name + ".tmp")
    n = len(pairs)
    opts = {"compression": compression}
    if compression == "gzip":
        opts["compression_opts"] = compression_level
    with h5py.File(tmp, "w") as f:
        f.create_dataset(
            "he", (n, patch, patch, 3), dtype="uint8",
            chunks=(1, patch, patch, 3), **opts,
        )
        f.create_dataset(
            "mihc", (n, patch, patch, 3), dtype="uint8",
            chunks=(1, patch, patch, 3), **opts,
        )
        f.create_dataset(
            "dapi", (n, patch, patch), dtype="uint8",
            chunks=(1, patch, patch), **opts,
        )
        for i, (he_p, lab_p) in enumerate(pairs):
            he = tifffile.imread(str(he_p))
            mihc = tifffile.imread(str(lab_p))
            if he.ndim != 3 or mihc.ndim != 3:
                raise ValueError(f"expected RGB TIFFs, got {he.shape} and {mihc.shape}")
            if he.shape[0] != patch:
                he = center_crop(he, patch)
                mihc = center_crop(mihc, patch)
            he = normalize_he(he, normalizer)
            dapi = extract_dapi(mihc)
            f["he"][i] = he
            f["mihc"][i] = mihc
            f["dapi"][i] = dapi
            if (i + 1) % 100 == 0 or i + 1 == n:
                f.flush()
                gc.collect()
                print(f"  [{split}] {i + 1}/{n}")
        f.attrs["patch_size"] = patch
        f.attrs["num_pairs"] = n
        f.attrs["source"] = "HEMIT"
        f.attrs["channel_order"] = ",".join(CHANNEL_ORDER)
        f.attrs["dapi_channel"] = DAPI_CHANNEL
        f.attrs["macenko"] = normalizer is not None
        f.attrs["macenko_reference"] = "torchstain_default"
        f.attrs["compression"] = compression
        if limit:
            f.attrs["subset_limit"] = limit
    os.replace(tmp, out)
    print(f"[OK] {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("data/raw/hemit/extracted"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--patch-size", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-macenko", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compression", choices=["lzf", "gzip"], default="gzip")
    ap.add_argument("--compression-level", type=int, default=4)
    args = ap.parse_args()

    if args.patch_size > 1024:
        raise SystemExit("HEMIT patches are 1024x1024")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    normalizer = get_normalizer(not args.no_macenko)
    for split in args.splits:
        build_split(
            args.root, args.out_dir, split, args.patch_size,
            normalizer, args.limit, args.overwrite,
            args.compression, args.compression_level,
        )


if __name__ == "__main__":
    main()
