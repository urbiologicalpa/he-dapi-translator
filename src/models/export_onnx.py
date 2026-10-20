"""Export the trained paired generator to ONNX after training finishes.

Keeping export out of the training process avoids a memory spike on the
3.7GB CPU box at epoch 200.

Run:
  PYTHONPATH=. .venv/bin/python -m src.models.export_onnx
"""
import argparse
from pathlib import Path

import torch

from .train_paired import UNetGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="resources/models/checkpoints/he2dapi/generator_last.pt")
    ap.add_argument("--out", default="resources/models/checkpoints/he2dapi/he2dapi_pix2pix.onnx")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--ngf", type=int, default=64)
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    G = UNetGenerator(ngf=args.ngf)
    state = torch.load(str(ckpt), map_location="cpu")
    G.load_state_dict(state.get("G", state))
    G.eval()

    dummy = torch.randn(1, 3, args.size, args.size)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        G,
        dummy,
        str(out),
        input_names=["he"],
        output_names=["dapi"],
        dynamic_axes={"he": {0: "batch"}, "dapi": {0: "batch"}},
    )
    print(f"[ONNX] Exported -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
