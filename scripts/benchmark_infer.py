#!/usr/bin/env python3
"""Benchmark patch inference: PyTorch FP32 vs ONNX FP32/INT8 on CPU.

Writes a JSON report with latency percentiles, peak RSS, model params, and the
FP32-vs-ONNX output difference. The INT8 path uses ONNX Runtime dynamic
quantization so it can run on the 3.7GB CPU box without calibration data.

Run:
  PYTHONPATH=. .venv/bin/python scripts/benchmark_infer.py \
    --direction he2dapi --checkpoint resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt \
    --size 512 --runs 5 --out docs/benchmark_infer.json
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from src.models.train_paired import UNetGenerator


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _latencies(fn, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    a = np.asarray(times)
    return {
        "runs": runs,
        "mean_ms": round(float(a.mean()), 2),
        "std_ms": round(float(a.std()), 2),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "min_ms": round(float(a.min()), 2),
        "max_ms": round(float(a.max()), 2),
    }


def _load_generator(direction: str, checkpoint: str | None, ngf: int):
    in_ch, out_ch = (3, 1) if direction == "he2dapi" else (1, 3)
    G = UNetGenerator(in_ch=in_ch, out_ch=out_ch, ngf=ngf).eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        G.load_state_dict(state.get("G", state.get("generator", state)))
    return G


def _export_onnx(G, direction: str, size: int, out: Path, dynamic: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    in_ch = 3 if direction == "he2dapi" else 1
    dummy = torch.randn(1, in_ch, size, size)
    axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
    torch.onnx.export(
        G, dummy, str(out), input_names=["input"], output_names=["output"],
        dynamic_axes=axes, opset_version=17,
    )
    print(f"[ONNX] {out} ({out.stat().st_size / 1e6:.1f} MB)")


def _quantize_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"[INT8] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default="he2dapi", choices=["he2dapi", "dapi2he"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--onnx-dir", default="resources/models/checkpoints/bench")
    ap.add_argument("--out", default="docs/benchmark_infer.json")
    args = ap.parse_args()

    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
    G = _load_generator(args.direction, args.checkpoint, args.ngf)
    in_ch = 3 if args.direction == "he2dapi" else 1
    x = torch.randn(1, in_ch, args.size, args.size)
    report = {
        "direction": args.direction,
        "checkpoint": args.checkpoint,
        "size": args.size,
        "ngf": args.ngf,
        "params_m": round(sum(p.numel() for p in G.parameters()) / 1e6, 3),
        "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }

    with torch.no_grad():
        ref = G(x).numpy()
        report["pytorch_fp32"] = _latencies(lambda: G(x), args.runs, args.warmup)

    onnx_dir = Path(args.onnx_dir)
    onnx_fp32 = onnx_dir / f"{args.direction}_{args.size}_fp32.onnx"
    onnx_int8 = onnx_dir / f"{args.direction}_{args.size}_int8.onnx"
    _export_onnx(G, args.direction, args.size, onnx_fp32, dynamic=False)

    import onnxruntime as ort
    xn = x.numpy()
    for label, path in (("onnx_fp32", onnx_fp32), ("onnx_int8", onnx_int8)):
        if label == "onnx_int8":
            _quantize_int8(onnx_fp32, onnx_int8)
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: xn})[0]
        report[label] = _latencies(
            lambda s=sess, n=sess.get_inputs()[0].name: s.run(None, {n: xn}),
            args.runs, args.warmup,
        )
        report[label]["size_mb"] = round(
            sum(p.stat().st_size for p in path.parent.glob(path.name + "*")) / 1e6, 1
        )
        report[label]["max_abs_diff"] = round(float(np.abs(out - ref).max()), 5)
        report[label]["mean_abs_diff"] = round(float(np.abs(out - ref).mean()), 5)

    report["peak_rss_mb"] = round(_peak_rss_mb(), 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
