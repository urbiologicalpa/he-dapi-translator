# Benchmark plan

Goal: compare virtual-staining approaches without mixing targets, resolutions,
or evaluation sets.

## Rules

- **Never rank panCK and DAPI in the same table.** Target channel is part of the
  experiment definition.
- Use the official split: **630 val**, **945 test** (not a 30-image subset).
- Report **mean ± std** and bootstrap confidence intervals.
- Keep data, split, normalization, architecture, checkpoint rule, and metric
  implementation fixed within a table.
- Put paper numbers in a separate "external reference" row.
- Save one JSON per run with: target, resolution, H5 path, checkpoint, git SHA,
  epochs, normalization, and sample count.
- Qualitative panels must include failures, not only best cases.

## Table 1 — DAPI main comparison

Only valid DAPI runs belong here.

| Method | Target | Res | Train data | Epochs | SSIM ↑ | Pearson ↑ | PSNR ↑ | Dice ↑ | Params | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| U-Net baseline | DAPI ch0 | 512 | 3,717 | 100 | TBD | TBD | TBD | TBD | TBD | planned |
| U-Net baseline | DAPI ch0 | 1024 | 3,717 | 100 | TBD | TBD | TBD | TBD | TBD | planned |
| Pix2Pix U-Net | DAPI ch0 | 1024 | 3,717 | 100 | TBD | TBD | TBD | TBD | TBD | planned |
| Dual-branch Swin+FMF | DAPI ch0 | 1024 | 3,717 | 100 | TBD | TBD | TBD | TBD | TBD | planned |
| Paper U-Net | DAPI | 1024 | HEMIT | — | 0.791 | 0.659 | 27.929 | — | — | external |
| Paper dual-branch | DAPI | 1024 | HEMIT | — | 0.815 | 0.716 | 28.610 | — | — | external |

## Table 2 — Ablations

Change one variable at a time.

| Scenario | Target | Res | Macenko | Flips | SSIM | Pearson | PSNR | Purpose |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Correct target | DAPI ch0 | 1024 | yes | no | TBD | TBD | TBD | main |
| Wrong target (legacy) | panCK ch2 | 512 | no | yes | 0.739 / 0.789 | 0.507 / 0.878 | 29.2 / 30.6 | pipeline baseline only |
| No normalization | DAPI ch0 | 1024 | no | no | TBD | TBD | TBD | stain sensitivity |
| Resolution | DAPI ch0 | 512 vs 1024 | yes | no | TBD | TBD | TBD | resolution effect |
| Architecture | DAPI ch0 | 1024 | yes | no | TBD | TBD | TBD | U-Net vs dual-branch |

The legacy 200-epoch panCK run belongs in this table, not Table 1. The row
shows **val / test**; full provenance is in
[`ABLATION_PANCK.md`](ABLATION_PANCK.md).

## Table 3 — Deployment scenarios

| Scenario | Input | Device | Precision | Latency / patch | Peak RSS | Notes |
|---|---|---|---:|---:|---:|---|
| Patch CPU | 512×512 | 3.7 GB CPU | FP32 | TBD | TBD | current path |
| Tiled CPU | 2048×2048 | 3.7 GB CPU | FP32 | TBD | TBD | 50% overlap |
| Quantized | 512×512 | 3.7 GB CPU | INT8/ONNX | TBD | TBD | target <3 GB |
| GPU | 1024×1024 | T4 16 GB | FP16/FP32 | TBD | TBD | training/inference |

## Table 4 — Reverse direction

Separate table. Label outputs **predicted H&E-like**, not H&E.

| Method | Target | Res | Data | SSIM | FID | Notes |
|---|---|---:|---|---:|---:|---|
| CycleGAN/CUT | H&E-like | 512 | unpaired | TBD | TBD | not trained |
| Paired reverse | H&E-like | 1024 | HEMIT reverse pairs | TBD | TBD | optional |

## Artifact naming

- `docs/metrics_panck_512.json` — legacy 200-epoch ablation.
- `docs/metrics_dapi_<res>_<method>.json` — corrected DAPI runs.
- `docs/metrics_reverse_<method>.json` — reverse runs.

Do not overwrite the legacy file with DAPI results. Target channel is part of the
record.
