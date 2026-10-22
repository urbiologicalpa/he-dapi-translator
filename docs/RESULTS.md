# Results

## TL;DR

- A corrected **H&E → DAPI baseline now exists**, but it is undertrained and
  below the paper reference. Its val/test gap is large; do not over-read it.
- The epoch-199/200 CPU run and all current demo/package weights are an
  **H&E → panCK ablation** caused by using label channel `ch2` instead of
  `ch0` (DAPI).
- The panCK ablation now has full **630 val + 945 test** metrics; they are
  engineering baselines, not DAPI results.
- Corrected 1024px HEMIT HDF5 is built and verified. The DAPI baseline has now
  been trained on a Kaggle T4 (20 epochs at 512 + 1,000 steps at 1024).
- Paper targets are listed below for orientation; they are **not** results from
  this repository.

## Why the old metrics are invalid for DAPI

HEMIT label TIFFs are ordered:

```
ch0 = DAPI
ch1 = CD3
ch2 = panCK
```

The original 512px builder stored `mihc[..., 2]` as the target. The model
therefore learned H&E → panCK. The public checkpoint, `docs/metrics_eval.json`,
the Hugging Face weights, and the demo output all belong to that ablation.

The bug is fixed in:

- `scripts/data/pack_hemit.py` (legacy 512 builder)
- `scripts/data/pack_hemit_1024.py` (correct 1024 builder)
- `src/models/datasets.py` (`mihc[..., 0]` fallback)
- `tests/test_hemit_channels.py` (regression tests)

The old metrics remain in the repository only as an engineering baseline. They
must not be reported as DAPI performance.

## Paper reference (HEMIT, DAPI column)

From the HEMIT paper (Bian et al., 2024):

| Model | SSIM | Pearson | PSNR (dB) |
|---|---:|---:|---:|
| U-Net | 0.791 | 0.659 | 27.929 |
| Pix2Pix U-Net | 0.775 | 0.652 | 27.691 |
| Dual-branch | **0.815** | **0.716** | **28.610** |

Protocol notes from the paper/reference implementation:

- 1024 × 1024 patches.
- Adam, `lr=3e-5`, 50 epochs then linear decay (100 epochs total).
- `λ_L1=30`.
- Batch size 2 on a 16 GB V100.
- No random flips.
- InstanceNorm in the official dual-branch implementation.

## This repository

| Run | Target | Status | Metrics |
|---|---|---|---|
| Local CPU, 200 epochs, 512px | panCK (`ch2`) | Ablation, retained | Full 630 val / 945 test: SSIM 0.739 / 0.789, Pearson 0.507 / 0.878, PSNR 29.23 / 30.65 dB; not DAPI |
| Corrected 1024px U-Net | DAPI (`ch0`) | First baseline, undertrained | Full 630 val / 945 test: SSIM 0.670 / 0.725, Pearson 0.255 / 0.611, PSNR 28.77 / 26.60 dB, Dice 0.104 / 0.449 |
| Corrected dual-branch | DAPI (`ch0`) | Not implemented | — |
| Reverse DAPI → H&E | H&E | First baseline, undertrained | 20 epochs @512 + 2×500 @1024; full 630/945 eval; not released |

## First corrected DAPI baseline

Run: Kaggle T4, 20 epochs at 512 (3,717 pairs) then 2×500 steps at 1024.
Checkpoint: `resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt`
(SHA-256 `871560a1…b4103d`). Target: DAPI `ch0`. No Macenko.

| Split | n | SSIM | Pearson | PSNR | Dice |
|---|---:|---:|---:|---:|---:|
| val | 630 | 0.670 | 0.255 | 28.77 | 0.104 |
| test | 945 | 0.725 | 0.611 | 26.60 | 0.449 |

## First reverse DAPI → H&E baseline

Run: Kaggle T4, 20 epochs at 512 then 2×500 steps at 1024, using the same
paired HEMIT data and budget as forward. Checkpoint:
`resources/models/checkpoints/dapi2he_1024/checkpoint_last.pt`
(SHA-256 `dbe6ebde…f634d`). Output is RGB H&E.

| Split | n | SSIM | Pearson | PSNR | L1 |
|---|---:|---:|---:|---:|---:|
| val | 630 | 0.437 | 0.521 | 15.18 | 0.1449 |
| test | 945 | 0.593 | 0.576 | 15.79 | 0.1270 |

This is **not release quality**. The low SSIM/PSNR indicates blur and/or color
mismatch, and only 1,000 steps were run at 1024. Dice is intentionally `null`
for RGB H&E. The next pass inspects qualitative grids, then tests longer 1024
fine-tuning plus perceptual/color losses before comparing against unpaired
CycleGAN/CUT.

FID + LPIPS (50 val crops, 512px, Inception pool3 / AlexNet):

| Run | Target | val FID ↓ | val LPIPS ↓ | PSNR | SSIM |
|---|---|---:|---:|---:|---:|
| Forward 1024 baseline | DAPI ch0 | 77.14 | 0.1407 | 33.88 | 0.728 |
| Reverse 1024 baseline | H&E | 313.18 | 0.6051 | 13.25 | 0.496 |

The reverse FID/LPIPS are far worse than forward, consistent with the low
output contrast (fake std ≈ 0.09 vs real ≈ 0.18). JSON:
`docs/metrics_dapi_1024_baseline.json`, `docs/metrics_dapi2he_1024_baseline.json`.

## Forward quality pass (colour + gradient) — negative result

Run: `bikasash/he-dapi-he2dapi-quality-1024`, 4 epochs at 1024 (batch 1,
grad-accum 8, lr 3e-5) seeded from the forward 1024 baseline, adding
`color_weight=5`, `grad_weight=2` on top of the existing losses. 50 val crops:

| Checkpoint | px | FID ↓ | LPIPS ↓ | PSNR | SSIM | L1 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Forward 1024 baseline | 256 | **48.52** | **0.1145** | **36.66** | **0.7438** | **0.0198** |
| Forward quality 4-epoch | 256 | 74.66 | 0.1166 | 35.54 | 0.713 | 0.0234 |
| Forward 1024 baseline | 512 | 77.14 | 0.1407 | 33.88 | 0.728 | 0.0214 |
| Forward quality 4-epoch | 512 | 89.32 | 0.1412 | 32.49 | 0.664 | 0.0259 |

The colour/gradient pass did **not** improve DAPI translation and slightly
degraded every metric. The baseline remains the default forward checkpoint.
Likely causes: colour-statistic loss overfitting the training slides (the val
slide differs), no val-based checkpoint selection, and only 4 epochs at a
constant lr. Next: lower `color_weight`, select on val FID/SSIM, and/or move to
the dual-branch architecture. JSON: `docs/metrics_dapi_quality_1024_256.json`,
`docs/metrics_dapi_quality_1024.json`.

This is **not** paper-level: paper U-Net is SSIM 0.791 / Pearson 0.659.
The val/test gap is large and likely reflects slide/domain differences plus
limited fine-tuning. It is a plumbing baseline, not a claim of DAPI quality.

## Evaluation protocol

The baseline used the official split:

- Validation: 630 pairs.
- Test: 945 pairs.
- Report SSIM, Pearson correlation, PSNR, and nuclei Dice.
- Record the exact H5 file, checkpoint, normalization, and target channel in the
  metrics JSON.
- Keep per-image outputs or hashes so the numbers can be audited.

Do not compare a 512px panCK ablation to 1024px DAPI paper numbers. They differ
in target, resolution, and evaluation set. Full ablation provenance is in
[`ABLATION_PANCK.md`](ABLATION_PANCK.md).
