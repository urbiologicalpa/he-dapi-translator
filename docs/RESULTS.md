# Results

> **CHANNEL AUDIT — 2026-10-23: corrected labels below.**
> The raw HEMIT RGB TIFF has **ch2=DAPI (blue), ch0=panCK (red), ch1=CD3
> (green)**. The legacy 512 model is the **real DAPI** model; the 1024 `ch0`
> forward/reverse models and R-006 are **panCK**. The 256 H5 was rebuilt on
> `ch2`; a corrected 1024 DAPI model is still pending. See
> [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md). Do not use R-006 as DAPI.

## TL;DR

- The only real **H&E → DAPI** baseline is the legacy 512px 200-epoch model
  (`ch2`); it is internal, non-paper protocol, and its provenance is being
  re-verified.
- The epoch-199/200 checkpoints and the 1024 `ch0` run are the **panCK
  ablation**; R-006 is also panCK. They are engineering baselines, not DAPI.
- The 256 H5 was rebuilt on `mihc[...,2]` on 2026-10-23 and verified; the
  loader ignores the stale 1024 `dapi` cache.
- No valid 1024 DAPI model and no valid DAPI→H&E model exist yet.
- Paper targets are listed below for orientation; they are **not** results from
  this repository.

## Why the old metrics are invalid for DAPI

HEMIT label TIFFs are ordered:

```
ch0 = panCK (red)
ch1 = CD3   (green)
ch2 = DAPI  (blue)
```

The legacy 512 builder stored `mihc[..., 2]`, so that model learned H&E → DAPI.
The later 1024/256 builders stored `ch0`, so those models and R-006 learned
H&E → panCK. The legacy 512 checkpoint/metrics are therefore the real DAPI
baseline; the 1024 `ch0` metrics are the panCK ablation.

The correction is implemented in:

- `scripts/data/pack_hemit.py` (legacy 512 builder; `ch2`)
- `scripts/data/pack_hemit_1024.py` (`DAPI_CHANNEL = 2`)
- `scripts/data/crop_hemit_h5.py` (256 rebuild from `mihc[...,2]`)
- `src/models/datasets.py` (loader prefers `mihc[...,2]`)
- `tests/test_hemit_channels.py` (regression tests)

The 1024 `ch0` metrics remain only as an engineering ablation. They must not be
reported as DAPI performance.

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
| Legacy CPU, 200 epochs, 512px | DAPI (`ch2`) | Real DAPI baseline | Full 630 val / 945 test: SSIM 0.739 / 0.789, Pearson 0.507 / 0.878, PSNR 29.23 / 30.65 dB |
| 1024px U-Net, `ch0` | panCK (`ch0`) | panCK ablation; not DAPI | Full 630 val / 945 test: SSIM 0.670 / 0.725, Pearson 0.255 / 0.611, PSNR 28.77 / 26.60 dB, Dice 0.104 / 0.449 |
| Corrected dual-branch | DAPI (`ch2`) | Not implemented | — |
| Reverse 1024, `ch0` | H&E from panCK | panCK→H&E ablation; not DAPI | 20 epochs @512 + 2×500 @1024; full 630/945 eval; not released |

## Legacy 512 DAPI re-verification (2026-10-23)

Checkpoint `resources/models/checkpoints/he2dapi/checkpoint_epoch200.pt`
(SHA-256 `6789d7d244cc6ac93dffd98a30ed2c17498aa7ee6081560dedac5173b44042aa`)
was re-evaluated with the corrected loader (which reads `mihc[...,2]`) on the
legacy 512 val H5, 50 center crops:

| n | FID ↓ | LPIPS ↓ | PSNR | SSIM | Pearson | L1 |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 146.27 | 0.2468 | 29.52 | 0.7662 | 0.5323 | 0.0221 |

This strongly supports the audit: the legacy 512 model predicts nuclei/DAPI,
not panCK. The full 630/945 numbers in the table above came from the same
`ch2` target. Provenance is now recorded; re-run full splits before release.

## First 1024 baseline (panCK target — invalid for DAPI)

Run: Kaggle T4, 20 epochs at 512 (3,717 pairs) then 2×500 steps at 1024.
Checkpoint: `resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt`
(SHA-256 `871560a1…b4103d`). Target: panCK `ch0` — **not DAPI**. No Macenko.

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
| Forward 1024 baseline | panCK ch0 (invalid for DAPI) | 77.14 | 0.1407 | 33.88 | 0.728 |
| Reverse 1024 baseline | H&E from panCK ch0 (invalid for DAPI) | 313.18 | 0.6051 | 13.25 | 0.496 |

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

The reverse quality pass is also a negative result (same 4-epoch recipe, 50 val
crops at 512px):

| Checkpoint | FID ↓ | LPIPS ↓ | PSNR | SSIM | L1 ↓ |
|---|---:|---:|---:|---:|---:|
| Reverse 1024 baseline | **313.18** | 0.6051 | **13.25** | **0.496** | **0.2009** |
| Reverse quality 4-epoch | 334.85 | **0.5988** | 12.95 | 0.458 | 0.2053 |

Train losses fell monotonically (G 11.59 → 11.02) while val FID/PSNR/SSIM
worsened: the pass fit the training slides, not the held-out slide. Keep the
baseline reverse checkpoint as the comparison point for the CycleGAN run.
JSON: `docs/metrics_dapi2he_quality_1024.json`.

## Unpaired CycleGAN, external H&E domain (MHIST)

Run: `bikasash/he-dapi-cyclegan-dapi2he-1024`, 10 epochs at 256, batch 4,
grad-accum 2, ResNet-9 generators + PatchGAN discriminators. The H&E domain was
MHIST (3,152 patches), the DAPI domain HEMIT train. 50 HEMIT val crops at 256:

| Model | Target domain | FID ↓ | LPIPS ↓ | PSNR | SSIM | L1 ↓ |
|---|---|---:|---:|---:|---:|---:|
| Paired reverse baseline | HEMIT H&E | **342.98** | **0.5904** | **13.17** | **0.4941** | **0.2056** |
| CycleGAN (MHIST H&E) | MHIST H&E | 362.34 | 0.6999 | 10.76 | 0.2324 | 0.2488 |

The unpaired run is worse on every metric, but this comparison is confounded:
the generator learned MHIST H&E colour/texture while the FID reference is HEMIT
H&E. Alignment metrics (PSNR/SSIM) are meaningless for an unpaired model.
JSON: `docs/metrics_dapi2he_cyclegan_256.json`.

A second CycleGAN run with **HEMIT H&E as the unpaired domain** is launched
(`bikasash/he-dapi-cyclegan-hemit-dapi2he-1024`, `configs/cyclegan_hemit_dapi2he.yaml`)
so the paired-vs-unpaired comparison uses the same target distribution.

This is **not** paper-level: paper U-Net is SSIM 0.791 / Pearson 0.659.
The val/test gap is large and likely reflects slide/domain differences plus
limited fine-tuning. It is a plumbing baseline, not a claim of DAPI quality.

## R-006 — HEMIT 256px CycleGAN (panCK `ch0`; invalid for DAPI)

Run: `bikasash/he-dapi-cyclegan-hemit-dapi2he-256`, 10 epochs, 929
steps/epoch, batch 4 × accum 2, AMP, HEMIT `hemit_{train,val}_256.h5` only
(`HE: 0 files + H5`, `synthetic=False`), exit 0 at 2026-10-22 18:55 UTC.

Local 50-crop val evaluation (same first 50 crops as the paired baseline):

| Model | Target | FID ↓ | LPIPS ↓ | PSNR | SSIM | L1 ↓ | Pearson |
|---|---|---:|---:|---:|---:|---:|---:|
| Paired reverse 1024 @256 | H&E | 342.98 | 0.5904 | 13.17 | 0.4941 | 0.2056 | 0.4856 |
| CycleGAN HEMIT 256 | H&E | 367.97 | 0.5948 | 11.37 | 0.2216 | 0.2292 | 0.1596 |

**Both models were trained on `ch0`, which is panCK, not DAPI.** This table is
therefore a panCK→H&E comparison, not the reverse DAPI comparison. Do not use
it to select the DAPI architecture. The full 630-crop JSON (no FID/LPIPS) is
L1 0.2543, PSNR 10.61, SSIM 0.2051, Pearson 0.1537.

## Evaluation protocol

The baseline used the official split:

- Validation: 630 pairs.
- Test: 945 pairs.
- Report SSIM, Pearson correlation, PSNR, and nuclei Dice.
- Record the exact H5 file, checkpoint, normalization, and target channel in the
  metrics JSON.
- Keep per-image outputs or hashes so the numbers can be audited.

**Correction (2026-10-23):** the legacy 512 model is **DAPI (`ch2`)**, and the
1024 `ch0` model is the panCK ablation. The paragraphs above that say otherwise
are frozen pending the ADR-014 rewrite. The legacy 512 DAPI model's full-split
metrics are val SSIM 0.7389 / Pearson 0.5072 and test SSIM 0.7890 / Pearson
0.8781; these are the numbers closest to the paper's DAPI target. See
[`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md) and
[`ABLATION_PANCK.md`](ABLATION_PANCK.md) (the latter title is now misleading;
it documents the real DAPI run and will be renamed after approval).
