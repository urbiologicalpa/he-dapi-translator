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
| Reverse DAPI → H&E | H&E | Not available | — |

## First corrected DAPI baseline

Run: Kaggle T4, 20 epochs at 512 (3,717 pairs) then 2×500 steps at 1024.
Checkpoint: `resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt`
(SHA-256 `871560a1…b4103d`). Target: DAPI `ch0`. No Macenko.

| Split | n | SSIM | Pearson | PSNR | Dice |
|---|---:|---:|---:|---:|---:|
| val | 630 | 0.670 | 0.255 | 28.77 | 0.104 |
| test | 945 | 0.725 | 0.611 | 26.60 | 0.449 |

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
