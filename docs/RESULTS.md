# Results

## TL;DR

- **No valid H&E → DAPI result exists yet.**
- The epoch-199/200 CPU run and all current demo/package weights are an
  **H&E → panCK ablation** caused by using label channel `ch2` instead of
  `ch0` (DAPI).
- The next milestone is a corrected 1024px DAPI baseline on a Kaggle T4,
  followed by full validation/test evaluation.
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
| Local CPU, 200 epochs, 512px | panCK (`ch2`) | Ablation, retained | 30-image val subset, ~29 dB PSNR; not DAPI |
| Corrected 1024px U-Net | DAPI (`ch0`) | Not started | — |
| Corrected dual-branch | DAPI (`ch0`) | Not implemented | — |
| Reverse DAPI → H&E | H&E | Not available | — |

## Planned evaluation

Use the official split, not a 30-image subset:

- Validation: 630 pairs.
- Test: 945 pairs.
- Report SSIM, Pearson correlation, PSNR, and nuclei Dice.
- Record the exact H5 file, checkpoint, normalization, and target channel in the
  metrics JSON.
- Keep per-image outputs or hashes so the numbers can be audited.

Do not compare a 512px panCK ablation to 1024px DAPI paper numbers. They differ
in target, resolution, and evaluation set.
