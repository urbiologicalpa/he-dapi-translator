# Reverse architecture selection (Stage 2.4)

> **Status: CLOSED 2026-10-23 12:20 UTC — paired U-Net selected by default.**
> R-007 (`bikasash/he-dapi-cyclegan-ch2-256-r-007`) completed 12:13 UTC but is
> **invalid for selection**: its eval omitted `--metrics fid,lpips` and its
> Tanh output was compared against `[0,1]` targets. No valid unpaired FID/LPIPS
> exists, so the paired `dapi2he_ch2_1024` U-Net remains the Stage 2.5
> architecture. Do not use R-006 or any `ch0` row for DAPI.
> See [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md) and `docs/PLAN.md` Stage 2.3/2.4.

## Question

For DAPI → H&E, should the product default be the paired U-Net or the unpaired
CycleGAN? The comparison must use the same HEMIT target domain, the same 256px
evaluation crops, and validation data only.

## References

- Paired reverse baseline: `docs/RESULTS.md`, checkpoint
  `resources/models/checkpoints/dapi2he_ch2_1024/checkpoint_last.pt`
  (SHA-256 `5f930960…a870f98`), 256 px / 50 val crops.
- Unpaired run: `bikasash/he-dapi-cyclegan-ch2-256-r-007` (v1), config
  `configs/cyclegan_hemit_dapi2he_kaggle.yaml`, `ch2` H5 only.
- Evaluation protocol: 630 val rows at 256px; FID/LPIPS on the first 50
  (`metric_num=50`), matching the paired reference; PSNR/SSIM/L1 diagnostic for
  unpaired (alignment is not guaranteed).
- Prior negative: MHIST-domain CycleGAN (FID 362.34, LPIPS 0.6999) was
  confounded by a different H&E target domain.

## Valid ch2 comparison (R-007)

| Model | Target domain | rows / metric n | FID ↓ | LPIPS ↓ | PSNR | SSIM | L1 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Paired reverse `dapi2he_ch2_1024` @256 | HEMIT H&E | 50 / 50 | **285.29** | **0.5129** | 15.45 | 0.4772 | 0.1548 |
| CycleGAN R-007 `ch2` 256 | HEMIT H&E | 630 / 50 | **invalid (null)** | **invalid (null)** | 8.45\* | 0.2215\* | 0.3449\* |

\*Raw R-007 diagnostics are invalid because the generator used Tanh (`[-1,1]`)
while eval compared it with `[0,1]` targets. A 5-crop spot check with correct
denormalization gives PSNR 12.25 / SSIM 0.3698 / Pearson -0.2975, still far
below the paired reference. The 10-epoch run is also undertrained. **No
selection can be made from these numbers.**

The paired row is `docs/metrics_ch2_1024_reverse_256_val50.json`. R-007's
`eval_cyclegan.json` is at
`/home/bikash/kaggle_r007_output/checkpoints/hemit_cyclegan_ch2_256/eval_cyclegan.json`;
its `val_FID` and `val_LPIPS` are null. Do not fill a winner row from it.

## Invalid ch0 rows (do not use for DAPI)

| Model | Target domain | n | FID ↓ | LPIPS ↓ | PSNR | SSIM | L1 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Paired reverse 1024 @256 (**ch0 panCK**) | HEMIT H&E | 50 | 342.98 | 0.5904 | 13.17 | 0.4941 | 0.2056 |
| CycleGAN HEMIT 256 (R-006, **ch0 panCK**) | HEMIT H&E | 50 | 367.97 | 0.5948 | 11.37 | 0.2216 | 0.2292 |

**Warning:** both rows are panCK→H&E, not DAPI→H&E. They are not valid for the
reverse DAPI decision. The R-006 630-crop JSON (L1 0.2543, PSNR 10.61, SSIM
0.2051, Pearson 0.1537) is in `/tmp/k256_eval/`; the 50-crop row above is the
fair comparison to the paired baseline.

## Qualitative audit

**Not performed.** R-007 is invalid for selection, so a blinded preference
audit would not change the architecture decision. No qualitative claim is made.
The R-006 grid (`docs/qualitative_dapi2he/hemit_val_compare.png`) is `ch0`
panCK and is not a DAPI audit.

## Cost / product fit

| Criterion | Paired U-Net | CycleGAN |
|---|---|---|
| Needs aligned pairs at train time | Yes (HEMIT only) | No |
| Inference cost | Lower (1 generator) | Higher (2 generators) |
| Hallucination risk | Lower | Higher |
| Real-world unpaired applicability | Limited | Better |
| Current evidence | valid but undertrained baseline | **invalid run; no valid comparison** |

## Decision

**Keep the paired U-Net as the reverse default.** The unpaired CycleGAN did not
produce a valid FID/LPIPS comparison, so it cannot displace the paired baseline.
Do not spend more quota on CycleGAN or start CUT by default. The next GPU spend
is Stage 2.5: longer 1024 paired training with val-based checkpoint selection,
then full 630/945 evaluation and CIs. Neither current model is release quality.

## Code fixes after R-007

- `src/models/train_unpaired.py`: `ResnetGenerator` now ends in `Sigmoid`, so
  its `[0,1]` output matches the `[0,1]` losses/eval. Legacy Tanh checkpoints
  (R-006/R-007) are not valid under the fixed code.
- `scripts/kaggle/he-dapi-cyclegan-ch2-256.py`: future evals pass
  `--metrics fid,lpips`.
- `src/backend/main.py`: reverse direction now serves the corrected
  `dapi2he_ch2_1024/generator_last.pt`, not the invalid aborted
  `dapi2he/cyclegan_G_last.pt`.

## Provenance

- Git SHA at launch: `d69eb65` (docs/guard edits uncommitted).
- Code dataset: `bikasash/he-dapi-code-dapi-1024` (2026-10-23 04:53 UTC).
- Kernel ref/version: `bikasash/he-dapi-cyclegan-ch2-256-r-007` v1.
- Config + hash: `configs/cyclegan_hemit_dapi2he_kaggle.yaml`,
  SHA-256 `689ebc40…e9079`.
- Checkpoint SHA-256: `cyclegan_G_last.pt` `815023a0…9e4857`;
  `generator_last.pt` `d5ef563e…39956f`.
- Val H5: `hemit_val_256.h5`, `dapi = mihc[...,2]`, verified in-kernel.
- Eval JSON: `/home/bikash/kaggle_r007_output/checkpoints/hemit_cyclegan_ch2_256/eval_cyclegan.json`
  (FID/LPIPS null; invalid).
- Decision author/date: coding agent, 2026-10-23 12:20 UTC; user can reopen
  only with a valid FID/LPIPS run.
