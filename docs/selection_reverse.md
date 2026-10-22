# Reverse architecture selection (Stage 2.4)

> **Status: PENDING R-006.** Do not fill with guesses. Update after the corrected
> HEMIT 256px CycleGAN exits and its val evaluation is pulled.
> Decision rule: `docs/PLAN.md` Stage 2.3 / ADR-008 (val only, test once).

## Question

For DAPI → H&E, should the product default be the paired U-Net or the unpaired
CycleGAN? The comparison must use the same HEMIT target domain, the same 256px
evaluation crops, and validation data only.

## References

- Paired reverse baseline: `docs/RESULTS.md`, checkpoint
  `resources/models/checkpoints/dapi2he_1024/checkpoint_last.pt`.
- Unpaired run: `bikasash/he-dapi-cyclegan-hemit-dapi2he-256` (v3), config
  `configs/cyclegan_hemit_dapi2he_kaggle.yaml`.
- Evaluation protocol: 630 val crops, 256px; FID/LPIPS primary; PSNR/SSIM/L1
  diagnostic for unpaired (alignment is not guaranteed).
- Prior negative: MHIST-domain CycleGAN (FID 362.34, LPIPS 0.6999) was
  confounded by a different H&E target domain.

## Metrics (fill from the run)

| Model | Target domain | n | FID ↓ | LPIPS ↓ | PSNR | SSIM | L1 ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| Paired reverse 1024 @256 | HEMIT H&E | 50* | 342.98 | 0.5904 | 13.17 | 0.4941 | 0.2056 |
| CycleGAN HEMIT 256 (R-006) | HEMIT H&E | 630 |  |  |  |  |  |

\*The paired reference FID/LPIPS was measured on 50 val crops. Re-run the paired
checkpoint with `--num 630` before declaring a winner if the unpaired run used
630; otherwise compare on the same 50-crop subset and say so.

## Qualitative audit (required before a decision)

- [ ] 6–10 paired examples: DAPI input, real H&E, paired output, CycleGAN output.
- [ ] Check colour/contrast collapse, checkerboard, nuclear ghosting, invented
      stroma/tumor structures.
- [ ] Check failure cases on the val slide(s) with the worst FID.
- [ ] Blind the model labels for the preference judgement.
- [ ] Record the grid path under `docs/qualitative_dapi2he/`.

## Cost / product fit

| Criterion | Paired U-Net | CycleGAN |
|---|---|---|
| Needs aligned pairs at train time | Yes (HEMIT only) | No |
| Inference cost | Lower (1 generator) | Higher (2 generators) |
| Hallucination risk | Lower | Higher |
| Real-world unpaired applicability | Limited | Better |
| Current evidence | undertrained baseline | pending R-006 |

## Decision

_Pending._ State one of:
- Keep paired U-Net as default; unpaired is not justified on val FID/LPIPS or
  qualitative audit.
- Switch to CycleGAN; state the val metric margin and the hallucination audit.
- Neither is release quality; proceed to Stage 2.5 with the chosen architecture.

## Provenance

- Git SHA at launch:
- Code dataset / SHA-256 of `src/models/datasets.py`:
- Kernel ref/version:
- Config + hash:
- Checkpoint SHA-256:
- Val H5 + SHA-256:
- Eval JSON:
- Decision author/date:
