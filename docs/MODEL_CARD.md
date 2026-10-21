# Model card — H&E ↔ DAPI virtual staining

## Current release status

**Alpha research software. Not for clinical use.**

The public checkpoints still resolve to an **H&E → panCK ablation**, not the
corrected DAPI model. A channel-order bug caused the previous data pipeline to
use `mihc[..., 2]` (panCK) as the target. The correct target is `mihc[..., 0]`
(DAPI). A corrected DAPI baseline has now been trained, but it is **undertrained,
not published, and not release quality**.

| Item | Value |
|---|---|
| Product target | Bidirectional H&E ↔ DAPI in one plugin/API |
| Current release | One-way forward baseline only; not a bidirectional product |
| Task (intended) | H&E RGB → DAPI grayscale |
| Task (current public weights) | H&E RGB → panCK (ablation only) |
| Task (new internal baseline) | H&E RGB → DAPI grayscale (`ch0`) |
| Direction | One-way; DAPI → H&E is not available |
| Generator (current) | U-Net, 3 → 1, `ngf=64` |
| Discriminator | PatchGAN |
| Losses | GAN + L1 + SSIM + optional DISTS + nuclei Dice |
| Training data | HEMIT, 5,292 pairs (3,717 / 630 / 945) |
| Native resolution | 1024 × 1024 |
| Current training resolution | 512 px pretrain + 1024 px fine-tune |
| Framework | PyTorch |
| Public weights | `resources/models/checkpoints/he2dapi/generator_last.pt` (ablation) |
| DAPI baseline weights | `resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt` (internal) |
| Package weights | `he2dapi_unet_epoch169.pt` on Hugging Face (ablation) |

## Intended use

- Research on virtual staining, image-to-image translation, and digital
  pathology.
- Pipeline and deployment experiments on low-resource hardware.
- Reproducibility studies using HEMIT.

## Out-of-scope use

- Clinical diagnosis, treatment selection, or pathology reporting.
- Replacing real DAPI/IF staining in a regulated workflow.
- Claims that the current checkpoint produces DAPI. It produces a panCK-like
  channel from the ablation run.
- Use on modalities, scanners, tissues, or stain protocols not represented in
  HEMIT without separate validation.

## Training data and labels

HEMIT provides same-slide aligned H&E and mIHC patches. Label channels:

| Channel | Marker |
|---|---|
| 0 | DAPI |
| 1 | CD3 |
| 2 | panCK |

The corrected builder stores all three channels and extracts DAPI explicitly.
See [`DATA.md`](DATA.md). Dataset license: CC BY 4.0.

## Metrics

### DAPI baseline metrics (internal)

Corrected target `ch0`, 1024px fine-tune, full splits. This is a short baseline,
not the paper protocol.

| Split | n | SSIM | Pearson | PSNR (dB) | Dice |
|---|---:|---:|---:|---:|---:|
| val | 630 | 0.670 | 0.255 | 28.77 | 0.104 |
| test | 945 | 0.725 | 0.611 | 26.60 | 0.449 |

The large val/test gap and low val Pearson/Dice mean this checkpoint is not
release quality. Paper reference points (HEMIT DAPI column, not this repository):

| Model | SSIM | Pearson | PSNR (dB) |
|---|---:|---:|---:|
| U-Net | 0.791 | 0.659 | 27.929 |
| Pix2Pix U-Net | 0.775 | 0.652 | 27.691 |
| Dual-branch | 0.815 | 0.716 | 28.610 |

### Ablation metrics (panCK, not DAPI)

The 512px CPU run and `docs/metrics_eval.json` describe the panCK ablation.
They must not be copied into a DAPI model card. The run is retained only as an
engineering baseline (resumability, RAM behavior, loss stability, inference
plumbing).

## Limitations and failure modes

- **Wrong target in public weights.** The current demo is panCK; the corrected
  DAPI baseline is internal only.
- **Domain shift.** HEMIT is a specific same-slide mIHC dataset; generalization
  is unknown.
- **Alignment.** The paired setup assumes aligned H&E/mIHC patches. Misalignment
  is a known failure mode in virtual staining.
- **Resolution.** The baseline was pretrained at 512px and only briefly
  fine-tuned at 1024px; WSI-scale morphology is not validated.
- **Architecture.** The code currently uses a U-Net baseline, not the
  dual-branch Swin/FMF architecture described in the HEMIT paper.
- **Evaluation size.** The new DAPI baseline uses full 630/945 splits; older
  panCK numbers were only a 30-image subset.
- **Stain variability.** Macenko normalization is optional and scanner/stain
  variation can change outputs.
- **Deployment.** Full 1024px inference is not expected to fit comfortably in
  3.7 GB RAM; tiled inference plus quantization/distillation may be required.

## Ethical and safety notes

Synthetic stains can look plausible while being biologically wrong. Do not use
them to make diagnostic or treatment decisions. Any downstream study should
include pathologist review, external validation, and explicit uncertainty
reporting. The software is provided under MIT without warranty.

## Planned corrected release

1. ~~Build corrected 1024px HEMIT HDF5 (`dapi = ch0`, Macenko).~~ Done.
2. ~~Train a first U-Net DAPI baseline on a T4.~~ Done; undertrained.
3. Improve/retrain with more 1024 steps or the dual-branch generator.
4. Evaluate full validation and test splits with mean ± std / CIs.
5. Train paired reverse DAPI → H&E, then unpaired CycleGAN.
6. Publish corrected weights with a manifest that states the target channel and
   evaluation split only after metrics are reproducible.
7. Update PyPI, Hugging Face, Docker, and the demo only after the DAPI metrics
   are reproducible.
