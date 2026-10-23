# Model card — H&E ↔ DAPI virtual staining

## Current release status

**Alpha research software. Not for clinical use.**

> **CHANNEL CORRECTION 2026-10-23.** The raw HEMIT RGB TIFF stores
> **`ch2=DAPI` (blue), `ch0=panCK` (red), `ch1=CD3` (green)**. The legacy 512
> H&E→DAPI checkpoint is therefore the real DAPI model; the 1024 `ch0` models
> and R-006 are the panCK ablation. ADR-014 is accepted, the 256 H5 has been
> rebuilt on `ch2`, and the loader prefers `mihc[...,2]`. A corrected 1024 DAPI
> model and a valid DAPI→H&E model are still pending. See
> [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md).

| Item | Value |
|---|---|
| Product target | Bidirectional H&E ↔ DAPI in one plugin/API |
| Current release | One-way H&E→DAPI only; not a bidirectional product |
| Task (intended) | H&E RGB → DAPI grayscale |
| Real DAPI weights | Legacy 512 H&E→DAPI (`ch2`); provenance re-verification pending |
| 1024 `ch0` internal run | H&E → panCK (ablation; invalid for DAPI) |
| DAPI → H&E | Not available (the 1024 `ch0` reverse run is panCK→H&E) |
| Generator | U-Net, 3 → 1, `ngf=64` |
| Discriminator | PatchGAN |
| Losses | GAN + L1 + SSIM + optional DISTS + nuclei Dice |
| Training data | HEMIT, 5,292 pairs (3,717 / 630 / 945) |
| Native resolution | 1024 × 1024 |
| Legacy training resolution | 512 px |
| 1024 ablation resolution | 512 px pretrain + 1024 px fine-tune |
| Framework | PyTorch |
| Public weights | `he2dapi_unet_epoch169.pt` (HF) and `resources/models/checkpoints/he2dapi/` — legacy 512 DAPI, re-verify |
| Internal panCK ablation | `resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt` — **panCK, not DAPI** |
| Package weights | Legacy 512 H&E→DAPI; provenance re-verification pending |

## Intended use

- Research on virtual staining, image-to-image translation, and digital
  pathology.
- Pipeline and deployment experiments on low-resource hardware.
- Reproducibility studies using HEMIT.

## Out-of-scope use

- Clinical diagnosis, treatment selection, or pathology reporting.
- Replacing real DAPI/IF staining in a regulated workflow.
- Claims that the 1024 `ch0` checkpoint produces DAPI. It produces panCK.
- Use on modalities, scanners, tissues, or stain protocols not represented in
  HEMIT without separate validation.

## Training data and labels

HEMIT provides same-slide aligned H&E and mIHC patches. Raw label channels:

| Channel | Marker |
|---|---|
| 0 | panCK (red) |
| 1 | CD3 (green) |
| 2 | DAPI (blue) |

The corrected builder stores all three channels and extracts DAPI from `ch2`.
The 256px H5 was rebuilt on 2026-10-23 with `dapi_channel=2`; the 1024 H5
`dapi` cache is stale (`ch0`) and is intentionally ignored by the loader in
favour of `mihc[...,2]`. See [`DATA.md`](DATA.md) and
[`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md). Dataset license: CC BY 4.0.

## Metrics

### Real DAPI baseline — legacy 512px, 200 epochs

Target `ch2`, full splits, local CPU. Non-paper protocol; provenance is being
re-verified before release.

| Split | n | SSIM | Pearson | PSNR (dB) |
|---|---:|---:|---:|---:|
| val | 630 | 0.739 | 0.507 | 29.23 |
| test | 945 | 0.789 | 0.878 | 30.65 |

### panCK ablation — 1024px `ch0` (not DAPI)

Target `ch0`/panCK, 20 epochs at 512 + 1,000 steps at 1024, full splits. This
is retained only as an engineering baseline (resumability, RAM behaviour, loss
stability, inference plumbing). Do not copy these numbers into a DAPI model
card.

| Split | n | SSIM | Pearson | PSNR (dB) | Dice |
|---|---:|---:|---:|---:|---:|
| val | 630 | 0.670 | 0.255 | 28.77 | 0.104 |
| test | 945 | 0.725 | 0.611 | 26.60 | 0.449 |

### External reference (HEMIT paper, DAPI column — not this repository)

| Model | SSIM | Pearson | PSNR (dB) |
|---|---:|---:|---:|
| U-Net | 0.791 | 0.659 | 27.929 |
| Pix2Pix U-Net | 0.775 | 0.652 | 27.691 |
| Dual-branch | 0.815 | 0.716 | 28.610 |

## Limitations and failure modes

- **No valid DAPI→H&E model.** The only reverse checkpoint was trained on
  panCK `ch0`; it must not be presented as DAPI→H&E.
- **No 1024 DAPI model yet.** The 1024 `ch0` run is panCK; a `ch2` retrain is
  pending.
- **Provenance.** The legacy 512 weights and demo need checkpoint-level
  verification before they are published as DAPI.
- **Domain shift.** HEMIT is a specific same-slide mIHC dataset; generalization
  is unknown.
- **Alignment.** The paired setup assumes aligned H&E/mIHC patches. Misalignment
  is a known failure mode in virtual staining.
- **Resolution.** The legacy model is 512px; WSI-scale morphology is not
  validated.
- **Architecture.** The code currently uses a U-Net baseline, not the
  dual-branch Swin/FMF architecture described in the HEMIT paper.
- **DAPI dynamic range.** DAPI is sparse and low-contrast (mean ~11/255,
  p99 ~51); normalization choices materially affect training and display.
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

1. ~~Rebuild 256px HEMIT HDF5 with `dapi = mihc[...,2]`.~~ Done and verified.
2. Re-verify the legacy 512 DAPI checkpoint provenance and record channel/split.
3. Train a 1024px DAPI model on `ch2` (paired U-Net first, then dual-branch).
4. Train and evaluate a valid DAPI→H&E model on `ch2`.
5. Evaluate full validation and test splits with mean ± std / CIs.
6. Publish weights with a manifest stating target channel and split only after
   metrics are reproducible.
7. Update PyPI, Hugging Face, Docker, and the demo only after the DAPI metrics
   are reproducible.
