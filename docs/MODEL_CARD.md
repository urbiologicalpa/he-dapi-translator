# Model card — H&E ↔ DAPI virtual staining

## Current release status

**Alpha research software. Not for clinical use.**

> **CHANNEL CORRECTION 2026-10-23.** The raw HEMIT RGB TIFF stores
> **`ch2=DAPI` (blue), `ch0=panCK` (red), `ch1=CD3` (green)**. The legacy 512
> H&E→DAPI checkpoint is therefore the real DAPI model; the 1024 `ch0` models
> and R-006 are the panCK ablation. ADR-014 is accepted, the 256 H5 has been
> rebuilt on `ch2`, and the loader prefers `mihc[...,2]`. Corrected 1024 `ch2`
> baselines now exist in both directions (2026-10-23), but they are undertrained
> and **not release quality**. The legacy 512 model remains the best forward
> DAPI checkpoint. See [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md) and
> [`RESULTS.md`](RESULTS.md).

| Item | Value |
|---|---|
| Product target | Bidirectional H&E ↔ DAPI in one plugin/API |
| Current release | One-way H&E→DAPI only; not a bidirectional product |
| Task (intended) | H&E RGB → DAPI grayscale |
| Real DAPI weights | Legacy 512 H&E→DAPI (`ch2`); local + HF SHA verified 2026-10-23 |
| 1024 `ch0` internal run | H&E → panCK (ablation; invalid for DAPI) |
| Corrected 1024 `ch2` forward | `resources/models/checkpoints/he2dapi_ch2_1024/` — valid but undertrained |
| Corrected 1024 `ch2` reverse | `resources/models/checkpoints/dapi2he_ch2_1024/` — first valid DAPI→H&E, not release quality |
| Reverse architecture | Paired U-Net selected; R-007 CycleGAN invalid for selection (no FID/LPIPS; Tanh/[0,1] mismatch) |
| Generator | U-Net, 3 → 1, `ngf=64` |
| Discriminator | PatchGAN |
| Losses | GAN + L1 + SSIM + optional DISTS + nuclei Dice |
| Training data | HEMIT, 5,292 pairs (3,717 / 630 / 945) |
| Native resolution | 1024 × 1024 |
| Legacy training resolution | 512 px |
| 1024 ablation resolution | 512 px pretrain + 1024 px fine-tune |
| Framework | PyTorch |
| Public weights | `he2dapi_unet_epoch169.pt` (HF), SHA-256 `c87751…e2afdc`, 46/46 generator tensors exactly equal to local `checkpoint_epoch169.pt["G"]` |
| Internal panCK ablation | `resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt` — **panCK, not DAPI** |
| Package weights | Legacy 512 H&E→DAPI; checkpoint/HF match verified; demo/API SHA manifest still pending |

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

Target `ch2`, full splits, local CPU. Non-paper protocol; target and artifact
provenance verified 2026-10-23 (HF SHA `c87751…e2afdc`).

| Split | n | SSIM | Pearson | PSNR (dB) | FID ↓ | LPIPS ↓ | L1 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| val (full, epoch200) | 630 | 0.739 | 0.507 | 29.23 | — | — | 0.0252 |
| test (full, epoch200) | 945 | 0.789 | 0.878 | 30.65 | — | — | 0.0217 |
| val (first 50, epoch200) | 50 | 0.766 | 0.532 | 29.52 | **146.27** | **0.2468** | 0.0221 |
| val (first 50, epoch169 = HF) | 50 | 0.798 | 0.500 | 29.85 | 161.91 | 0.2745 | 0.0210 |

**Artifact alignment (release blocker):** the published HF generator is
epoch169 (`docs/metrics_dapi_512_legacy_epoch169_val50.json`), while the
API/Docker local fallback prefers `generator_last.pt` (epoch200). Epoch200 is
better on FID/LPIPS/L1; epoch169 is better on SSIM/PSNR. Standardize one SHA
across HF, API, UI, Docker, and packages before v0.2.0 (recommended: publish
epoch200 and update `WEIGHT_FILES`).

**Package smoke check (2026-10-23):** `he_dapi.translate` loaded the HF
generator and ran `hemit_val_512.h5["he"][0]` on CPU; output was 512×512 uint8,
range 1–41, mean 7.81 (real DAPI mean 9.30), SHA-256
`a614e882…4704f08`. This verifies the published artifact loads and is
deterministic; it is not a quality claim.

**Reverse API smoke check (2026-10-23):** the backend now serves the corrected
`dapi2he_ch2_1024/generator_last.pt` (SHA-256 `1e9bbf2b…9689bc2`) instead of the
invalid aborted `dapi2he/cyclegan_G_last.pt`. A 512×512 val DAPI crop produced
a 512×512×3 H&E output (mean 181.59, SHA-256 `c201b78e…87db6ce`).

### Corrected 1024 `ch2` baselines — 50 val crops (2026-10-23)

Kaggle T4, 14 epochs at 512 then 2 × 500 steps at 1024 per direction. Target
verified as `mihc[...,2]`. Not a full-split evaluation and not release quality.

| Direction | px | FID ↓ | LPIPS ↓ | PSNR | SSIM | Pearson | L1 ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| H&E → DAPI (fine-tune) | 1024 | 215.38 | 0.3899 | 27.91 | 0.7733 | 0.5515 | 0.0250 |
| DAPI → H&E (fine-tune) | 1024 | 282.92 | 0.5317 | 15.42 | 0.4891 | 0.6644 | 0.1505 |
| H&E → DAPI (pretrain) | 512 | 184.73 | 0.4044 | 27.54 | 0.7431 | 0.2877 | 0.0296 |
| DAPI → H&E (pretrain) | 512 | 372.22 | 0.5908 | 15.13 | 0.4619 | 0.6448 | 0.1613 |

The forward model does not beat the 200-epoch legacy 512 model on
FID/LPIPS/PSNR/L1; the reverse model is weak (SSIM 0.489, PSNR 15.42) and shows
blur/contrast failure. Training caveats: only 500 fine-tune steps, 512px
pretrain on a 1024px file, DISTS disabled by a `use_dists` logic bug, and no
val-based checkpoint selection. JSON:
`docs/metrics_ch2_1024_{forward,reverse}_val50.json`.

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

- **DAPI→H&E is not release quality.** A corrected `ch2` reverse checkpoint
  exists, but val SSIM 0.489 / PSNR 15.42 indicates blur and colour/contrast
  mismatch. The older panCK `ch0` reverse checkpoint is not DAPI→H&E.
- **Corrected 1024 DAPI is undertrained.** The `ch2` forward model has only
  500 native-resolution fine-tune steps and does not beat legacy 512 on
  FID/LPIPS/PSNR/L1. Full 630/945 evaluation is still pending.
- **Provenance.** Verified 2026-10-23: legacy 512 H5 `dapi` exactly equals
  `mihc[...,2]`; the HF generator matches local epoch169 tensor-for-tensor.
  The demo/API still need the same SHA recorded in their manifests.
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
3. ~~Train a 1024px DAPI model on `ch2` (paired U-Net first).~~ First pass done;
   longer fine-tune and dual-branch comparison still required.
4. ~~Train a valid DAPI→H&E model on `ch2`.~~ First valid checkpoint done;
   quality gate not passed.
5. Evaluate full validation and test splits with mean ± std / CIs.
6. Publish weights with a manifest stating target channel and split only after
   metrics are reproducible.
7. Update PyPI, Hugging Face, Docker, and the demo only after the DAPI metrics
   are reproducible.
