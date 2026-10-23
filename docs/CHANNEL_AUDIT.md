# HEMIT channel audit — BLOCKER (2026-10-23)

> **Status: ACCEPTED; CORRECTION IN PROGRESS (2026-10-23).** Do not publish
> DAPI claims or promote the 1024px `ch0` models until the rebuild and
> re-verification are complete.
> **Correction: `ch2 = DAPI`, `ch0 = panCK`, `ch1 = CD3`.** ADR-014 reverses
> ADR-001. This changes published weights, metrics, and the release narrative.

## Finding

The raw HEMIT `label` TIFF is an **RGB pseudo-colour composite**. The marker
that looks like nuclei is in the **blue** channel (`ch2`), not `ch0`:

| Raw channel | Colour | Morphology | Most likely marker |
|---|---|---|---|
| `ch0` | red | diffuse cytoplasmic / membranous | **panCK** |
| `ch1` | green | sparse single cells | **CD3** |
| `ch2` | blue | dense discrete nuclei | **DAPI** |

The HEMIT paper says: *"DAPI is shown in blue, panCK in red, and CD3 in green"*
(arXiv:2403.18501). In an RGB TIFF, blue is channel 2, red is channel 0, green
is channel 1. That matches the visual evidence and the data below.

The comment in the official `post_process.py` (`ch0 #dapi, ch1 #CD3, ch2
#Panck`) is **inconsistent with the raw TIFF**; do not use it as channel-order
ground truth.

## Evidence

### 1. Pixel statistics (val, 50 patches)

| Channel | mean | std | 99th pct | Correlation with H&E nuclei (`255 - gray`) |
|---|---:|---:|---:|---:|
| `ch0` | 4.01 | 8.41 | 44 | **0.296** |
| `ch1` | 5.21 | 16.99 | 82 | 0.241 |
| `ch2` | 11.16 | 10.97 | 51 | **0.578** |

`ch2` is the channel that tracks nuclei.

### 2. HDF5 provenance

| File | Stored `dapi` matches | Correlation |
|---|---|---:|
| `hemit_val_512.h5` (legacy) | `mihc[...,2]` | **1.0000** |
| `hemit_val_1024.h5` ("corrected") | `mihc[...,0]` | **1.0000** |

So the legacy 512 pipeline actually used **ch2 = DAPI**, and the new 1024
pipeline uses **ch0 = panCK**.

### 3. Model behaviour

- Legacy 512 `he2dapi/checkpoint_epoch200.pt` (trained on the 512 H5, i.e.
  `ch2`) produces **nuclear** staining that matches the real DAPI pattern.
  Its full-split metrics (val SSIM 0.7389, test 0.7890) are close to the paper's
  **DAPI** SSIM 0.791, not its panCK SSIM 0.901.
- The 1024 `ch0` "DAPI" model (val/test SSIM 0.670/0.725) is the actual
  **panCK** ablation.
- The completed 256px CycleGAN `R-006` and the paired reverse baseline were
  both trained on `ch0`; they are **panCK→H&E**, not DAPI→H&E.

### 4. R-006 completion (2026-10-22 18:55 UTC)

`bikasash/he-dapi-cyclegan-hemit-dapi2he-256` completed with exit 0 after 10
epochs (929 steps/epoch, AMP, ~414s/epoch). The log confirms H5-only
(`HE: 0 files + H5 …`, `synthetic=False`) and CUDA. Its kernel eval
(`--num 630`, no `--metrics`) reported L1 0.2543, PSNR 10.61, SSIM 0.2051,
Pearson 0.1537; FID/LPIPS were `null` because the metric flag was not passed.
A local re-eval on the same first 50 val crops produced FID 367.97,
LPIPS 0.5948, PSNR 11.37, SSIM 0.2216, L1 0.2292, Pearson 0.1596
(`docs/metrics_cyclegan_hemit_256_val50.json`). **This is panCK→H&E, not DAPI.**

### 5. Visual artifacts

- `docs/qualitative_dapi2he/mihc_channels_preview.png` — all three channels.
- `docs/qualitative_dapi2he/channel_order_check.png` — official composite vs
  file RGB; file RGB shows blue nuclei.
- `docs/qualitative_dapi2he/legacy512_dapi_check.png` — H&E, H5 `dapi`, model
  output, `ch0`; the legacy model follows nuclei.
- `docs/qualitative_dapi2he/mihc_rgb_preview.png` — raw RGB composite.

## Consequences

1. **ADR-001 is wrong.** The correct mapping is `ch2=DAPI`, `ch0=panCK`,
   `ch1=CD3` for the raw TIFF. Proposed **ADR-014** supersedes it.
2. **The "panCK ablation" is the real DAPI model.** The legacy 512 checkpoint,
   public HF weights, demo, and `docs/metrics_eval.json` should be re-labelled
   as **H&E→DAPI**, not panCK. Verify the HF/package artifacts before changing
   public text.
3. **The "corrected DAPI 1024" is panCK.** The 1024 H5, forward 1024 model,
   reverse 1024 model, and the 256px CycleGAN R-006 are panCK, not DAPI.
4. **R-006 is not a DAPI result.** Its val-50 numbers
   (FID 367.97, LPIPS 0.5948, SSIM 0.2216, Pearson 0.1596) are a panCK→H&E
   unpaired result. Do not use them in `docs/selection_reverse.md` as DAPI.
5. **The reverse direction has no valid DAPI model.** The paired reverse
   (`dapi2he_1024`) and R-006 used panCK. A fair DAPI→H&E comparison must be
   re-run on `ch2`.

## Accepted correction (in progress)

1. Adopt `DAPI_CHANNEL = 2` in builders, loader, tests, configs, and docs;
   record ADR-014 and mark ADR-001 superseded.
2. Rebuild `hemit_{train,val,test}_1024.h5` with `dapi = mihc[...,2]`; rebuild
   the 256px derivatives. Keep the raw `mihc` for audit.
3. Re-label:
   - legacy 512 `he2dapi` → **H&E→DAPI (real)**;
   - 1024 `ch0` forward/reverse and R-006 → **panCK ablation/invalid for DAPI**.
4. Re-evaluate the legacy 512 DAPI model on val/test with the corrected target
   and record SHA-256 provenance.
5. If reverse DAPI→H&E is still a product requirement, re-run the paired
   baseline and the CycleGAN on `ch2` (new Kaggle code/data bundles; do not
   reuse the ch0 checkpoints).
6. Correct `README.md`, `docs/PLAN.md`, `RESULTS.md`, `TRACKER.md`,
   `progress.json`, `ABLATION_PANCK.md`, package/HF model cards, and the
   benchmark tables.

## Do not

- Do not publish, tag, or promote the 1024 `ch0` models as DAPI.
- Do not present the legacy 512 model as a panCK ablation.
- Do not launch another DAPI run on the current `ch0` H5.
- Do not delete the raw `mihc` or the ch0 checkpoints; they are the panCK
  ablation evidence.

## Provenance

- Audit date: 2026-10-23 UTC.
- Raw data: `data/raw/hemit/extracted` (5,292 pairs, 1024px RGB TIFFs).
- Legacy 512 H5: `data/processed/hemit_{split}_512.h5`.
- New 1024 H5: `data/processed/hemit_{split}_1024.h5`.
- Paper: arXiv:2403.18501, "DAPI is shown in blue, panCK in red, CD3 in green".
- Dataset README: `BianChang/HEMIT-DATASET`.
