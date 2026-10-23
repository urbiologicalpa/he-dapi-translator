# Legacy H&E → DAPI baseline — 200 epochs, 512px

> **CORRECTION 2026-10-23: this is the real H&E→DAPI model, not panCK.**
> The legacy 512 H5 stores `dapi = mihc[...,2]`, which the channel audit
> identifies as DAPI (blue). This 512px model is the **real DAPI baseline**. The
> 1024 `ch0` model is the panCK ablation.
> Do not delete this run; it has been relabeled under ADR-014. See
> [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md).

**Status: internal benchmark artifact. The title/filename are historical and
will be corrected. Not a product release.**

This run exists to validate the pipeline, resumability, and evaluation path. It
must not be presented as H&E → DAPI.

## Provenance

| Item | Value |
|---|---|
| Target | panCK, HEMIT label `ch2` (wrong target for this product) |
| Data | `data/processed/hemit_{train,val,test}_512.h5` (legacy build) |
| Train / val / test | 3,717 / 630 / 945 |
| Resolution | 512px central crop |
| Epochs | 200 |
| Checkpoint | `resources/models/checkpoints/he2dapi/checkpoint_epoch200.pt` |
| Checkpoint SHA-256 | `6789d7d244cc6ac93dffd98a30ed2c17498aa7ee6081560dedac5173b44042aa` |
| Generator SHA-256 | `849697d73e9315a558d5963d1fa256e79fdaeefd1eb541017a1f196596f8891a` |
| Git SHA at evaluation | `b2eebf8b86885f1ce3c024281996cdec82166e29` |
| Finished | 2026-10-20 19:34 UTC |
| Hardware | 3.7 GB RAM CPU box |

Final training losses: **G 1.5796**, **D 0.7397**.

## Full evaluation

Per-image metrics, evaluated on the complete official splits. SSIM and Pearson
were computed with `skimage` / NumPy; Dice uses threshold 0.1.

| Split | n | L1 ↓ | PSNR dB ↑ | SSIM ↑ | Pearson ↑ | Dice ↑ | D loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| val | 630 | 0.0252 | 29.23 | 0.7389 | 0.5072 | 0.1526 | 0.8012 |
| test | 945 | 0.0217 | 30.65 | 0.7890 | 0.8781 | 0.4833 | 0.7007 |

Raw JSON: [`metrics_panck_512.json`](metrics_panck_512.json).

Caveats:

- The target is panCK, so these numbers cannot be compared with HEMIT DAPI
  paper numbers or used in the DAPI benchmark table.
- The val/test Pearson gap is large; inspect per-image failures before treating
  either number as stable.
- No standard deviations / bootstrap CIs are recorded yet. Add them before using
  this as a formal benchmark.
- Dice at threshold 0.1 is not comparable to segmentation papers using 0.5.
- The model is the U-Net baseline, not the dual-branch Swin/FMF architecture.

## What this run proves

- The streaming HDF5 loader and 512px CPU training path work.
- Checkpoint/resume and long-run RAM behavior are stable.
- Full-split evaluation is now automated and labeled by target channel.
- The pipeline can produce and persist a complete model without OOM.

What it does **not** prove: DAPI quality, bidirectional capability, reverse
translation, or clinical validity.
