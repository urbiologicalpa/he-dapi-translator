# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project aims for
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 2026-10-20

Documentation accuracy and data-correction pass. No new model weights.

### Fixed
- **DAPI channel order:** HEMIT labels are `ch0=DAPI`, `ch1=CD3`,
  `ch2=panCK`. Previously `ch2` was used as the target, so the epoch-199/200
  run, current demo/package weights, and `docs/metrics_eval.json` are a
  **panCK ablation**, not DAPI. Fixes in `scripts/data/pack_hemit.py`,
  `scripts/data/pack_hemit_1024.py`, and `src/models/datasets.py`.
- Added `tests/test_hemit_channels.py` to lock the channel order and loader
  fallback to `ch0`.

### Added
- `scripts/data/pack_hemit_1024.py`: native 1024px, Macenko, gzip/lzf,
  resumable HEMIT builder with channel metadata.
- `docs/DATA.md`, `docs/MODEL_CARD.md`, `docs/RESULTS.md`: explicit dataset,
  model, and evaluation status, including the panCK/DAPI correction.
- `scripts/` layout for data and Kaggle entry points.

### Changed
- README and package READMEs now describe the **U-Net baseline** that is
  actually implemented, not the planned dual-branch Swin/FMF model.
- Removed stale "DAPI result" and "live 181/200" claims. Current status is
  stated explicitly, including the missing reverse model and full evaluation.

## [v0.1.1] — 2026-10-18

Distribution + product surfaces. Repo release stays **v0.1.0**; the pip
packages are versioned independently at **0.1.1 (alpha)**.

### Added
- **PyPI packages:** `he-dapi` (inference library + `he-dapi` CLI) and
  `napari-he-dapi` (npe2 plugin). Both marked `Development Status :: 3 - Alpha`.
  Weights auto-download from the HF model repo.
- **Hugging Face:** model repo (weights) + free **Static Space** running the
  model in-browser via ONNX Runtime Web.
- **Tests:** package tests (`packages/*/tests`), paired + unpaired `--dummy`
  smoke tests (`tests/test_training_dummy.py`); CI `packages` job; manual/weekly
  `training-smoke` workflow.
- **Docs:** `docs/DEPLOYMENT_GUIDE.md`, `docs/PUBLISHING.md`,
  `docs/LAUNCH_CHECKLIST.md`, `AGENTS.md` (operating rules incl. manual prod approval).
- **Contributor on-ramp:** `good first issue` (#6–#10) + `help wanted` (#1, #3).

### Fixed
- `cd-production` failed on release events (`fetch-tags` broke checkout: *"Cannot
  fetch both <sha> and refs/tags/vX"*) → tag-push-only trigger + pinned `ref`.
- Stopped tracking the live training profile snapshot
  (`wiki/06-training/profile_paired.json`); kept on disk, now git-ignored.

## [v0.1.0] — 2026-10-18

First public milestone: streaming data pipeline, product-grade paired training
in progress, backend + Docker + CI/CD in place.

### Added
- **Data pipeline (streaming):** HEMIT 5,292 pairs (3717/630/945, 512px) via
  `scripts/data/hemit_extract.py` → `scripts/data/pack_hemit.py` (chunked copy, 10.6GB → 140MB peak);
  BCI HER2 and MHIST builders.
- **Models:** U-Net Pix2Pix baseline for H&E→DAPI; PatchGAN discriminator.
  The config name `hemit_dual_branch.yaml` and the "dual-branch" description
  referred to a planned CNN-ResNet + Swin + FMF generator that was **not
  implemented** at v0.1.0. The CycleGAN reverse trainer is a smoke-test skeleton.
- **Training:** resumable `train_paired.py` / `train_unpaired.py`, per-epoch
  atomic checkpoints, `--resume auto`, OOM guard, grad-accum (eff. batch 8).
- **Inference:** patch + per-tile streaming WSI translation (`infer.py`,
  `infer_wsi_streaming`), ONNX export hook.
- **Backend:** FastAPI `/health`, `/models`, `/translate`, `/translate/wsi` with
  a 429 RAM guard; drag-and-drop frontend scaffold.
- **Low-RAM guardrails:** `ResourceMonitor`, streaming tiling/WSI, and
  `tests/test_resource_guards.py` (11 regression tests, mirrored in CI).
- **CI/CD:** `guards` on every push; `cd-staging` builds an immutable image to
  GHCR (paths-filtered); `cd-production` promote path (tag + env approval).
- **Operations:** crash-resume across reboots via cron `@reboot`+`*/5`
  `logs/boot_launch.sh`; progress tracker (`docs/progress.json`, `docs/TRACKER.md`).
- Docs: WORKPLAN, per-stage optimization wiki (9 tracks), Kaggle offload
  post-mortem, citation file, contributing guide, code of conduct.

### Status
- Paired training live at **181/200 epochs** (G-loss 2.2603 → 1.5302, best ep169).
  This run was later identified as the **panCK ablation** (wrong HEMIT channel);
  see the Unreleased section.
- Full 630-image validation eval and Docker Hub mirror are on the roadmap.
