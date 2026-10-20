# MEMORY - H&E <-> DAPI Session Checkpoint 2026-09-27

> **2026-10-20 correction:** HEMIT labels are `ch0=DAPI, ch1=CD3, ch2=panCK`.
> The previous 512px pipeline used `ch2`, so the epoch-199/200 run, current HF
> weights, demo, and `docs/metrics_eval.json` are a **panCK ablation, not DAPI**.
> Fixes: `scripts/data/pack_hemit.py`, `scripts/data/pack_hemit_1024.py`,
> `src/models/datasets.py`, `tests/test_hemit_channels.py`. Current canonical
> status: `README.md`, `docs/MODEL_CARD.md`, `docs/RESULTS.md`, `docs/TRACKER.md`.
> Kaggle T4 path is verified (2x T4 15.64GB); corrected 1024px DAPI training is
> the next milestone. Everything below this banner is historical unless it
> agrees with the 2026-10-20 docs.

> Updated 2026-10-18 15:25 UTC: 🟢 RUNNING PID1081 epoch183/200 (~91%). **ALL SHIPPED**: GitHub **Release v0.1.0** + prod promote (GHCR `:v0.1.0`+`:latest`); **HF model repo + FREE Static Space (in-browser ONNX) demo**; GitHub Pages; **PyPI packages `he-dapi` + `napari-he-dapi` v0.1.1 (alpha)** (`pip install he-dapi`); package + dummy-training tests in CI. Fixed cd-production (tag-push-only trigger + pinned checkout). ~17 left → ~Oct-20, pace ~2.4h/ep, best ep169 1.5302. KAGGLE OFF. Tokens: `~/.github_pat`, `~/.hf_token`, `~/.pypi_token` (all 600).

## Session log — 2026-10-18 (deliverables shipped)

Condensed story of the day. Machine-readable: `docs/progress.json:daily_log`; human tracker with a "Shipped 2026-10-18" table: `docs/TRACKER.md`; changelog: `CHANGELOG.md` ([v0.1.1]).

- **Training** unaffected all day: `PID1081` (epoch 182→183/200), reboot-proof via cron `logs/boot_launch.sh`; Kaggle STAND DOWN.
- **Release/CD:** fixed `cd-production` (release-event checkout bug → tag-push-only trigger + pinned `ref`); promoted **v0.1.0** → GHCR `:v0.1.0` + `:latest`.
- **Hugging Face:** model repo (weights) + **free Static Space** in-browser ONNX demo (Gradio Spaces now need PRO).
- **GitHub Pages:** landing live.
- **PyPI:** `he-dapi` + `napari-he-dapi` **v0.1.1 (alpha)** — inference library+CLI and npe2 napari plugin; token at `~/.pypi_token`.
- **Tests/CI:** new `packages` CI job; `training-smoke.yml` (manual/weekly); dummy smoke for paired & unpaired (`tests/test_training_dummy.py`).
- **Docs:** `DEPLOYMENT_GUIDE.md`, `PUBLISHING.md`, `LAUNCH_CHECKLIST.md`, `AGENTS.md` (manual prod approval rule).
- **Next:** ~17 epochs → ep200 (~Oct-20) → v1.0 (backend + frontend) → single launch; unpaired CycleGAN decision at ep200.

## Current Status (Updated 2026-10-03 14:43 UTC — syncs with `docs/progress.json:runs` + `docs/TRACKER.md:Live Runs`)

### Local Training — Product FULL No Compromise (RUNNING PID 935 10h, primary)
- **Run**: `PID 935` epoch58 step920/1858 (~50%), `57/200 done (28.5%)` (`checkpoint_epoch57.pt` + `checkpoint_last.pt` Oct-03 13:33, loss 1.8502 NEW LOW; ep54 1.8904, ep55 1.8806, ep56 1.8618), log `logs/train_paired.log`, 50s watch PID 971 → `logs/monitor.log`
- **Config**: `configs/hemit_dual_branch.yaml:38` `3717` `ngf 64 n_blocks 9` `batch 2+accum4 eff8` `epochs 200` `dists_on_cpu false` `max_cache 4` + `src/models/train_paired.py:61` `grad_accum + OOM guard + D-loss/components logging (future resumes)`
- **Health**: 4 epochs since resume all new lows, pace recovered ~2.2h/ep (reboot fixed thermal/swap); RAM ~54%/1747MB; 143/200 remaining to goal 200; gaps till 53 fixed (Dice 0.1, eval_offline + metrics_eval.json)
- **Gap**: NO validation in `train_paired.py` (`val_every: 5` unused) — training is G-loss-only; SSIM/PSNR/Dice deferred to Phase 4, do NOT disturb the running job to add it
- **Torch**: `torch 2.14.0+cpu` `torchvision 0.29.0` (py3.14, 2.3.0 unsat) via `uv pip --index-url https://download.pytorch.org/whl/cpu` `h5py 3.16, numpy 2.5.3`

### Kaggle — CANCELLED, STAND DOWN (post-mortem done, do NOT re-push yet)
- **Kernel**: `bikasash/he-dapi-product-gpu-train` v8 `CANCEL_ACKNOWLEDGED` during epoch19 (https://www.kaggle.com/code/bikasash/he-dapi-product-gpu-train); final log + checkpoints salvaged to `/tmp/kaggle_out_v8d`
- **Post-mortem**: **NO GPU** (`torch 2.10.0+cpu`, quota exhausted) — trained ep15–18 on CPU at `~2.8h/epoch`, **slower than local** (2.3h). Loss 2.2742→2.2094 falling but behind local (ep23 @ 2.1516). Fork resolved in local's favor; Kaggle ckpts kept in `/tmp` only, NOT copied over local
- **Fix chain (kept for next push)**: dataset uploader drops subdirs → ship `code_bundle.zip` flat (Kaggle extracts to `code_bundle/` prefix) → script merges up; newest-ckpt resume + fail-fast `sys.exit(2)`
- **Payload/stage**: `/tmp/kaggle_kernel_he_dapi` (in-sync with repo script) + `.tmp_kaggle_code` (flat layout only!)
- **Token**: saved `~/.kaggle/api_token` (chmod 600), used via `KAGGLE_API_TOKEN` env; never in repo
- **Rule**: re-push ONLY when GPU quota refreshes (weekly free 30h), resuming from latest local `checkpoint_epochN.pt`

### Previous Status (2026-09-27 10:11, superseded)

## Current Status (Updated 2026-09-19 — Legacy)

### Machine Profile (WSL2) - `src/utils/resource_monitor.py:1`, `wiki/01-machine-profile/`
- **CPU**: Intel i5-10210U @1.60GHz, 8 vCPU (4c/8t), no GPU (`nvidia-smi` not found)
- **RAM**: 3.7GB total, 2.6GB avail, 1GB swap (avail checked via `/proc/meminfo` fallback, `src/utils/resource_monitor.py:13`)
- **Disk**: 867GB free / 1007GB total (`df -h`), `data/raw 72G` + `data/processed 14G` = 86G used (was 12.8G on 08-31)
- **Budgets**: per-process <1.5GB, spike >500MB/5s → throttle, WSI canvas 512MB (vs 4.8G for 20k²), batch 2+grad_accum 4=eff 8
- **Verification**: `.venv/bin/python -m src.utils.resource_monitor --check` → 35% used, 2473MB avail, `Advised batch 4` for 512px

### Data Extraction — COMPLETE (user asked "we complete daat extraction ri8?" → YES)
- **HEMIT**: `data/raw/hemit/extracted/{train,val,test}/{input,label}/*.tif` = 5292 pairs (3717/630/945) — extracted from `HEMIT.7z` (44G raw) via `scripts/data/hemit_extract.py:1` → `scripts/data/pack_hemit.py:1` (streaming)
  - `data/processed/hemit_train_512.h5` 3.7G (3717), `hemit_val_512.h5` 684M (630), `hemit_test_512.h5` 968M (945) — resumable `.tmp` + `os.replace`, flush 500, skip if `he.shape[0]==len(pairs)`
  - `hemit_512.h5` combined 5292 (5.3G) — **previously OOM** via `np.concatenate` (10.6GB peak) → fixed Sep 19 to chunked copy B=100 (peak 140MB, `wiki/02-data-extraction/SKILL.md:1`)
  - Pipeline: `scripts/data/hemit_extract.py:6` (outer zip → `HEMIT.7z` → py7zr → `extracted`), `scripts/data/pack_hemit.py:86` (now streaming), `resources/datasets/prepare_patches.py:33` (512 central crop, lzf, gc)
- **BCI HER2** (alternative): `data/raw/bci_her2match/HE|IHC` → `scripts/data/pack_bci.py:1` streaming combined (peak 150MB, was 9GB), `wiki/02-data-extraction/OPTIMIZATION.md:1`
- **Paired fallback** (08-31): `maxpmx_*.h5` 4192 pairs (3353/419/420) — still valid
- **Unpaired H&E**: `mhist_256.h5` 498M + `mhist_512.h5` 1.9G (3152), `codex_512.h5`, `monuseg_256.h5`, `data/raw/wsi/CMU-1.svs`
- **Report**: `docs/DATA_EXTRACTION_REPORT.md:1` (08-31) + `docs/optimization/REPORT.md:1` (09-19) + `wiki/README.md:1`

### Optimization Done 2026-09-19 (9 wiki sub-agents, spike-guarded)

| # | Wiki Folder | Skill | File Patched | Spike Fixed |
|---|-------------|-------|--------------|-------------|
| 1 | `wiki/01-machine-profile/` | machine_profiler | `src/utils/resource_monitor.py:1` | 3.7GB profile + budgets |
| 2 | `wiki/02-data-extraction/` | extraction_optimizer | `scripts/data/pack_hemit.py:1`, `scripts/data/pack_bci.py:1`, `prepare_patches.py:1` | 10.6GB concat → 140MB streaming |
| 3 | `wiki/03-datasets/` | dataset_streamer | `src/models/datasets.py:26` | SWMR lazy open, mihc fallback, H5 for unpaired |
| 4 | `wiki/04-tiling/` | tiling_streamer | `src/utils/tiling.py:55` | 4.8GB canvas → memmap streaming `merge_tiles_streaming` |
| 5 | `wiki/05-wsi-loader/` | wsi_streamer | `src/utils/wsi_loader.py:76` | `asarray()` 4.8GB → `aszarr` zarr slice |
| 6 | `wiki/06-training/` | train_cpu_optimizer | `configs/hemit_dual_branch.yaml:35`, `configs/cyclegan_dapi2he.yaml:27`, `src/models/train_paired.py:61`, `train_unpaired.py:54` | batch 8→2, VGG 400MB off, grad_accum 4, num_workers 0, OOM catch |
| 7 | `wiki/07-inference/` | infer_streamer | `src/models/infer.py:17,108`, `src/backend/main.py:40` | full WSI 4.8GB → `infer_wsi_streaming` per-tile |
| 8 | `wiki/08-backend/` | backend_throttler | `src/backend/main.py:40` | 429 guard if used>85% |
| 9 | `wiki/09-orchestration/` | orchestrator | `Makefile:1`, `wiki/README.md:1` | resumable nohup + checkpoint |

- **Tests**: `tests/test_resource_guards.py:1` — 11 passed (resource, tiling streaming, pack streaming, SWMR, wsi aszarr, lowram configs, OOM guards, backend 429)
- **Env**: `.venv/bin/python` now has `numpy 2.5.3, pillow 12.3, h5py 3.16, tifffile 2026.9.15, zarr 3.4` via `uv pip`; `torch` still **not installed** (needs `torch==2.3.0` CPU ~2GB, install in steps)
- **Pyproject fix**: `pyproject.toml:31` removed `openseadragon` (broke `uv` resolve), added `lowram` extra

## Resumable Principle (still enforced)
For any long step, make resumable + checkpoint so crash can resume (since machine was crashing abruptly).

1. **Fetch**: `resources/datasets/download_hemit_resumable.sh:1` (curl -C -)
2. **Prepare**: `prepare_patches.py:33` + `scripts/data/pack_hemit.py:1` streaming + gc + `ResourceMonitor`
3. **Training paired**: `src/models/train_paired.py:98` `--resume auto`, `checkpoint_last.pt` every 1 epoch, now grad_accum + OOM guard + `dists_on_cpu false`
4. **Training unpaired**: `src/models/train_unpaired.py:90` same, `train_dummy(monitor, batch, grad_accum)`
5. **Configs**: `configs/hemit_dual_branch.yaml:35` + `cyclegan_dapi2he.yaml:27` lowram defaults
6. **Inference**: `infer.py:108` `--stream` + `infer_wsi_streaming` (+ `tiling.py:55` streaming)
7. **Backend**: `src/backend/main.py:40` 429 throttling

## Tracker Added 2026-09-19 (must follow till end, system weak)
- **Tracker files**: `docs/TRACKER.md:1` (human, 38% overall, Phase1 DONE 100%, Phase2 NEXT 10%) + `docs/progress.json:1` (machine) + `src/utils/progress_tracker.py:1` (`--check/--update/--enforce`)
- **Rule**: Every Phase 2-5 step must keep today's low-RAM pattern (streaming, `ResourceMonitor` `src/utils/resource_monitor.py:1`, batch 2+accum4, SWMR, memmap, 429 guard). Enforce via `PYTHONPATH=. .venv/bin/python -m src.utils.progress_tracker --enforce` (all ✅ 09-19) + `tests/test_resource_guards.py:1` (11 passed).
- **Usage**: `PYTHONPATH=. .venv/bin/python -m src.utils.progress_tracker --update phase2 --percent 25 --note "dummy passed"` → updates `docs/progress.json:1` + log, overall % auto.

## Next Phase: Proper Training (TODO)

```bash
# 1. Install torch CPU (do in chunks to avoid RAM spike on 3.7GB box)
uv pip install --python .venv/bin/python torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
# or: .venv/bin/pip install torch --index-url ... --no-cache

# 2. Verify datasets still stream low-RAM
PYTHONPATH=. .venv/bin/python -c "from src.models.datasets import PairedHEMITDataset; ds=PairedHEMITDataset('data/processed/hemit_train_512.h5', indices=range(2)); print(len(ds), ds[0]['he'].shape)"

# 3. Dummy smoke (already verified 09-19)
PYTHONPATH=. .venv/bin/python src/models/train_paired.py --config configs/hemit_dual_branch.yaml --dummy --optimize-for lowram
PYTHONPATH=. .venv/bin/python src/models/train_unpaired.py --config configs/cyclegan_dapi2he.yaml --dummy --optimize-for lowram

# 4. Full training (resumable, checkpoint_last.pt, effective batch 8 via 2+accum4)
# Paired H&E->DAPI (pix2pix, ~3717 samples, ~1858 steps/epoch at batch2)
nohup env PYTHONPATH=. .venv/bin/python src/models/train_paired.py --config configs/hemit_dual_branch.yaml --optimize-for lowram 2>&1 | tee /tmp/train_paired.log &
# Unpaired DAPI->H&E (CycleGAN, 4 nets heavier, batch2+accum4)
nohup env PYTHONPATH=. .venv/bin/python src/models/train_unpaired.py --config configs/cyclegan_dapi2he.yaml --optimize-for lowram 2>&1 | tee /tmp/train_unpaired.log &

# 5. Monitor RAM spikes (watch)
PYTHONPATH=. .venv/bin/python -m src.utils.resource_monitor --watch 60 &

# 6. Resume if crash (auto-detects checkpoint_last.pt)
PYTHONPATH=. .venv/bin/python src/models/train_paired.py --config configs/hemit_dual_branch.yaml --resume auto
ls -lh resources/models/checkpoints/*/checkpoint_*.pt

# 7. Inference after train
PYTHONPATH=. .venv/bin/python src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/synth.png
PYTHONPATH=. .venv/bin/python src/models/infer.py --input data/raw/wsi/CMU-1.svs --direction he2dapi --wsi --stream --out /tmp/wsi.tiff

# 8. Health + docs
cat docs/optimization/REPORT.md
cat wiki/README.md
PYTHONPATH=. .venv/bin/python tests/test_resource_guards.py  # 11 passed 09-19
```

## Legacy 2026-08-31 Snapshot (kept for history)
### Status 2026-08-31 17:02
- **HEMIT download**: PAUSED/STALLED `PID 19727` dead (no curl in `ps aux | grep curl`, last Modify 16:31:20). 
  - File: `data/raw/hemit/hemit.zip` = **9.9G** now (6176604160 resume offset + 4.08G chunk = ~10.2G total, ~97% done). Grew 6.2G->9.9G after last checkpoint.
  - Logs: `/tmp/hemit_download.log` (99K, frozen at 16:31), `/tmp/hemit_nohup.log` (99K), `/tmp/curl_hemit_resume2.log` (97K). Stalled 31m, needs manual resume.
  - Resume: `bash resources/datasets/download_hemit_resumable.sh` (curl -C - idempotent, will continue from 9.9G).
  - Next after done: `unzip` / `7z x` (contains `HEMIT.7z` inside zip) -> `python resources/datasets/prepare_patches.py --input data/raw/hemit --patch 512 --out data/processed/hemit_512.h5`
- **Paired data DONE (MVP substitute)**: `data/processed/maxpmx_*.h5` 4192 pairs (3353/419/420 split) 512x512 `docs/DATA_EXTRACTION_REPORT.md:7` — HEMIT gated 404 replaced by `maxpmx/dataset_dapi-he` HF.
- **Unpaired H&E DONE**: `data/processed/mhist_{256,512}.h5` 3152 HP/SSA `data/raw/mhist/MHIST/images/*.png`
- **Other**: `codex_512.h5` (35), `monuseg_256.h5` (37), `data/raw/wsi/CMU-1.svs` + synthetic WSI, examples `data/examples/*.png`
- **Disk**: `data/raw 5.0G + data/processed 7.8G = 12.8G` (HEMIT will add ~10G)

## Current Status (Updated 2026-08-31 17:02)
- **HEMIT download**: PAUSED/STALLED `PID 19727` dead (no curl in `ps aux | grep curl`, last Modify 16:31:20). 
  - File: `data/raw/hemit/hemit.zip` = **9.9G** now (6176604160 resume offset + 4.08G chunk = ~10.2G total, ~97% done). Grew 6.2G->9.9G after last checkpoint.
  - Logs: `/tmp/hemit_download.log` (99K, frozen at 16:31), `/tmp/hemit_nohup.log` (99K), `/tmp/curl_hemit_resume2.log` (97K). Stalled 31m, needs manual resume.
  - Resume: `bash resources/datasets/download_hemit_resumable.sh` (curl -C - idempotent, will continue from 9.9G).
  - Next after done: `unzip` / `7z x` (contains `HEMIT.7z` inside zip) -> `python resources/datasets/prepare_patches.py --input data/raw/hemit --patch 512 --out data/processed/hemit_512.h5`
- **Paired data DONE (MVP substitute)**: `data/processed/maxpmx_*.h5` 4192 pairs (3353/419/420 split) 512x512 `docs/DATA_EXTRACTION_REPORT.md:7` — HEMIT gated 404 replaced by `maxpmx/dataset_dapi-he` HF.
- **Unpaired H&E DONE**: `data/processed/mhist_{256,512}.h5` 3152 HP/SSA `data/raw/mhist/MHIST/images/*.png`
- **Other**: `codex_512.h5` (35), `monuseg_256.h5` (37), `data/raw/wsi/CMU-1.svs` + synthetic WSI, examples `data/examples/*.png`
- **Disk**: `data/raw 5.0G + data/processed 7.8G = 12.8G` (HEMIT will add ~10G)

## Resumable Principle Enforced (per user request)
For any long step (fetch/train/patch), make resumable + checkpoint so crash can resume.

1. **Fetch**: `curl -L -C - --retry 20 --retry-delay 10 --speed-limit ...` + `nohup bash -c "curl ... | tee -a /tmp/hemit_download.log"` -> `resources/datasets/download_hemit_resumable.sh:1` (idempotent)
2. **Prepare patches**: `resources/datasets/prepare_patches.py:33` now writes to `.tmp` + `os.replace()` atomic + `flush()` every 500 + skip if `he.shape[0]==len(pairs)` (re-run safe)
3. **Training paired**: `src/models/train_paired.py:98` added `--resume auto|PATH`, auto-detects `checkpoint_last.pt`, saves `checkpoint_epoch{epoch}.pt` + `checkpoint_last.pt` every `save_every:1` with `G/D/optim/epoch` (atomic tmp rename)
4. **Training unpaired**: `src/models/train_unpaired.py:90` same `--resume`, `checkpoint_dir: resources/models/checkpoints/dapi2he`
5. **Configs**: `configs/hemit_dual_branch.yaml:35` + `configs/cyclegan_dapi2he.yaml:27` added `save_every: 1`, `resume: auto`, `checkpoint_dir`

## Tomorrow Resume Commands
```bash
# 1. Check HEMIT
du -sh data/raw/hemit/hemit.zip; tail -n 20 /tmp/hemit_download.log; ps -o pid,ppid,stat,cmd -p 19727 2>&1 | head
# If dead, resume:
bash resources/datasets/download_hemit_resumable.sh
# Or manual:
nohup bash -c 'curl -L -C - --retry 20 --retry-delay 10 --retry-all-errors "https://data.mendeley.com/public-api/zip/3gx53zm49d/download/1" -o data/raw/hemit/hemit.zip 2>&1 | tee -a /tmp/hemit_download.log' > /tmp/hemit_nohup.log 2>&1 &

# 2. After zip complete, verify + extract (needs p7zip):
# apt install p7zip-full  # if missing
# unzip -l data/raw/hemit/hemit.zip | head
# 7z x data/raw/hemit/hemit.zip -o/tmp/hemit_test && 7z x /tmp/hemit_test/HEMIT.7z -odata/raw/hemit/
# python resources/datasets/prepare_patches.py --input data/raw/hemit --patch 512 --out data/processed/hemit_512.h5

# 3. Training (already resumable):
# Paired H&E->DAPI
python src/models/train_paired.py --config configs/hemit_dual_branch.yaml --resume auto
# Unpaired DAPI->H&E
python src/models/train_unpaired.py --config configs/cyclegan_dapi2he.yaml --resume auto
# Checkpoints: ls -lh resources/models/checkpoints/*/checkpoint_*.pt
```

## BCI Status
- Planned in `docs/WORKPLAN.md:30` for unpaired test (consecutive sections, not aligned) but NOT downloaded. User asked what needed: needs Go/No-Go + Grand Challenge login or public mirror approval. Skip is OK for MVP (MHIST covers unpaired H&E).

## Infra
- Env: `/tmp/he-venv/bin/python` (h5py ok, torch not installed yet, needs `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)
- Tests: `tests/test_imports.py:1` smoke tests (tiling, datasets, infer)
- Tokens/auth (reboot-safe, lesson 2026-10-14): GitHub `~/.github_pat`, Kaggle `~/.kaggle/api_token`, HF `~/.hf_token` (all 600, home dir, never in repo) + `git credential.helper` wired to `~/.github_pat`. Full map + reboot-relaunch order: `wiki/09-orchestration/SKILL.md:Token`.
- Reboot auto-recovery (lesson 2026-10-17): cron `@reboot` + `*/5` runs `logs/boot_launch.sh` (idempotent, flock-guarded) which relaunches keepalive→train+watch if dead. Supersedes manual reboot relaunch; survives reboots with no sudo/linger.
- CI/CD (audited 2026-10-18): `guards` green on every push; `cd-staging` now also builds on `main` (paths-filtered docker/src/resources) so `:staging` stays fresh (refreshed to `d36922c` Oct-18); `.dockerignore` added (excludes 14GB checkpoints); `main` protected (no force-push/delete, `resource-guards` required for PRs); `cd-production` promotes via `v*.*.*` tag + prod-env approval (untested — dry-run pending). Full runbook: `wiki/09-orchestration/SKILL.md` CI/CD section.

## Links
- HEMIT paper: https://arxiv.org/html/2403.18501v1 | Mendeley: 3gx53zm49d
- MaxPMX HF: maxpmx/dataset_dapi-he | MHIST: NYCU-PCSxNTHU-MIS/MHIST
- Workplan: docs/WORKPLAN.md | Extraction report: docs/DATA_EXTRACTION_REPORT.md

Generated: 2026-08-31 16:12 via nohup PID 19727 | Updated: 2026-08-31 17:02 stalled at 9.9G (PID dead)
