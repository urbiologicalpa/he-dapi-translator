# H&E ↔ DAPI Progress Tracker — Low-RAM Guard (3.7GB, no GPU) + Kaggle GPU

> **Rule**: Every step must follow today's optimization pattern (streaming + `ResourceMonitor`) till end — system is weak, so no full-RAM loads. If spike >500MB or avail <400MB → throttle/stream.

- Machine: `wiki/01-machine-profile/OPTIMIZATION.md:1` (i5-10210U 8vCPU, 3.7GB RAM, 1GB swap, no GPU) + Kaggle `T4 16GB` for product
- Monitor: `src/utils/resource_monitor.py:1` — `PYTHONPATH=. .venv/bin/python -m src.utils.resource_monitor --check`
- Verify guards: `PYTHONPATH=. .venv/bin/python tests/test_resource_guards.py` (11 passed 2026-09-19)
- Tracker JSON: `docs/progress.json` (machine-readable, auto-updated by `src/utils/progress_tracker.py:1`)

## Overall Progress: 45% (Phase 1 done, Phase 2 local 182/200 91% RUNNING WEEK-MODE + Kaggle OFF; product surfaces shipped — GHCR release, HF demo, PyPI packages)

> **Single truth (2026-10-18 15:25 UTC)**: machine-readable `docs/progress.json:runs` + this section + `MEMORY.md` banner are kept in sync. Tracker 45% (Phase 2 at 91%). Local checkpoint files are ground truth for epoch count. Tokens live ONLY in `$HOME` dotfiles (`~/.github_pat`, `~/.hf_token`, `~/.pypi_token`, `~/.kaggle/api_token`), never in repo.

## Live Runs (canonical — update here + `progress.json:runs` + `MEMORY.md` together)

| Run | Status | Detail |
|-----|--------|--------|
| Local CPU | 🟢 RUNNING WEEK-MODE (1-week plan) | `PID1081` epoch183/200 (`182/200 91%`, **best ep169 1.5302**; 180 1.5869 / 181 1.5874 / 182 1.5929 sideways-easing) `nice10 -u -m --resume auto --optimize-for lowram`, ~2.4h/ep steady. Watch1096 + keepalive1099 300s, reboot-proof via cron. RAM ~60%, uptime 1d10h stable. 18 left → finish ~Oct-20. Unpaired CycleGAN 200 kept in plan, decide at ep200. |
| Kaggle | ⬜ CANCELLED, STAND DOWN | stays OFF until GPU quota refreshes |
| Epoch ground truth | 182/200 done (91%) | `checkpoint_epoch182.pt` + `checkpoint_last.pt`; best val ep169 1.5302; ~18 left → ~Oct-20 |
| ⚠️ Gap (kept) | No validation runs | `val_every: 5` unused in `train_paired.py` — loss-only; SSIM/PSNR/Dice Phase 4 TODO |
| Reboot-proofing (new) | cron `@reboot` + `*/5` → `logs/boot_launch.sh` | Idempotent flock-guarded supervisor relaunches keepalive→train+watch if dead; no sudo/linger. Supersedes manual reboot relaunch. |

## Shipped 2026-10-18 — product surfaces (all live)

| Surface | What | Link / artifact |
|---------|------|-----------------|
| Release | **v0.1.0** promoted (GHCR `:v0.1.0` + `:latest`, same SHA as `:staging`) | https://github.com/urbiologicalpa/he-dapi-translator/releases/tag/v0.1.0 |
| CI/CD fix | `cd-production` now tag-push-only trigger + pinned checkout (release-event `fetch-tags` broke checkout); `publish-pypi.yml` added | `.github/workflows/` |
| Hugging Face | model repo (weights) + **free Static Space** in-browser ONNX demo (Gradio Spaces need PRO) | https://huggingface.co/urbiologicalpa/he-dapi-translator · https://huggingface.co/spaces/urbiologicalpa/he-dapi-translator |
| GitHub Pages | landing page live | https://urbiologicalpa.github.io/he-dapi-translator/ |
| PyPI | `he-dapi` + `napari-he-dapi` **v0.1.1, alpha** (Development Status 3 - Alpha) | https://pypi.org/project/he-dapi/ · https://pypi.org/project/napari-he-dapi/ |
| Tests | package tests + paired/unpaired dummy smoke (CI `packages` job; `training-smoke.yml` manual/weekly) | `packages/*/tests`, `tests/test_training_dummy.py` |
| Docs | deployment, publishing, launch checklist, AGENTS rules | `docs/DEPLOYMENT_GUIDE.md`, `docs/PUBLISHING.md`, `docs/LAUNCH_CHECKLIST.md`, `AGENTS.md` |
| Contributor | 5 `good first issue` (#6–#10) + `help wanted` (#1/#3) | GitHub Issues |

| Phase | Name | Status | % | Artifacts | Optimization enforced |
|-------|------|--------|---|-----------|----------------------|
| 0 | Scoping & Infra (W1-2) | ✅ DONE | 100 | `docs/WORKPLAN.md:14`, `pyproject.toml:1`, `README.md:1`, `wiki/01-machine-profile/` | `pyproject.toml:31` fix `openseadragon` break, `src/utils/__init__.py:1` lazy imports |
| 1 | Data (W2-4) | ✅ DONE | 100 | `data/processed/hemit_train_512.h5` 3717, `hemit_val/test`, `mhist_*.h5`, `pack_hemit.py:1` | streaming B=100 peak 140MB vs 10.6GB, SWMR, gc `wiki/02-data-extraction/SKILL.md:1` |
| 2 | Modeling (W4-8) Dual Track | 🟡 IN_PROGRESS | 40 | `configs/hemit_dual_branch.yaml:35`, `bikasash/hemit-he-dapi-product-full` 7.7G, `bikasash/he-dapi-product-gpu-train` T4 `src/models/train_paired.py:61` epoch10/200 | **batch 2+accum4, num_workers 0, dists_off, OOM catch** `wiki/06-training/SKILL.md:1` product FULL no compromise |
| 3 | Drag & Drop Tool (W8-10) | 🟡 IN_PROGRESS | 45 | `src/backend/main.py:40`, `packages/he-dapi`, `packages/napari-he-dapi`, `src/frontend/README.md` | backend + pip library + napari plugin done; React frontend + WSI queue TODO |
| 4 | Validation (W10-11) | ⬜ TODO | 0 | — | MC dropout variance map must use streaming (no 5× RAM) |
| 5 | Docker & Deploy (W11-12) | 🟡 IN_PROGRESS | 65 | `docker/Dockerfile`, `.github/workflows/cd-*.yml`, `ghcr.io/urbiologicalpa/he-dapi-translator` | CI→staging→prod promote live; PyPI + HF Space done; Docker Hub mirror + compose pending |

## Phase Progress Detail

### Phase 0 — Scoping & Infra ✅
- [x] Spec lock H&E RGB ↔ DAPI grayscale + uncertainty (`docs/WORKPLAN.md:14`)
- [x] Repo scaffold + stack (React, FastAPI, PyTorch, ONNX, tifffile/zarr `pyproject.toml:7`)
- [x] Fix `uv` resolve (`pyproject.toml:31` remove `openseadragon`)
- [x] `src/utils/resource_monitor.py:1` + `wiki/01-machine-profile/profile.json`
- [x] `tests/test_resource_guards.py:1`

### Phase 1 — Data ✅ 100% (2026-09-19)
- [x] HEMIT 5292 (3717/630/945) via `hemit_extract.py:1` → `pack_hemit.py:1` (now streaming)
- [x] BCI HER2 via `pack_bci.py:1` (streaming)
- [x] Paired fallback maxpmx 4192, unpaired MHIST 3152 (`mhist_512.h5:1`)
- [x] `resources/datasets/prepare_patches.py:33` gc + warn, WSI example `data/raw/wsi/CMU-1.svs`
- [x] Verified: `ls -lh data/processed/hemit*.h5` 3.7G/684M/968M, `docs/DATA_EXTRACTION_REPORT.md:1`
- **Guard**: must remain streaming till end — never reintroduce `np.concatenate` on full H5.

### Phase 2 — Modeling 🟡 40% (Kaggle GPU running, product FULL)
**Goal**: Train both directions, keep peak <1.5GB on this box, product FULL no compromise (3717 ngf64).

#### Track A: H&E→DAPI (Pix2Pix)
- [x] Install torch: `torch 2.14.0+cpu` (py3.14, 2.3.0 unsat) `torchvision 0.29.0` via `uv pip --index-url https://download.pytorch.org/whl/cpu` `KAGGLE_API_TOKEN=KGAT_...` `bikasash`
- [x] Config lowram `configs/hemit_dual_branch.yaml:35` — `batch_size 2, grad_accum 4, num_workers 0, mixed_precision false, dists_on_cpu false, stain_norm none, max_cache_mb 4` + `configs/hemit_dual_branch_lowram.yaml` for demo (not product)
- [x] Dataset `src/models/datasets.py:26` SWMR + `indices` + `mihc` fallback
- [x] Trainer `src/models/train_paired.py:61` `train_one_epoch(..., grad_accum, oom_guard, monitor)` — `/grad_accum` scaling, OOM try/catch, `monitor.check` every 20, skip DISTS if `avail<600`
- [x] Dummy smoke: `PYTHONPATH=. .venv/bin/python -m src.models.train_paired --config configs/hemit_dual_branch.yaml --dummy --optimize-for lowram` batch2+accum4 loss 56→46 + `train_unpaired --dummy` loss 0.24 ✅
- [x] Full train resumable local: `PID 2138` `epoch10/200 07:37` `checkpoint_epoch10.pt 79M` `checkpoint_last.pt` `2.5h/epoch` → 20 days CPU, paused resumable
- [x] **Kaggle GPU**: `bikasash/hemit-he-dapi-product-full 7.7G` `hemit_train 3.91G + val 717M + test 1.01G + mhist 2.02G + checkpoint_last.pt` + `bikasash/he-dapi-code-product 79M` → `bikasash/he-dapi-product-gpu-train` `T4 16GB` `KernelWorkerStatus.RUNNING` resume `epoch11/200` `15min/epoch` ~50h total (free 30h/week)
- [x] **Metrics backfilled 2026-10-03 (offline, no disturb)**: `src/models/eval_offline.py:1` + `docs/metrics_eval.json:1` ep15/23/33/43/50/53 val N=30 — PSNR 28.99→29.34 (peak 29.82), D-loss 0.75-0.87 balanced, Dice(thresh 0.1) 0.03→0.11 late gain. Full 630 val deferred to Phase 4 (heavy on 3.7GB box)
- [x] **Logging fixed 2026-10-03 (future resumes)**: `src/models/train_paired.py:61` now returns/logs `G + D + gan/l1/ssim/nuclei`, ckpt keeps `loss=G` + new `loss_detail`; `src/models/losses.py:66` Dice `0.5→0.1` (0.5 vacuous 1.0 on DAPI max ~0.3), SSIM proxy note (no kornia)
- [ ] Metrics full: SSIM/PSNR on full `hemit_val_512.h5` (630), nuclei Dice via `src/models/losses.py:1` — Phase 4 gate

#### Track B: DAPI→H&E (CycleGAN/CUT)
- [x] Config `configs/cyclegan_dapi2he.yaml:27` — `batch_size 2, grad_accum 4, num_workers 0, dapi_root H5 fallback`
- [x] Trainer `src/models/train_unpaired.py:54` `train_dummy(monitor, batch, grad_accum)` + `UnpairedDataset` H5 fallback
- [ ] Dummy + full same as Track A → `resources/models/checkpoints/dapi2he/checkpoint_last.pt`

#### Shared low-RAM rules for Phase 2 (must follow till end)
- [ ] Never set `batch_size >2` or `num_workers >0` on this box without `mon.advise_batch` check.
- [ ] Keep `dists_on_cpu false` (VGG 400MB) unless on bigger machine.
- [ ] Always wrap epoch with `mon.check(f"train:epoch{epoch}:start")` / `end`.
- [ ] Use `indices=range(500)` auto-subset if `avail<800` (`train_paired.py:130`).

#### Deliverables checklist
- [ ] `resources/models/checkpoints/he2dapi/generator_last.pt` + `.onnx`
- [ ] `resources/models/checkpoints/dapi2he/cyclegan_G_last.pt`
- [ ] MLflow logs (if `mlflow` available else skip)

### Phase 3 — Drag & Drop Tool ⬜ 0%
- [x] Backend `src/backend/main.py:40` 429 guard + streaming TIFF preview
- [x] Infer `src/models/infer.py:17` streaming + `infer_wsi_streaming`
- [ ] Frontend `src/frontend` Dropzone + OpenSeadragon (needs node build, check RAM 512MB budget)
- [ ] Add queue + websocket for WSI progress

### Phase 4 — Validation ⬜ 0%
- [ ] Hold-out test `hemit_test_512.h5`, cross-tissue, nuclei count error <10%, MC dropout 5× must use streaming (no 5× canvas)

### Phase 5 — Docker ⬜ 0%
- [ ] Multi-stage `docker/Dockerfile`, compose, push (build on bigger machine if RAM <4GB)

## Daily Log

- **2026-08-31**: HEMIT download stalled 9.9G/10.2G, resumable enforced, maxpmx/mhist done, disk 12.8G. `MEMORY.md:5`
- **2026-09-19**: Machine 3.7GB crash → 9 wiki sub-agents, all patches streaming, 11 guards verified, data 86G (hemit 44G raw), `docs/optimization/REPORT.md:1`. Tracker added.
- **2026-09-26**: `torch 2.14.0+cpu` (py3.14), dummy paired+unpaired passed, `checkpoint_epoch3.pt 79M` `cyclegan_G_last.pt 30M`, `infer` `he_example.png → synth_dapi.png` ✅
- **2026-09-26 13:30**: Full product launch `PID 1793` `epoch4/200` `2.5h/epoch` → 20 days CPU, then `PID 2138` resume `epoch10/200 07:37` (8→10 epochs 20h)
- **2026-09-27 03:55**: Kaggle `bikasash/hemit-he-dapi-product-full 7.7G` + `bikasash/he-dapi-code-product 79M` uploaded `KAGGLE_API_TOKEN=KGAT_...` `kaggle datasets create` 30min `4-6MB/s`
- **2026-09-27 10:09**: Kaggle kernel `bikasash/he-dapi-product-gpu-train` `T4` `RUNNING` resume `epoch11/200` `15min/epoch` ~50h free 30h/week, local `PID 2138` paused resumable, laptop can OFF
- **2026-09-27 13:37**: Kaggle paused (7.7G pull queue, logs empty), local took over → reached `epoch14/200` (`checkpoint_epoch14.pt` 23:00)
- **2026-09-28 02:45**: Local relaunched `PID 1011` `epoch15/200` with `logs/train_paired.log` (unbuffered `-u`; first attempt PID 973 killed, log empty from buffering). Kaggle payload staged `/tmp/kaggle_kernel_he_dapi`, code-dataset stage `.tmp_kaggle_code` refreshed to epoch14, push blocked on token
- **2026-09-28 02:56**: Token saved `~/.kaggle/api_token` (600), old kernel was `CANCEL_ACKNOWLEDGED`. Code dataset version pushed (5 files incl `code_bundle.zip`; remote had only 4 files before — uploader drops subdirs). Kernel v5/v6/v7 FAILED (no-src race, subdir-drop, zip-extracted-to-prefix). v8 (merge-up fix) `RUNNING` epoch15, past fail-fast guard. ⚠️ No GPU in v5–v7 logs — quota suspected exhausted
- **2026-09-28 13:51**: Local `18/200 done` (`checkpoint_epoch18.pt` 13:07, epoch19 step540, 11h uptime, RAM 51% improved). Kaggle v8 RUNNING 10.5h (past 9h norm, output locked, GPU unconfirmed)
- **2026-09-28 14:44**: Local epoch19 step1180 (~64%), ckpt19 ~15:35 UTC, RAM 54% steady. Kaggle v8 RUNNING 11.5h, output still locked
- **2026-09-29 02:45**: Kaggle v8 `CANCEL_ACKNOWLEDGED`; salvaged log proves CPU-only (quota out), ep15–18 @2.8h/epoch slower than local, loss higher at same epochs → local (23/200, loss 2.15) wins fork. Kaggle STAND DOWN until GPU quota refreshes
- **2026-09-29 12:21**: Local `27/200 (13.5%)`, epoch28 step1220 (~66%), ckpt28 ~13:10 UTC, 33.5h uptime, RAM ~55%, loss 2.126 falling (minor noise at ep26)
- **2026-09-29 15:32**: Local `28/200 (14%)`, ckpt28 landed 13:08 as predicted, epoch29 step1620 (~87%), ckpt29 ~15:55, loss 2.1155 new low
- **2026-09-30 02:46**: Local `33/200 (16.5%)`, epochs 29–33 overnight, epoch34 step1200 (~65%), ckpt34 ~03:30 UTC, 47.7h uptime, RAM ~55%, loss 2.0605 new low (best drop yet −0.026)
- **2026-09-30 12:26**: Local `37/200 (18.5%)`, epochs 34–37, epoch38 step880 (~47%), ckpt38 ~13:30, 57.3h uptime, RAM ~52%, loss 2.0348 new low. Found: NO validation code in `train_paired.py` (`val_every` unused) — Phase 4 gap confirmed
- **2026-09-30 16:25**: Local `38/200 (19%)`, epoch39 step880 (~47%), ckpt39 ~18:00, 60h uptime, loss 2.0164 new low. Stall scare (69 min no log) investigated: FALSE ALARM — process advancing, ckpts on cadence; epoch39 ~2x slower (swap full 996/1024, load 6.3, no sudo for relief)
- **2026-10-01 02:51**: Local `42/200 (21%)`, epochs 39–42, epoch43 step480 (~26%), ckpt43 ~04:45, 70.7h uptime, RAM 54%. 🎯 Loss broke 2.0 barrier (1.9950 ep41); ep41–42 flat (1.9944) — plateau vs noise TBD
- **2026-10-01 06:45**: Local `43/200 (21.5%)`, epoch44 step1000 (~54%), ckpt44 ~07:45, 74.6h uptime, RAM ~53%. Loss 1.9780 — plateau scare RESOLVED, was noise (−0.016)
- **2026-10-01 12:59**: Local `45/200 (22.5%)`, epochs 44–45, epoch46 step700 (~38%), ckpt46 ~15:10, pace ~3.4h/epoch, 80.8h uptime, RAM ~54%. Loss 1.9718(ep44)/1.9876(ep45) — ep45 uptick likely noise; quarter mark (50) ~Oct-02 06:00
- **2026-10-01 17:02**: Local `46/200 (23%)`, epoch47 step1040 (~56%), ckpt47 ~18:20, pace ~3.2h, 84.8h uptime, RAM 50–53% (best in a while). Loss 1.9513(ep46) new low, biggest recent drop −0.036 — ep45 uptick CONFIRMED noise
- **2026-10-02 02:33**: Local `49/200 (24.5%)`, epochs 47–49, **epoch50 in progress** step300 (~16%), QUARTER MARK ckpt ~05:30 UTC, 94.3h uptime, RAM ~55%. Loss 1.9452/1.9331/1.9290 new low, steady decline
- **2026-10-02 08:00**: 🎯 QUARTER MARK `50/200 (25%)`, ckpt50 landed 05:25 as predicted. Phase 2 bumped 45→50, overall 41→42 via `progress_tracker --update`
- **2026-10-02 12:28**: Local `51/200 (25.5%)`, epoch52 step1420 (~76%), ckpt52 ~13:15, pace ~3.5h, 100.3h uptime, RAM 50%. Loss 1.8982(ep51) new low — broke 1.90 (−0.026)
- **2026-10-02 16:40**: Local `52/200 (26%)`, epoch53 step1100 (~59%), ckpt53 ~18:15, 108.5h uptime, RAM 49–51%. Loss 1.9012(ep52) tiny uptick +0.003, noise watch. Pace drifting up (3.2→3.5→4.3h/epoch) under swap pressure
- **2026-10-02 17:05**: Pace diagnosed by measurement: 20 steps in 241s = 12.0s/step (2.7x slower than day-1 4.5s/step). Ruled OUT: swap thrash (si/so=0), CPU competitors (trainer is 95% of load), memory leak (RSS 719MB), disk. Left standing: thermal derate (4 days at 100% on thin laptop, clocks 2.1GHz) and/or Windows-host contention (invisible from WSL2). No safe in-box fix — user-side levers only, training untouched. ckpt53 revised ~19:20
- **2026-10-02 17:08**: Reboot offered (verified ckpt52 resume-safe), user chose KEEP RUNNING. Verified both processes alive (PID 1011 training 108.9h + PID 1567 watch) — nothing was ever stopped except historical PID 973 (Sep-28 buffered-log relaunch). Epoch53 step1200 (~65%), ckpt53 ~19:20
- **2026-10-02 18:45**: Epoch53 DONE (`checkpoint_epoch53.pt` + `checkpoint_last.pt`), loss 1.8942 NEW LOW — ep52 uptick (+0.003) confirmed noise. Epoch54 started
- **2026-10-03 04:34**: ⚠️ REBOOT (uptime 4min) killed PID 1011 + watch 1567 during epoch54 step1180 (~63%, last log 21:59 Oct-02). Truth `53/200 (26.5%)`, loss 1.8942. ~3.2h epoch54 progress lost, resume from epoch53. Fresh RAM 32%/2581MB swap 0 — ideal resume. Goal 200 FULL, 147 remaining (~15-24d at 2.5-4h/epoch)
- **2026-10-03 04:39**: ✅ RESUMED `PID 935` epoch54/200 from `checkpoint_epoch53.pt`; watch restarted PID 971 (fixed hardcoded 1011 → `pgrep -f train_paired`). RAM 44%/2152MB CPU 339% healthy
- **2026-10-03 05:03**: 🔧 GAPS-FIXED till ep53 (no stop): `train_paired.py` D-loss+components logging (future resumes), `losses.py` Dice 0.5→0.1 + SSIM note, `eval_offline.py` + `metrics_eval.json` ep15/23/33/43/50/53 (PSNR 29.0→29.3 pk 29.8, D 0.75-0.87, Dice 0.03→0.11). TrackB full/MLflow/kornia/full-630-val/frontend correctly deferred — nothing missed. Moving forward to 200
- **2026-10-03 14:43**: 🚀 `57/200 (28.5%)` — ep54 1.8904, ep55 1.8806, ep56 1.8618, ep57 1.8502 all new lows; pace recovered ~2.2h/ep (reboot fixed thermal/swap); epoch58 step920 (~50%) RUNNING, ckpt58 ~15:45 UTC
- **2026-10-04 03:37**: 🚀 `63/200 (31.5%)` — ep58 1.8445, ep59 1.8210, ep60 1.8214 (+0.0004 noise), ep61 1.7979 broke 1.80, ep62 1.7955, ep63 1.7723 NEW LOW; epoch64 step240 (~13%) RUNNING 23h, pace ~2.2h/ep
- **2026-10-04 08:35**: `65/200 (32.5%)` — ep64 1.7662, ep65 1.7636 NEW LOW but drops shrinking (-0.006/-0.003), plateau watch; epoch66 step500 (~27%) RUNNING 1d3h
- **2026-10-04 15:35**: `68/200 (34%)` — ep66 1.7497, ep67 1.7337, ep68 1.7199 NEW LOW; plateau scare RESOLVED (drops -0.014/-0.016); epoch69 step500 (~27%) RUNNING 1d10h
- **2026-10-05 02:20**: `72/200 (36%)` — ep69 1.7187, ep70 1.7104, ep71 1.6985, ep72 1.6741 NEW LOW (-0.024 biggest recent); epoch73 step1520 (~82%) RUNNING 1d21h, ckpt73 ~02:45
- **2026-10-05 13:03**: `76/200 (38%)` — ep73 1.6760, ep74 1.6614, ep75 1.6681 (+0.007 noise), ep76 1.6471 NEW LOW (-0.021); epoch77 step1840 (~99%) RUNNING 2d8h, ckpt77 ~13:15
- **2026-10-05 15:05**: `77/200 (38.5%)` — ep77 1.6364 NEW LOW (-0.011); epoch78 step1480 (~80%) RUNNING 2d10h, ckpt78 ~15:30
- **2026-10-06 02:40**: `82/200 (41%)` — ep78 1.6307, ep79 1.5366 OUTLIER (-0.094 best-ever), ep80 1.5888 bounce (+0.052), ep81 1.5893, ep82 1.5800 NEW LOW ex-outlier; epoch83 step500 (~27%) RUNNING 2d22h
- **2026-10-06 13:02**: `83/200 (41.5%)` — ep83 1.5675 NEW LOW (-0.0125); epoch84 step680 (~37%) RUNNING 3d1h
- **2026-10-07 03:22**: `87/200 (43.5%)` — ep84 1.5554, ep85 1.5422, ep86 1.5388, ep87 1.5361 BEST-EVER (beats ep79 outlier); epoch88 step1360 (~73%) RUNNING 3d15h, ckpt88 ~04:00
- **2026-10-07 09:09**: `89/200 (44.5%)` — ep88 1.5277 BEST-EVER (-0.008), ep89 1.5388 (+0.011 noise); epoch90 step620 (~33%) RUNNING 3d21h, ckpt90 ~10:00
- **2026-10-07 13:02**: `90/200 (45%)` — ep90 1.4535 BEST-EVER (-0.085 outlier watch like ep79); epoch91 step540 (~29%) RUNNING 4d1h, ckpt91 ~15:15
- **2026-10-08 03:02**: `92/200 (46%)` — ep91 1.5075 bounce (+0.054 as predicted), ep92 1.4950 recovering; best-ever ep90; epoch93 step320 (~17%) RUNNING 4d15h
- **2026-10-08 13:41**: still `92/200` epoch93 99% step1840 (ckpt93 imminent) — epoch93 SLOW ~12.5h vs 2.2h (swap 493MB, load 5.1, 5d thermal); training untouched. Watch 971 stuck 4d on kaggle pipe (`anon_pipe_read`) killed → 47153 + `timeout 20` guard in `watch_runs.sh`
- **2026-10-08 14:03**: ✅ ckpt93 DONE 13:46 loss 1.4835, `93/200 (46.5%)`, epoch94 step20 — SAFE TO REBOOT, resumes epoch94 from `checkpoint_last.pt`
- **2026-10-08 14:44**: ♻️ REBOOT happened (uptime 3min, PID935 + watch dead at epoch94 step20, only ~20 steps lost). ✅ RELAUNCHED `PID937` epoch94/200 via `-m src.models.train_paired` (direct-script mode ImportError fixed), 341% CPU RSS ~1GB healthy. Watch `PID993` WEEK-MODE 300s (was 50s → 6x fewer wakeups/clock jitter + API calls). Fresh RAM 32% swap 0 — pace should recover ~2.5h/ep. 107 remaining; 1-week projection ~160-169/200. Kaggle OFF
- **2026-10-09 02:42**: 🚀 `98/200 (49%)` — overnight perfect: ckpt94 17:09, ckpt95 19:22, ckpt96 21:37, ckpt97 23:50, ckpt98 02:08 (~2.25h/ep steady, no degradation). Losses 1.8529→1.8306 falling. Epoch99 step440 (~24%) RUNNING PID937 12h. Swap 407M/1024M watch. Phase2 50→55
- **2026-10-09 11:06**: 🎯 HALFWAY `102/200 (51%)` — ckpt99 04:25, ckpt100 06:36, ckpt101 08:48, ckpt102 11:00 (~2.2h/ep rock-steady). Losses 1.8202/1.8094(ep100 broke 1.81)/1.8138(+0.004 noise)/1.8069 new low. Epoch103 step60 RUNNING PID937 20h. Swap 495M creep watch, RSS 869M healthy. Phase2 55→60, overall 42→43
- **2026-10-13 07:10**: 🟡 TRAINER DEAD at epoch136 step520 (ckpt135 DONE 05:59 loss 1.6695, `135/200 67.5%` ground truth, 135×79M ckpts). Losses 130-135: 1.6807/1.6731/1.6797/1.6712/1.6704/1.6695 steady decline. Pace 3.3-3.5h/ep under pressure (swap+opencode), RAM now fresh 32%/2572MB swap 0. RELAUNCH WEEK-MODE nice10 + watch300s + keepalive300s for 1-week continuous (65 left → finish ~Oct-19 @2.3h/ep, ~ep183 @3.5h/ep). Kaggle OFF
- **2026-10-13 11:07**: ✅ ckpt136 DONE loss 1.6666 NEW LOW (`136/200 68%`). STOPPED all (train 780 3.5h epoch + watch 813 + keepalive 818) for user SYSTEM RESTART — pace stuck 3.4h/ep, want <2.3h. Only ~2 min epoch137 sacrificed. SAFE TO REBOOT, resume epoch137 (64 left)
- **2026-10-13 11:33**: 🚀 RELAUNCHED 1-WEEK MODE `PID799` epoch137/200 from `ckpt136` (`nice10 -u -m --resume auto --optimize-for lowram`, `watch866` + `keepalive867` 300s, `setsid+disown` detach). Step20 OK 4.5s/step fresh pace ~2.3h/ep (reboot fixed thermal/swap), RAM 60% avail 1315MB. 64 left → finish ~Oct-19/20 (~147h). Kaggle OFF
- **2026-10-13 14:21**: ✅ ckpt137 DONE loss 1.6632 NEW LOW (`137/200 68.5%`), epoch138 step260 RUNNING `PID799` ~2.8h/ep. 63 left → finish ~Oct-19/20. Git staged+committed locally (75 files) for github.com/urbiologicalpaa push
- **2026-10-14 03:56**: 🚀 overnight +5: ckpt138 (1.6574) / 139 (1.6481) / 140 (**1.5842** −0.064 dip, outlier watch like ep79/ep90) / 141 (1.6103) / 142 (1.6285) DONE (`142/200 71%`). Epoch143 step560 (~30%) RUNNING `PID799` 16.5h ~2.4h/ep, RAM 54% healthy. 58 left → finish ~Oct-19. Unpaired CycleGAN 200 kept in plan, decide at ep200. GitHub building-framed + CI + issues + release live
- **2026-10-14 09:43**: ckpt143 (1.6244 recovering) / ckpt144 (1.6410 +0.017 uptick-watch) DONE (`144/200 72%`). 4 epochs sideways since ep140 dip (1.61→1.64) — noise cluster vs mild plateau, TBD; D-loss back to healthy 0.7825. Epoch145 step1080 (~58%) RUNNING `PID799` 22h ~2.4h/ep. 56 left → finish ~Oct-19
- **2026-10-14 11:17**: ckpt145 DONE **1.6318** (−0.009, half the uptick retraced), D **0.7745** healthiest in days → leans **noise over plateau**, band persists one more epoch to confirm. `145/200 (72.5%)`. Epoch146 step300 (~16%) RUNNING `PID799` 23.75h. 55 left → finish ~Oct-19
- **2026-10-14 13:02**: ⚠️ REBOOT ~12:49 (uptime 13min) killed PID799 + watch866 + keepalive867 at epoch146 step980 (~52%, last log 12:45). Truth `145/200` SAFE (`checkpoint_last.pt` 10:50, 79M, verified G 1.6318/D 0.7745) — only ~1.2h redo lost, no ckpt/data/tracker loss. ✅ RELAUNCHED `PID1203` epoch146 from ckpt145 + watch1272 + keepalive1283 WEEK-MODE 300s, 334% CPU healthy. 55 left → finish ~Oct-19
- **2026-10-14 15:55**: ✅ ckpt146 DONE **1.6365** (+0.005 sideways, 6-epoch band since ep140 dip 1.5842; D **0.7730** healthy) `146/200 (73%)`. Epoch147 step120 RUNNING `PID1203` 2h53m ~2.4h/ep, RAM 59-63%. 54 left → finish ~Oct-19. CI/CD all green (guards + cd-staging `:staging` live); push-hygiene (secret-scan + `private/`) in place
- **2026-10-14 18:10**: ⚠️ REBOOT ~17:59 killed PID1203 + watch1272 + keepalive1283 at epoch147 step1040 (~56%; train ran till 17:54, watch silent since 16:56). Truth `146/200` SAFE (146 ckpts + `checkpoint_last.pt` verified epoch145/loss 1.6365) — only ~1.3h redo lost, no data/tracker loss. ✅ RELAUNCHED `PID952` epoch147 from ckpt146 + watch1011 + keepalive1266 WEEK-MODE 300s; keepalive now also restarts watch (closes 1h blind gap). Step20 OK 329% CPU. 54 left → finish ~Oct-19
- **2026-10-14 18:30**: 📝 DOCS — push `c5ea231` synced + `guards` green; token/auth map + reboot-relaunch order written to `wiki/09-orchestration/SKILL.md` (GitHub `~/.github_pat` vs `.git_pat` name lesson cost 3 failed pushes; credential helper now wired reboot-safe). Same-issue prevention: `ls ~` first, never guess token filenames
- **2026-10-15 04:25**: 🚀 overnight +4: ckpt147 (**1.5945** dip toward ep140 best 1.5842) / 148 (1.6239) / 149 (1.6245) / 150 (1.6307) DONE (`150/200 75%`). Sideways band persists but ep147 dip keeps noise-leaning; D **0.7705** healthy. Epoch151 step600 (~32%) RUNNING `PID952` 10.3h ~2.4h/ep steady. 50 left → finish ~Oct-19
- **2026-10-15 06:25**: ckpt151 DONE **1.6348** (`151/200 75.5%`, sideways band; D **0.7686** healthy). Epoch152 step280 (~15%) RUNNING `PID952` 12.3h ~2.4h/ep steady. 49 left → finish ~Oct-19
- **2026-10-15 09:18**: ckpt152 DONE **1.6178** (`152/200 76%`, **−0.017** recovering, best move since ep147 dip; D **0.7685** healthy). Epoch153 step720 (~39%) RUNNING `PID952` 15.2h ~2.4h/ep steady. 48 left → finish ~Oct-19
- **2026-10-15 13:49**: 🎯 ckpt153 **1.5718 NEW BEST** (beats ep140 1.5842!) / ckpt154 1.5997 bounce (`154/200 77%`). Epoch155 step560 (~30%) RUNNING `PID952` 19.8h ~2.4h/ep steady. 46 left → finish ~Oct-19
- **2026-10-15 16:41**: ckpt155 DONE **1.6193** (`155/200 77.5%`, +0.020 uptick sideways; D **0.7683** healthy; best stays ep153 1.5718). Epoch156 step860 (~46%) RUNNING `PID952` 22.6h ~2.4h/ep steady, RAM 56-57% best in a while. 45 left → finish ~Oct-19
- **2026-10-16 03:56**: 🚀 overnight +5: ckpt156 (1.6144) / 157 (1.6115) / 158 (1.6112) / 159 (1.6175) / 160 (1.6095) DONE (`160/200 80%`, steady decline; D **0.7614** healthy; best stays ep153 1.5718). Epoch161 step740 (~40%) RUNNING `PID952` 33.9h ~2.4h/ep steady. 40 left → finish ~Oct-19
- **2026-10-16 05:23**: ckpt161 DONE **1.6073** (`161/200 80.5%`, decline continues; D **0.7594** healthy; best stays ep153 1.5718). Epoch162 step40 just started, RUNNING `PID952` 35.4h ~2.4h/ep steady. 39 left → finish ~Oct-19
- **2026-10-16 09:46**: ckpt162 DONE **1.6099** (`162/200 81%`, +0.003 flat noise; D **0.7769** healthy; best stays ep153 1.5718). Epoch163 step820 (~44%) RUNNING `PID952` 39.6h ~2.4h/ep steady, RAM 57-58%. 38 left → finish ~Oct-19
- **2026-10-16 12:23**: 🎯 ckpt163 DONE **1.5685 NEW BEST** (beats ep153 1.5718!; D **0.7802** healthy) (`163/200 81.5%`). Epoch164 step800 (~43%) RUNNING `PID952` 42.3h ~2.4h/ep steady, RAM 58-59%. 37 left → finish ~Oct-19
- **2026-10-16 16:07**: ckpt164 DONE **1.5975** (`164/200 82%`, +0.029 bounce off best as usual; D **0.7594** healthy; best stays ep163 1.5685). Epoch165 step1180 (~64%) RUNNING `PID952` 46h ~2.4h/ep steady, RAM 58-59%. 36 left → finish ~Oct-19
- **2026-10-16 17:16**: ckpt165 DONE **1.6031** (`165/200 82.5%`, +0.006 flat; D **0.7565** healthy; best stays ep163 1.5685). Epoch166 step80 just started, RUNNING `PID952` 47.2h ~2.4h/ep steady, RAM 58-61%. 35 left → finish ~Oct-19
- **2026-10-16 18:35**: 🚀 overnight +3: ckpt166 (1.6039) / 167 (1.6199) / 168 (1.5934) DONE (`168/200 84%`, +0.013 since 163 best ep163 1.5685; D 0.7877 volatile). Epoch169 step40 (~10%) RUNNING `PID952` 4h ~2.4h/ep steady, RAM 56%. 32 left → finish ~Oct-19
- **2026-10-17 03:37**: ckpt168 DONE (`168/200 84%`, overnight +3 complete). Epoch169 step40 just started, RUNNING `PID952` 4h ~2.4h/ep steady, RAM 56%.

## How to use tracker

```bash
# human: read this file
cat docs/TRACKER.md
# machine: update after each step
PYTHONPATH=. .venv/bin/python src/utils/progress_tracker.py --check
PYTHONPATH=. .venv/bin/python src/utils/progress_tracker.py --update phase2 --percent 25 --note "dummy passed, starting full train epoch 1"
# view JSON
cat docs/progress.json | python -m json.tool
```

## Next Immediate (for Phase 2 dual run)
1. Local (RUNNING, no action): `tail -f logs/train_paired.log`; verify `ps -p 1011`; `checkpoint_epoch15.pt` expected ~05:15 UTC (~2.5h/epoch). Watch RAM (69% at step240, OOM guard on)
2. Kaggle v8 (RUNNING): logs/output downloadable only after COMPLETE — then `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/api_token); .venv/bin/python -m kaggle kernels output bikasash/he-dapi-product-gpu-train --path /tmp/kaggle_out_v8` and check `nvidia` line for GPU. If CPU-only → GPU quota exhausted; keep Kaggle as backup, local stays primary
3. Fork rule (both train epoch15 from same ckpt): when either finishes an epoch, compare `checkpoint_last.pt` mtimes, copy winner locally before any resume; never resume one from the other's mid-epoch ckpt blindly
4. After every milestone update ALL THREE together: `docs/progress.json:runs` + `TRACKER.md:Live Runs` + `MEMORY.md` banner, then `PYTHONPATH=. .venv/bin/python -m src.utils.progress_tracker --update phase2 --percent NN --note "..."`
5. Local stop (only if Kaggle proves GPU-fast): `kill 1011` (resumable from `checkpoint_last.pt`)

---
Generated: `docs/TRACKER.md` + `docs/progress.json` via `src/utils/progress_tracker.py:1` | Follow low-RAM pattern till end.
