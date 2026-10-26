# LIVE STATE — 2026-10-23 12:45 UTC

> **Canonical strategy/decisions: [`docs/PLAN.md`](PLAN.md) v2.0.**
> **Read this file first after any WSL crash, reboot, or "what is running?" question.**
> It is the source of truth for the active run. Update it **before** launching and
> **after** every exit/crash. Do not rely on memory or assumptions.
>
> Machine: WSL2, 3.7GB RAM, 1GB swap, no GPU, 4c/8t i5-10210U.
> Repo: `/home/bikash/he-dapi-translator`.

---

## 0. RESOLVED — HEMIT channel audit (2026-10-23)

**ADR-014 is accepted and the corrected run is complete.** The raw HEMIT RGB
TIFF has **`ch2=DAPI` (blue), `ch0=panCK` (red), `ch1=CD3` (green)**. The legacy
512 model is the real DAPI model; the 1024 `ch0` forward/reverse models and
R-006 are panCK. See [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md). The ch2 H5 rebuild
is verified; **do not launch any DAPI run from a `ch0` cache.** Corrected ch2
results are in §0.3 and [`RESULTS.md`](RESULTS.md).

### 0.1 Data correction (completed 2026-10-23 04:35 UTC)

- **Job:** rebuild 256px H5 with `dapi = mihc[...,2]` (DAPI blue), sequential
  val → test → train. Old wrong files renamed `*_256_ch0_invalid.h5`.
- **Command:**
  ```bash
  for s in val test train; do
    PYTHONPATH=. .venv/bin/python -m scripts.data.crop_hemit_h5 \
      --src data/processed/hemit_${s}_1024.h5 \
      --out data/processed/hemit_${s}_256.h5 \
      --size 256 --compression lzf
  done
  ```
- **PID:** `13311`; **log:** `logs/rebuild_256_ch2.log`.
- **Expected:** `hemit_{val,test,train}_256.h5` with attrs
  `dapi_channel=2`, `channel_order=panCK,CD3,DAPI`; sizes ~104MB / 148MB /
  596MB. Streaming, <2GB RAM.
- **Result:** DONE 04:35:46 UTC, exit 0. Verified val/test/train `dapi` is
  byte-identical to the center crop of `mihc[...,2]` (indices 0/17/629 on val),
  attrs `dapi_channel=2`; sizes 102MB/152MB/596MB. Old ch0 derivatives are
  preserved as `*_256_ch0_invalid.h5`.
- **Loader:** `PairedHEMITDataset` and `UnpairedDataset` now prefer
  `mihc[...,2]`, so the wrong 1024 `dapi` cache is ignored. Full suite:
  51 passed, 1 skipped.
- **Legacy re-verification:** 50-crop val on the legacy 512 checkpoint
  (SHA-256 `6789d7d2…4042aa`) gives FID 146.27, LPIPS 0.2468, PSNR 29.52,
  SSIM 0.7662, Pearson 0.5323, L1 0.0221; saved to
  `docs/metrics_dapi_512_legacy_val50.json`. This confirms it predicts DAPI.
- **Artifact proof (2026-10-23 10:53 UTC):** `hemit_{train,val}_512.h5["dapi"]`
  is array-equal to `mihc[...,2]`; public HF `he2dapi_unet_epoch169.pt`
  (SHA-256 `c87751…e2afdc`) has 46/46 generator tensors exactly equal to local
  `checkpoint_epoch169.pt["G"]`.
- **Next:** R-007 completed but is invalid for selection; paired U-Net remains
  default. Stage 2.5 (forward/reverse quality) is next after the quota reset.

---

## 0.2 COMPLETED corrected run — paired ch2 1024, both directions (2026-10-23 04:54–08:55 UTC)

- **Kernel:** `bikasash/he-dapi-paired-ch2-1024-both` v1
  (`/tmp/kaggle_kernel_paired_ch2_1024/`), GPU T4, private.
- **Push command:**
  ```bash
  KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)" \
    .venv/bin/kaggle kernels push -p /tmp/kaggle_kernel_paired_ch2_1024
  ```
- **Data:** `bikasash/hemit-dapi-1024-ch0-macenko` (historical name; `mihc`
  intact, loader reads `mihc[...,2]`) + `bikasash/he-dapi-code-dapi-1024`
  v2026-10-23 04:53 (ADR-014 loader/both runner).
- **Plan:** `DAPI_DIRECTION=both`, `DAPI_STAGE=all`,
  `DAPI_PRETRAIN_EPOCHS=14`, `DAPI_FINETUNE_EPOCHS=2`,
  `DAPI_FINETUNE_STEPS=500`. Forward then reverse.
- **Outputs (kernel):** `/kaggle/working/checkpoints/dapi_{pretrain,finetune}/`,
  `/kaggle/working/checkpoints/dapi2he_{pretrain,finetune}/`.
- **Expected:** ~4–5 h; quota was **6.18 h remaining** at 04:45 UTC.
- **Stop rule:** if either direction exceeds ~2.4 h before fine-tune, cancel and
  keep the partial checkpoint; do not exhaust quota without a result.
- **Resume:** re-push the same kernel; `--resume auto` continues from the
  highest checkpoint if the output is reattached.
- **Local monitor:** PID `15902`, log `logs/kaggle_ch2_monitor.log`. The first
  output pull filled the 1.9GB `/tmp` tmpfs; checkpoints were re-pulled to
  `/home/bikash/kaggle_ch2_output`, log included.
- **Status:** `KernelWorkerStatus.COMPLETE` at 08:55 UTC; quota used ~27.3h,
  2.67h left. Both directions exit 0.
- **Verified from log:** CUDA T4, H5-only (`hemit_train_1024.h5` 11.52GB),
  `[DIRECTION] he2dapi` / `dapi2he`, `device cuda`; loader target confirmed
  against `mihc[...,2]` (MAE 0.0 vs ch2, corr 1.0 on val crop).
- **Caveats:** pretrain used 512 crops from the 1024 H5 (`patch 512 != file 1024`
  warning); DISTS was disabled (`use_dists` requires `dists_on_cpu: true`, a
  code/logic bug); no val-based checkpoint selection; only 500 fine-tune steps.
- **Canonical checkpoints** (copied into repo, gitignored):
  - `resources/models/checkpoints/he2dapi_ch2_1024/checkpoint_last.pt`
    SHA-256 `78b15ef57ef62555bcb645acca501340d6dd56348f4b17681a3b2e273952ebd5`
  - `resources/models/checkpoints/dapi2he_ch2_1024/checkpoint_last.pt`
    SHA-256 `5f930960c6fd5a0adb3beb40e1a4d1d1d239e24908b7d38c2012af8cba870f98`
- **Status docs:** `docs/metrics_ch2_1024_forward_val50.json`,
  `docs/metrics_ch2_1024_reverse_val50.json`,
  `docs/metrics_ch2_1024_{forward,reverse}_pretrain512_val50.json`.

---

## 0.3 Corrected ch2 1024 results (50 val crops, no full-split claim)

| Run | px | FID ↓ | LPIPS ↓ | PSNR | SSIM | Pearson | L1 ↓ | Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Legacy 512 DAPI (200 ep) | 512 | **146.27** | **0.2468** | **29.52** | 0.7662 | 0.5323 | **0.0221** | — |
| ch2 forward pretrain | 512 | 184.73 | 0.4044 | 27.54 | 0.7431 | 0.2877 | 0.0296 | 0.0381 |
| ch2 forward fine-tune | 1024 | 215.38 | 0.3899 | 27.91 | **0.7733** | **0.5515** | 0.0250 | 0.0828 |
| ch2 reverse pretrain | 512 | 372.22 | 0.5908 | 15.13 | 0.4619 | 0.6448 | 0.1613 | — |
| ch2 reverse fine-tune | 1024 | **282.92** | **0.5317** | **15.42** | **0.4891** | **0.6644** | **0.1505** | — |
| panCK reverse (invalid DAPI) | 512 | 313.18 | 0.6051 | 13.25 | 0.4960 | 0.4634 | 0.2009 | — |

Both corrected directions are **valid but not release quality**. Forward does
not beat the 200-epoch legacy 512 model on FID/LPIPS/PSNR/L1; reverse improves on
the invalid panCK baseline but remains blurry/low-contrast (SSIM 0.489, PSNR
15.42). More 1024 fine-tuning is the next cheap test; do not promote or publish
these checkpoints.

---

## 0.4 INCIDENT — WSL VM termination, 2026-10-23 09:30 UTC

**Verdict: host disk ceiling, not a Python/OOM crash.**

- Boot `-1` (`e4c11b9f…`) ran 2026-10-22 15:17:19 → 2026-10-23 09:30:01 UTC and
  ended abruptly: the last journal line is the 09:30:01 cron entry, there is no
  shutdown target, and the next boot (`d4a60901…`, 09:37:20 UTC) logged
  `system.journal … corrupted or uncleanly shut down`.
- No Linux OOM line in boot `-1`; no Windows Kernel-Power 41 for this event.
  The WSL VM was terminated host-side.
- **Cause: Windows `C:` is full — 256 MB free at 09:52 UTC (475.5 GB volume).**
  The WSL `ext4.vhdx` is **166.9 GB (155.4 GiB)** and cannot grow. Writes that
  need new blocks fail with ENOSPC even though `df` inside WSL shows 821 GB
  free: that is the sparse vhdx's virtual maximum, not real disk.
- **Evidence chain:**
  - `logs/kaggle_ch2_monitor.log` ends at 08:55 with
    `[Errno 28] No space left on device`.
  - The first pull filled the 1.9 GB `/tmp` tmpfs; the re-pull to `/home` left
    `kaggle_ch2_output/checkpoints/dapi_pretrain/checkpoint_epoch14.pt` at
    **0 bytes** (09:08 UTC).
  - `crash_guard --preflight` at 09:52: `C: free 256 MB < 8192 MB` → BLOCKED.
- **Resolved 2026-10-23 10:37 UTC.** Removed only verified duplicates/caches:
  inside WSL 53.1 GB (`kaggle_dapi_1024` 16G; HEMIT/BCI/transplant/maxpmx/
  MHIST archives ~33G; invalid `*_256_ch0_invalid.h5` 0.85G; `~/.cache/uv`
  2.5G) and Windows safe caches 5.2 GB (pip 2.4G, Ollama updates 1.6G, Temp,
  SoftwareDistribution, browser caches, Downloaded Installations). **No
  checkpoint or extracted dataset was touched.** `C:` free rose **0.25 → 12.07
  GB** (9.7–12 GB while the Windows pagefile resizes). The `ext4.vhdx` still
  reports ~156 GB and was **not compacted**; the gain is from deleted
  files/caches and pagefile movement, so do not assume the vhdx shrank.
  `--preflight` now returns `RESULT: SAFE`.

**Loss audit — nothing scientifically important lost.**

| Artifact | State |
|---|---|
| Kaggle `bikasash/he-dapi-paired-ch2-1024-both` | **COMPLETE, exit 0, 08:55 UTC**; nothing to resume |
| Full Kaggle output | `/home/bikash/kaggle_ch2_output`, 2.2 GB, 33 files |
| Forward `he2dapi_ch2_1024/checkpoint_last.pt` | intact, SHA-256 `78b15ef5…52ebd5` (matches §0.2) |
| Reverse `dapi2he_ch2_1024/checkpoint_last.pt` | intact, SHA-256 `5f930960…ba870f98` (matches §0.2) |
| `dapi_pretrain/checkpoint_epoch14.pt` | **0 bytes / truncated** (optional epoch snapshot; `epoch13`, `checkpoint_last`, `generator_last` intact → no resume impact) |
| Local monitor PID 15902 | died with the VM (expected) |
| Code/docs | intact; working tree still uncommitted |

**Guard installed 2026-10-23 09:57 UTC** — `scripts/crash_guard.sh`:

- `--preflight`: hard gates (C: ≥ 8 GB, WSL root ≥ 4 GB, SwapFree ≥ 300 MB,
  MemAvailable ≥ 800 MB, no 0-byte checkpoints). Exit `2` = do not launch.
- `--sample`: 5-minute heartbeat (`logs/crash_reports/heartbeat.log`).
- `--boot-check`: writes a forensic report after any boot-id change.
- `--report`, `--watch PID`, `--install-cron`.
- `logs/boot_launch.sh`, `logs/keepalive_week.sh`, and
  `scripts/train_unpaired_safe.sh` now call `--preflight` and refuse to launch
  when it is red. Cron records heartbeats every 5 min and a boot report at
  reboot.
- **Hard rule: no training launch and no Kaggle output pull until
  `bash scripts/crash_guard.sh --preflight` exits 0.**

---

## 0.5 R-007 (Stage 2.3) — COMPLETE, INVALID FOR SELECTION

**Status: COMPLETE 2026-10-23 12:13 UTC, exit 0.** Kernel
`bikasash/he-dapi-cyclegan-ch2-256-r-007`, version 1. Training finished all 10
epochs (416–430 s/epoch; epoch 10 G 1.1104, D 0.4318). The result **cannot be
used for the Stage 2.4 decision**:

- The kernel eval command omitted `--metrics fid,lpips`, so
  `eval_cyclegan.json` has `val_FID: null` and `val_LPIPS: null`. The primary
  comparison metrics do not exist.
- The raw eval (`L1 0.3449`, `PSNR 8.45`, `SSIM 0.2215`, `Pearson -0.2984`)
  is also invalid: R-007's generator used `Tanh` output (`[-1,1]`) while
  `eval_offline` compared it directly with `[0,1]` targets. A 5-crop spot check
  with the correct denormalization gives PSNR 12.25 / SSIM 0.3698 / Pearson
  -0.2975, still far below the paired reference (15.45 / 0.4772). Even a valid
  re-eval would not be competitive.
- The run is only 10 epochs on 3,717 crops, so it is also undertrained relative
  to the paired baseline. It is a failed smoke run, not a fair architecture
  verdict.

**Artifacts:** archived invalid JSON `docs/metrics_cyclegan_ch2_256_r007_invalid.json`;
`cyclegan_G_last.pt` (SHA-256 `815023a0…9e4857`) and `generator_last.pt`
(SHA-256 `d5ef563e…39956f`) in
`/home/bikash/kaggle_r007_output/checkpoints/hemit_cyclegan_ch2_256/`.
The full output pull would have downloaded ten 289 MB epoch checkpoints; it was
stopped and a filtered pull fetched only the 91 MB needed artifacts.

**Code fixes applied after the run:** `ResnetGenerator` now ends in `Sigmoid`
so its `[0,1]` output matches the `[0,1]` losses/eval; the Kaggle kernel now
passes `--metrics fid,lpips`; regression tests pass (`43 passed, 3 deselected`).
Legacy `Tanh` CycleGAN checkpoints (R-006/R-007 and the old
`dapi2he/cyclegan_G_last.pt`) are not valid under the fixed code. The Kaggle
code dataset still contains the pre-fix copy; re-upload it before any future
CycleGAN run (none is planned).

**Decision:** R-007 is **inconclusive/invalid**; the paired U-Net remains the
default. Do not spend more GPU quota on CycleGAN. The next GPU spend is Stage
2.5 (longer 1024 paired schedule with val-based selection) after the
`2026-10-24 00:00 UTC` quota reset.

**Correct ch2 reference:** paired reverse `dapi2he_ch2_1024` at **256 px, 50
val crops** = **FID 285.29, LPIPS 0.5129, PSNR 15.45, SSIM 0.4772, L1 0.1548**
(`docs/metrics_ch2_1024_reverse_256_val50.json`). At **1024 px** the same
checkpoint gives FID 282.92 / LPIPS 0.5317 / SSIM 0.4891
(`docs/metrics_ch2_1024_reverse_val50.json`); do not mix resolutions. The old
`342.98 / 0.5904` reference is **ch0 panCK and must not be used**.

---

## 0.6 Stage 2.5 reverse — COMPLETE 2026-10-23 ~15:00 UTC (output pulled 2026-10-24)

**Status: COMPLETE (exit 0), 4/4 epochs.** The quota did not kill it; it ran
~2 h 2 min on a T4 and finished the full 4-epoch schedule. Output was pulled to
`/home/bikash/kaggle_out_stage25_reverse/` (518 MB) at 2026-10-24 03:00 UTC.

- **Result (training-time val, 50 crops):** L1 0.1520, PSNR 15.24,
  `SSIM` 0.2225. **Do not read 0.2225 as SSIM**: `validate()` used
  `SSIMLoss`, which returns 1-SSIM (or a cosine proxy when kornia is absent),
  so the best-checkpoint score was inverted. This is fixed in the 2026-10-24
  code bundle; the real SSIM/FID/LPIPS must come from `eval_offline`.
- **Checkpoints:** `checkpoint_best.pt` and `checkpoint_last.pt` are both
  epoch 3 (4th epoch) and have identical generator weights
  (G hash `a3afeab7…`); `checkpoint_best` SHA-256 `53c8d3e5…`, last
  `98b74b76…`. Epochs 1–4 are preserved locally.
- **Next:** run the Stage 2.5 eval kernel (val 630 + FID/LPIPS on 50 crops;
  test 945 is skipped because the 1024 Kaggle dataset has no test H5).

### 0.6.1 Original launch record (2026-10-23 12:56 UTC)

User approved using the remaining ~1.1 GPU-h because per-epoch checkpoints allow
resume after the quota reset. Datasets were uploaded and the training kernel was
pushed at 12:56 UTC.

- **Kernel:** `bikasash/he-dapi-paired-stage25-reverse` v1
  (https://www.kaggle.com/code/bikasash/he-dapi-paired-stage25-reverse).
- **Push command:**
  ```bash
  KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)" \
    .venv/bin/kaggle kernels push -p /home/bikash/kaggle_stage/stage25_kernel
  ```
- **Datasets:** `bikasash/he-dapi-code-stage25` (code_bundle.zip, 60 KB) and
  `bikasash/he-dapi-dapi2he-ch2-1024-seed` (checkpoint_last.pt, 78.3 MB), both
  private and `ready`; plus the existing
  `bikasash/hemit-dapi-1024-ch0-macenko` H5 payload.
- **Outputs (kernel):** `/kaggle/working/checkpoints/dapi2he_quality/`:
  `checkpoint_epoch*.pt`, `checkpoint_last.pt`, `checkpoint_best.pt`,
  `generator_last.pt`, `generator_best.pt`.
- **Expected stop:** quota exhaustion after ~1–2 epochs; epoch 1 is saved at
  ~33 min, so a quota kill does not lose the run.
- **Resume (after reset):** pull only the output/checkpoints, create a private
  resume dataset (e.g. `bikasash/he-dapi-stage25-resume`), add it to the
  kernel's `dataset_sources`, and re-push the same kernel. The quality stage
  detects a quality checkpoint and continues from the next epoch instead of
  resetting to epoch 1. Run `bash scripts/crash_guard.sh --preflight` before
  pulling.
- **Monitor:** `kaggle kernels status bikasash/he-dapi-paired-stage25-reverse`;
  log-only pull:
  `kaggle kernels output ... -p /home/bikash/kaggle_stage25_logs --file-pattern '.*\.log$'`.

- **Seed:** `resources/models/checkpoints/dapi2he_ch2_1024/checkpoint_last.pt`,
  SHA-256 `5f930960…ba870f98` (epoch 1, direction `dapi2he`). Staged copy:
  `/home/bikash/kaggle_stage/stage25_seed/checkpoint_last.pt` (79 MB).
- **Config:** `configs/hemit_dapi2he_quality_1024.yaml` — 4 epochs, batch 1 ×
  grad-accum 8, lr 3e-5, fp32, GAN 3 / L1 50 / SSIM 10 / colour 20 / gradient
  25; DISTS off (offline VGG risk). Fixed validation: 50 crops from
  `hemit_val_1024.h5` every epoch.
- **Selection:** `score = val_L1 + 0.25 × (1 − val_SSIM)`; saves
  `checkpoint_best.pt` and `generator_best.pt` in addition to
  `checkpoint_last.pt`. Final FID/LPIPS on best and last; test 945 only once for
  the selected checkpoint.
- **Code changes:** `src/models/train_paired.py` gained `validate()`,
  `--val-h5/--val-num/--val-every`, and best-checkpoint saves;
  `src/models/losses.py` now moves VGG to the input device and the DISTS gate
  respects `dists_weight` on CUDA; `scripts/kaggle/eval_dapi_1024.py` now
  evaluates best/last with `--metrics fid,lpips --metric-num 50`.
  Tests: **44 passed, 3 deselected**.
- **Local staging (uploaded):**
  - code: `/home/bikash/kaggle_stage/stage25_code/` (~312 KB; `code_bundle.zip`
    + minimal `scripts/kaggle`)
  - seed: `/home/bikash/kaggle_stage/stage25_seed/` (79 MB)
  - training kernel: `/home/bikash/kaggle_stage/stage25_kernel/`
    (`bikasash/he-dapi-paired-stage25-reverse`)
  - eval kernel: `/home/bikash/kaggle_stage/stage25_eval/`
    (`bikasash/he-dapi-paired-stage25-eval`; uses the training kernel output)
- **Data:** existing `bikasash/hemit-dapi-1024-ch0-macenko` (historical name;
  loader reads `mihc[...,2]`).
- **Budget/stop:** ~14,868 microbatches (~1,859 optimizer steps). The 2026-10-23
  1024 fine-tune measured ~0.53 s/microbatch on the T4, so expect **~2.2 h
  training + ~0.5 h eval**; quota will stop this first attempt after ~1–2
  epochs. Stop at 4 h or on OOM/no-progress; keep partial checkpoints.
- **Eval (after training completes):**
  ```bash
  KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)" \
    .venv/bin/kaggle kernels push -p /home/bikash/kaggle_stage/stage25_eval
  ```
  Run `bash scripts/crash_guard.sh --preflight` before any output pull. Do not
  launch while `SwapFree < 300 MB`.

## 0.7 Stage 2.5b forward — LAUNCHED 2026-10-24 03:10 UTC (long run)

**Why a new forward experiment, not the 4-epoch Stage 2.5 recipe.** The
2026-10-22 forward quality pass (panCK ch0, same 4-epoch colour/gradient recipe)
was a **negative result**: FID 77.14 → 89.32 at 512px, LPIPS 0.1407 → 0.1412,
PSNR 33.88 → 32.49, SSIM 0.728 → 0.664. Repeating that recipe on `ch2` would
spend quota on a known-bad loss mix. The forward direction is also the product's
primary direction, and its best checkpoint is still the **legacy 512 epoch-200**
model (FID 146.27 / SSIM 0.7662), while the corrected 1024 `ch2` model is
undertrained (FID 215.38) after only 2×500 fine-tune steps.

**Hypothesis H1:** a long, paper-aligned 1024 fine-tune seeded from the legacy
512 epoch-200 checkpoint beats both the 1024 `ch2` baseline and the legacy 512
model on val FID/LPIPS, without colour collapse.

- **Kernel:** `bikasash/he-dapi-paired-stage25b-forward` (private, T4,
  internet off). Staged at
  `/home/bikash/kaggle_stage/stage25b_forward/`.
- **Push command:**
  ```bash
  KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)" \
    .venv/bin/kaggle kernels push -p /home/bikash/kaggle_stage/stage25b_forward
  ```
- **Datasets:** `bikasash/he-dapi-code-stage25` (new version 2026-10-24 03:05:
  real-SSIM validation, eval skips missing test, forward long config),
  `bikasash/he-dapi-he2dapi-legacy512-seed` (new, 78.3 MB, SHA-256
  `6789d7d2…4042aa` = `resources/models/checkpoints/he2dapi/checkpoint_epoch200.pt`),
  `bikasash/hemit-dapi-1024-ch0-macenko` (train/val H5; loader reads
  `mihc[...,2]`).
- **Config:** `configs/hemit_dapi_quality_1024.yaml` — 20 epochs, batch 1 ×
  grad-accum 8, lr 3e-5 (paper), fp32, GAN 1 / L1 50 / SSIM 10 / nuclei Dice 2,
  **colour 0 / gradient 0** (negative-result ablation), DISTS off. Fixed val:
  50 crops from `hemit_val_1024.h5` every epoch; `checkpoint_best.pt` selected
  by `L1 + 0.25*(1 - SSIM_skimage)`.
- **Compute:** ~1,802–1,881 s/epoch measured in Stage 2.5, so 20 epochs ≈
  **10 h**; fits one T4 session. Stop on OOM/no-progress; per-epoch checkpoints
  survive a quota stop.
- **Outputs:** `/kaggle/working/checkpoints/dapi_quality/` (epochs, last, best,
  generators). Pull after completion, then run the eval kernel on val 630 with
  FID/LPIPS (50 metric crops).
- **Gate:** val FID **and** LPIPS below the legacy 512 reference (146.27 /
  0.2468) and below the 1024 ch2 baseline (215.38 / 0.3899), with no visible
  colour collapse/hallucination. Otherwise record a negative result and keep
  the legacy 512 model as default.
- **Concurrent:** `bikasash/he-dapi-paired-stage25-eval` is pushed at the same
  time to evaluate the completed Stage 2.5 reverse checkpoints (val 630 +
  FID/LPIPS); the two runs use separate kernels. Monitor quota; total expected
  spend ≈ 10.5 GPU-h of the fresh weekly allowance.

## 1. Current stage

| Item | State (verified 2026-10-24 03:10 UTC) |
|---|---|
| Roadmap phase | **Phase 2 — Modeling** |
| Sub-stage | **2.5 reverse COMPLETE; 2.5b forward RUNNING (long, 20 ep); 2.5 reverse eval RUNNING** |
| Primary compute | **Kaggle T4 — `bikasash/he-dapi-paired-stage25b-forward`** |
| Primary run status | Pushed 2026-10-24 03:10 UTC; per-epoch checkpoints + best selection |
| **Primary run validity** | Real-SSIM val selection fixed 2026-10-24; test split not attached (val-only gate) |
| Local CPU run | **STOPPED**; WSL protected (see §3) |
| Host safety | **GREEN — C: 12.9 GB free; preflight SAFE at 03:06 UTC** |
| Next stage | Eval reverse (running) → eval forward → full 630/945 + CIs on the winner |

**No release: corrected baselines are undertrained; no model promotion.**
**Do not restart the local CPU run while swap free is below ~300MB.**

---

## 2. What is running / not running now (verified 2026-10-24 03:10 UTC)

### Running
- **Kaggle `bikasash/he-dapi-paired-stage25b-forward`:** pushed 03:10 UTC,
  long 20-epoch 1024 H&E→DAPI fine-tune from the legacy 512 epoch-200 seed.
  Writes `/kaggle/working/checkpoints/dapi_quality/checkpoint_epoch*.pt`,
  `checkpoint_last.pt`, `checkpoint_best.pt`, `generator_*.pt`.
- **Kaggle `bikasash/he-dapi-paired-stage25-eval`:** pushed at the same time to
  evaluate the completed Stage 2.5 reverse checkpoints on val 630 with
  FID/LPIPS. Test is skipped (no `hemit_test_1024.h5` in the attached dataset).
- **No local training.** No output pull active.

### Stage 2.5 / 2.5b status
- **Stage 2.5 reverse:** COMPLETE (4/4 epochs). Output pulled to
  `/home/bikash/kaggle_out_stage25_reverse/`. Best == last (epoch 4); eval
  running.
- **Stage 2.5b forward:** launched 2026-10-24 03:10 UTC; expected ~10 h.
- **Code:** `validate()` now uses real skimage SSIM (fixes inverted best
  selection); eval skips a missing test split and dedupes best==last. Tests:
  **45 passed, 3 deselected**.
- **Datasets:** `bikasash/he-dapi-code-stage25` updated 03:05 UTC;
  `bikasash/he-dapi-he2dapi-legacy512-seed` created 03:07 UTC (ready).

### Completed — R-007 (invalid for selection)
- **Kaggle `bikasash/he-dapi-cyclegan-ch2-256-r-007` v1:** COMPLETE 12:13 UTC,
  exit 0, 10/10 epochs. Eval JSON has `val_FID`/`val_LPIPS` null because the
  kernel omitted `--metrics fid,lpips`; raw metrics are additionally invalid
  due the Tanh/[0,1] mismatch. See §0.5. Paired U-Net remains default.

### Completed — valid for DAPI
- **Kaggle `bikasash/he-dapi-paired-ch2-1024-both`:** COMPLETE 08:55 UTC,
  both directions exit 0, output pulled to `/home/bikash/kaggle_ch2_output`.
  See §0.2–0.3. This is the corrected ch2 baseline, not a release candidate.

### Completed — invalid for DAPI
- **Kaggle R-006:** `bikasash/he-dapi-cyclegan-hemit-dapi2he-256`
  - Status `COMPLETE` at 2026-10-22 18:55 UTC; 10 epochs, 929 steps/epoch,
    AMP, ~414s/epoch; exit 0.
  - Logs: `logs/kaggle_256_final.log`, `logs/kaggle_256_stdout.log`.
  - Output pulled: `cyclegan_G_last.pt` (45.5MB) and
    `eval_cyclegan.json` to `/tmp/k256_eval/`.
  - Training log verified H5-only: `HE: 0 files + H5 …`, `synthetic=False`,
    `device cuda`, `[AMP] mixed precision fp16`.
  - **Target is panCK (`ch0`), not DAPI.** Local 50-crop eval:
    FID 367.97, LPIPS 0.5948, PSNR 11.37, SSIM 0.2216, Pearson 0.1596.
    Do not use these as DAPI metrics. See [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md).
  - Qualitative grid: `docs/qualitative_dapi2he/hemit_val_compare.png`
    (DAPI column is `ch0`/panCK; the CycleGAN output is generic nuclei).

### Cancelled
- **Kaggle (invalid):** `bikasash/he-dapi-cyclegan-hemit-dapi2he-1024`
  - Status `CANCEL_ACKNOWLEDGED` (confirmed 17:19 UTC). Earlier `RUNNING`
    readings were API lag; the user's UI check was correct.
  - Its metrics must not be used; see §4.

### Not running
- Local `train_unpaired` / `train_unpaired_safe.sh`: **none** (`pgrep` empty).
- Paired `train_paired`: **none** (long-finished; `logs/train_done.flag` present).
- Local auto-resume cron for unpaired: **removed** (only the original paired
  `boot_launch.sh` entries remain).
- Local unpaired checkpoint dir
  `resources/models/checkpoints/hemit_cyclegan_dapi2he_256/`: **empty**.
  The old `.../hemit_cyclegan_dapi2he_1024/cyclegan_G_last.pt` (Oct 22 14:52,
  30MB) is from an aborted run, has **no optimizer/D state**, and must **not**
  be resumed.

### Memory state at 10:44 UTC (after reboot)
```
Mem: 3807 total, 1557 used, 2250 available
Swap: 1024 total, 0 used, 1023 free
```
Swap recovered after the reboot and C: cleanup. There is still **no
passwordless sudo**; keep local 4-net training stopped and use Kaggle.

---

## 3. Crash history (what actually happened)

- Boot `-1` (journal boot ID `96a7469f…`) ran **14:38:37 → 15:10:18 UTC** and
  then WSL disappeared (host-side kill, no Linux OOM line in journal).
- `wiki/06-training/profile_unpaired.json` last entries:
  - `15:09:23` and `15:09:45` `cyclegan:start`
  - previous epochs ended with **`swap_free_mb: 0.0`**, RAM ~80–84%
- The crashing config (working tree at that time) was:
  - `batch_size: 2`, `grad_accum: 4`, `num_workers: 2`
  - `mixed_precision: true` (no-op on CPU)
  - 4 networks: 2× ResNet-9 generators + 2× PatchGAN discriminators
  - `he_root: data/raw/hemit` → **10,584 TIFFs**, including label/val/test
- Conclusion: batch 2 + two DataLoader worker copies + 4-net activations
  exceeded the 3.7GB WSL commit and the VM was killed by the host.
- The local run was restarted later with safe settings and reached step 50, but
  global swap hit **1023/1024MB**; it was stopped deliberately at 16:20 UTC to
  avoid a second crash.

---

## 4. Why the 14:25 Kaggle kernel is INVALID (evidence, not assumption)

The kernel itself ran, but the **code bundle it mounted is stale**.

- Code dataset: `bikasash/he-dapi-code-dapi-1024` (5,434,626 bytes,
  last updated `2026-10-22 14:23:48 UTC`).
- Local copy of the uploaded zip: `/home/bikash/kaggle_code/code_bundle.zip`
  (4.9MB, mtime `Oct 22 14:23`).
- Inside that zip:
  - `configs/cyclegan_hemit_dapi2he.yaml` uses
    `he_root: data/raw/hemit`,
    `dapi_root: data/processed/hemit_train_1024.h5`,
    `patch_size: 256`.
  - `data/examples/` **is included** in the zip (7 example PNGs).
  - `src/models/datasets.py` is the **pre-fix** version: it still has the
    `[None]*8` sentinel and the examples fallback before the H5 is resolved.
- Runtime consequence on Kaggle:
  - `data/raw/hemit` does not exist → `he_files` falls back to
    `data/examples/he_*.png` (2 images).
  - DAPI falls back to `data/examples/dapi_*.png` (3 images).
  - With the old 50/50 branch, **~50% of both domains come from those 2–3
    repeated example PNGs**, not HEMIT.
  - Reads are 1024px gzip chunks center-cropped to 256 → ~6s/step, ~15h run.
- **Decision:** cancel this session; do not report its metrics. It is a
  plumbing run at best and a contaminated comparison at worst.

---

## 5. Exact code/data changes made on 2026-10-22 (uncommitted)

Run `git status --short` to see them. Summary:

| File | Change | Why |
|---|---|---|
| `configs/cyclegan_hemit_dapi2he.yaml` | H5-only 256 paths, `he_root` = train/input, `batch_size: 1`, `grad_accum: 8`, `num_workers: 0`, `mixed_precision: false`, `grad_checkpoint: true`, checkpoint dir `..._256` | Prevent WSL OOM; keep effective batch 8; no label/val/test leakage |
| `src/models/datasets.py` | Explicit `he_h5`/`dapi_h5` is now authoritative: no root glob, no `data/examples` fallback, no synthetic sentinel | Stop 50% example-image sampling; make H5-only real |
| `src/models/train_unpaired.py` | ResBlock gradient checkpointing (`torch.utils.checkpoint`, `use_reentrant=False`); checkpoint hardlink (`checkpoint_last.pt` → latest epoch inode) | Cut activation RAM ~1GB and avoid a second 276MB write |
| `tests/test_datasets.py` | Added explicit-H5-authoritative regression test | Lock the fix |
| `scripts/train_unpaired_safe.sh` | New crash-safe local launcher: flock, cgroup `MemoryHigh=2200M/MemoryMax=2500M/SwapMax=128M`, auto-resume, fast-failure backoff | Never kill WSL again |
| `scripts/data/crop_hemit_h5.py` | New 1024→256 center-crop builder (LZF) | Fast 256 reads |
| `AGENTS.md` | Removed obsolete Kaggle **STAND DOWN**; documented `KAGGLE_API_TOKEN` usage | Reflect verified 2026-10-22 reality |
| `docs/LIVE_STATE.md` | This file | Crash-recovery source of truth |
| `scripts/status.sh` | One-shot status snapshot | Avoid assumptions |
| `scripts/kaggle/he-dapi-cyclegan-hemit-dapi2he-1024.*` | Pulled kernel + metadata, preserved for audit | Reproducibility |

Local artifacts:
- `data/processed/hemit_train_256.h5` — 596MB, 3,717 pairs (built 15:07 UTC)
- `data/processed/hemit_val_256.h5` — 104MB, 630 pairs (15:08 UTC)
- `data/processed/hemit_test_256.h5` — 148MB, 945 pairs (15:08 UTC)
- Labels (historical, **WRONG** — do not use): the 1024 H5 `dapi` is `ch0`,
  which the 2026-10-23 audit identifies as panCK. Correct mapping is
  `ch2=DAPI`, `ch0=panCK`, `ch1=CD3`. See `docs/CHANNEL_AUDIT.md`.

---

## 6. Exact commands

### Status
```bash
cd /home/bikash/he-dapi-translator
bash scripts/status.sh
```

### Kaggle status / logs / output
```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
.venv/bin/kaggle kernels status bikasash/he-dapi-cyclegan-hemit-dapi2he-1024
.venv/bin/kaggle kernels logs   bikasash/he-dapi-cyclegan-hemit-dapi2he-1024 | tail -100
.venv/bin/kaggle kernels output bikasash/he-dapi-cyclegan-hemit-dapi2he-1024 -p /tmp/k_hemit_out
```
Live logs are not available through the old API until the session exits; use the
Kaggle web page for live output.

### Cancel the invalid Kaggle session (manual, one click)
1. Open <https://www.kaggle.com/code/bikasash/he-dapi-cyclegan-hemit-dapi2he-1024>
2. Bottom-left **View Active Events** → three dots → **Stop Session**.
3. Confirm `kaggle kernels status …` no longer says `RUNNING`.

### Corrected GPU run (only after the invalid session is stopped)
**Staged and ready as of 2026-10-22 16:32 UTC:**

1. Code dataset **`bikasash/he-dapi-code-hemit-256`** (private, READY) contains
   `code_bundle.zip` (41.8KB, 22 files): fixed `src/models/datasets.py`, the
   T4 config `configs/cyclegan_hemit_dapi2he_kaggle.yaml`, and
   `scripts/data/crop_hemit_h5.py`. No `data/` and no `node_modules`.
2. New kernel script `scripts/kaggle/he-dapi-cyclegan-hemit-dapi2he-256.py`
   slices `hemit_{train,val}_1024.h5` from
   `bikasash/hemit-dapi-1024-ch0-macenko` to 256 LZF H5 **in-kernel** (no
   848MB upload), trains 10 epochs with batch 4 × accum 2 + AMP, then evals on
   `hemit_val_256.h5` (`--num 630`).
3. Staging folder `/tmp/kaggle_kernel_hemit256/` with
   `kernel-metadata.json` (id `bikasash/he-dapi-cyclegan-hemit-dapi2he-256`,
   T4, private).

**Push command (already executed 16:42 UTC; version 1 queued):**
```bash
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
.venv/bin/kaggle kernels push -p /tmp/kaggle_kernel_hemit256
```
Expected runtime: ~10 min crop + ~40–60 min training + ~5 min eval.
Checkpoints/output: `/kaggle/working/checkpoints/hemit_cyclegan_dapi2he_256/`.

Suggested eval after training:
```bash
PYTHONPATH=. .venv/bin/python -m src.models.eval_offline \
  --checkpoint resources/models/checkpoints/hemit_cyclegan_dapi2he_256/cyclegan_G_last.pt \
  --direction dapi2he --h5 data/processed/hemit_val_256.h5 \
  --patch-size 256 --num 630 --label cyclegan-hemit-dapi2he-256 \
  --target 'H&E (HEMIT unpaired)' --out docs/metrics_dapi2he_cyclegan_hemit_256.json
```

### Local resume (only if Kaggle is dead AND swap free > 300MB)
```bash
cd /home/bikash/he-dapi-translator
free -m                      # check SwapFree
rm -f logs/unpaired_done.flag
MAX_STEPS=300 EPOCHS=10 setsid nohup bash scripts/train_unpaired_safe.sh \
  >> logs/train_unpaired.log 2>&1 < /dev/null &
```
The launcher fences the trainer in a 2.5GB cgroup. It does **not** fix global
swap pressure; do not start it on a box with full swap.

---

## 7. Known risks / unknowns (do not assume)

1. **Kaggle session is invalid** — see §4; cancel it.
2. **Kaggle live log** is not visible until exit; status `RUNNING` does not prove
   the config is correct (we proved it is wrong via the bundle zip).
3. **Swap is full** (1022/1024MB) with no passwordless sudo. Local training is
   unsafe until reboot or swap recovery.
4. **Working tree is uncommitted.** Do not run `git checkout --` on
   `configs/`, `src/models/datasets.py`, `src/models/train_unpaired.py`,
   `tests/test_datasets.py`.
5. **Kaggle token was echoed during auth debugging** on 2026-10-22. Rotate it in
   Kaggle account settings; keep the new value in `~/.kaggle/api_token` (600).
6. The local 256 H5 was built from the 1024 H5 with a center crop; it is not a
   native 256 acquisition. This matches the paired baseline crop policy.

---

## 8. Recovery checklist after a crash / reboot

1. `uptime` and `journalctl --list-boots | tail -3` → identify the dead boot.
2. `free -m` → if SwapFree < 300MB, do **not** start training.
3. `pgrep -af 'train_unpaired|train_paired'` → confirm what survived.
4. Read this file and `git status --short`.
5. Local unpaired checkpoints:
   `ls -lt resources/models/checkpoints/hemit_cyclegan_dapi2he_256/`.
6. Kaggle:
   `kaggle kernels status bikasash/he-dapi-cyclegan-hemit-dapi2he-256` and
   `...-1024` (expect `RUNNING`/`CANCEL_ACKNOWLEDGED` respectively).
7. If the Kaggle session finished, pull output and evaluate before launching
   anything new.
8. Update this file with the new state **before** the next launch.

---

## 9. Update log

- `2026-10-22 16:25 UTC` — Created. Recorded crash, invalid Kaggle run, local
  stop, exact changes, commands, and recovery steps.
- `2026-10-22 16:32 UTC` — Staged corrected run: code dataset
  `bikasash/he-dapi-code-hemit-256` READY; 256 kernel script/metadata staged in
  `/tmp/kaggle_kernel_hemit256/`; push blocked on stopping the invalid T4
  session.
- `2026-10-22 16:42 UTC` — Pushed corrected kernel version 1; status `QUEUED`
  behind the invalid 14:25 session. Git checkpoint `3a52ed8`.
- `2026-10-22 17:21 UTC` — Invalid 14:25 kernel confirmed `CANCEL_ACKNOWLEDGED`.
  v1/v2 of the corrected kernel exited in 6s because the wrapper lacked the
  `if __name__ == "__main__"` guard; added it, pushed v3, and confirmed
  `RUNNING` + GPU quota consumption. Added ADR-013.
- `2026-10-22 16:40 UTC` — Added `docs/PLAN.md` v2.0 (stages, gates, ADR log,
  run registry). Marked `WORKPLAN.md` v1.1 historical. Reconciled
  TRACKER/progress/README status. Invalid session still RUNNING; corrected run
  remains staged.
- `2026-10-23 09:52 UTC` — Crash forensics: WSL terminated 09:30:01 due to full
  Windows `C:` (256 MB free); `ext4.vhdx` 155.4 GiB could not grow. Loss audit:
  canonical ch2 checkpoints intact; one 0-byte `dapi_pretrain/checkpoint_epoch14.pt`.
  Added `scripts/crash_guard.sh` + cron heartbeat/boot reports and preflight
  fencing in `boot_launch.sh`, `keepalive_week.sh`, `train_unpaired_safe.sh`.
- `2026-10-23 09:59 UTC` — Evaluated corrected `dapi2he_ch2_1024` at 256 px/50
  val crops: FID 285.29, LPIPS 0.5129. This replaces the invalid ch0
  `342.98 / 0.5904` reference for R-007.
- `2026-10-23 10:00 UTC` — R-007 (`bikasash/he-dapi-cyclegan-ch2-256`, Stage
  2.3) staged at `/home/bikash/kaggle_stage/he-dapi-cyclegan-ch2-256`.
- `2026-10-23 10:37 UTC` — Host recovered: removed 53.1 GB of verified
  duplicates/archives/caches in WSL and 5.2 GB of Windows safe caches;
  `C:` free 0.25 → 12.07 GB, preflight `SAFE`. Checkpoints untouched.
- `2026-10-23 10:38 UTC` — Pushed R-007 v1 as
  `bikasash/he-dapi-cyclegan-ch2-256-r-007`; status `RUNNING`, quota 2.64 GPU-h.
  Monitor `logs/kaggle_r007_monitor.sh` polls every 2 min.
- `2026-10-23 12:13 UTC` — R-007 COMPLETE, exit 0; 10/10 epochs, 416–430 s/epoch.
  Eval JSON has no FID/LPIPS (kernel omitted `--metrics fid,lpips`), and raw
  metrics are invalid due Tanh/[0,1] mismatch. **INVALID for selection**; paired
  U-Net remains default. Filtered pull kept only 91 MB of artifacts.
- `2026-10-23 12:20 UTC` — Fixed `ResnetGenerator` to `Sigmoid` ([0,1] contract),
  added `--metrics fid,lpips` to the CycleGAN kernel, and fixed the backend
  reverse checkpoint (`dapi2he_ch2_1024/generator_last.pt`). Tests: 43 passed,
  3 deselected.
- `2026-10-23 12:41 UTC` — Preflight `SAFE` (C: 9.9 GB, SwapFree 894 MB); one
  warning remains for the known 0-byte
  `kaggle_ch2_output/checkpoints/dapi_pretrain/checkpoint_epoch14.pt` (optional
  epoch snapshot; no resume impact).
- `2026-10-23 12:45 UTC` — Stage 2.5 prepared: val-based best-checkpoint
  selection (`val_L1 + 0.25*(1-val_SSIM)` on 50 fixed 1024 crops), DISTS device
  fix, eval best/last with FID/LPIPS, and staged code/seed/kernel/eval dirs.
  No upload, no push, no GPU spent. Tests: **44 passed, 3 deselected**.
- `2026-10-23 12:54 UTC` — Preflight `SAFE` (C: 9.9 GB, SwapFree 895 MB) before
  the Stage 2.5 upload; known 0-byte optional epoch snapshot still warned.
- `2026-10-23 12:55 UTC` — Uploaded private datasets
  `bikasash/he-dapi-code-stage25` (code_bundle.zip) and
  `bikasash/he-dapi-dapi2he-ch2-1024-seed` (78.3 MB); both `ready`.
- `2026-10-23 12:56 UTC` — Pushed `bikasash/he-dapi-paired-stage25-reverse` v1;
  status `RUNNING`. Fixed the quality-stage resume path so a re-push continues
  from a quality `checkpoint_last.pt` instead of resetting to epoch 1.
- `2026-10-24 02:50 UTC` — Host reboot (new boot ID
  `b15d4b64…`); WSL was down since ~02:35, no crash report was generated.
  After boot: C: 14.0 GB free, SwapFree 1024 MB, no training process;
  preflight `SAFE`. No checkpoint loss.
- `2026-10-24 03:00 UTC` — Stage 2.5 reverse output pulled (518 MB) to
  `/home/bikash/kaggle_out_stage25_reverse/`. Confirmed best == last == epoch 4
  (G hash `a3afeab7…`); checkpoint_best SHA-256 `53c8d3e5…`, last `98b74b76…`.
- `2026-10-24 03:05 UTC` — Fixed `validate()` to use real skimage SSIM (the old
  `SSIMLoss` reported 1-SSIM and inverted best selection); eval now skips a
  missing test split and dedupes best==last. New regression test. Tests:
  **45 passed, 3 deselected**. Updated code dataset `he-dapi-code-stage25`
  (version 2026-10-24 03:05).
- `2026-10-24 03:07 UTC` — Created private dataset
  `bikasash/he-dapi-he2dapi-legacy512-seed` (78.3 MB, legacy epoch-200,
  SHA-256 `6789d7d2…4042aa`); `ready`.
- `2026-10-24 03:10 UTC` — Pushed `bikasash/he-dapi-paired-stage25b-forward`
  (20 epochs, lr 3e-5, no colour/grad, nuclei Dice, real-SSIM val selection)
  and `bikasash/he-dapi-paired-stage25-eval` (reverse val 630 + FID/LPIPS).
  Expected spend ≈10.5 GPU-h of the fresh weekly quota.
- `2026-10-24 03:21 UTC` — Staged the reverse control kernel
  `bikasash/he-dapi-paired-stage25b-reverse` at
  `/home/bikash/kaggle_stage/stage25b_reverse/`: 20 epochs, lr 3e-5, no
  colour/grad, nuclei Dice off, seeded from the corrected 1024 reverse
  baseline. New config `configs/hemit_dapi2he_1024_long.yaml`; code dataset
  updated with `DAPI_QUALITY_CONFIG` override (version 2026-10-24 03:21).
- `2026-10-24 03:52 UTC` — Stage 2.5 reverse local eval (n=50, 1024):
  **FID 284.27 / LPIPS 0.5221 / SSIM 0.5024 / PSNR 15.24 / L1 0.1520 /
  Pearson 0.6654** vs baseline **282.92 / 0.5317 / 0.4891 / 15.42 / 0.1505 /
  0.6644**. Mixed: LPIPS/SSIM improved, FID/L1/PSNR slightly worse; the
  colour/grad recipe is not a clear win and is not the bottleneck.
  JSON: `docs/metrics_stage25_reverse_best_val50_local_fid.json`.
  The Kaggle eval kernel hung ~55 min on the FID/LPIPS weight download and was
  cancelled; local eval used the cached weights instead.
- `2026-10-24 03:56 UTC` — Pushed Stage 2.5b reverse control
  `bikasash/he-dapi-paired-stage25b-reverse` v1 (RUNNING, ~10 h): 20 epochs,
  lr 3e-5, no colour/grad, nuclei Dice off, seeded from the corrected 1024
  reverse baseline. Config `configs/hemit_dapi2he_1024_long.yaml`; code dataset
  version 2026-10-24 03:22 adds `DAPI_QUALITY_CONFIG`. Monitor now watches both
  forward and reverse and pulls filtered best/last/generators/logs.
- `2026-10-24 08:05 UTC` — Live Kaggle snapshot (via
  `kaggle kernels logs -f <kernel>`, which works for running kernels):
  - forward: epoch **10/20** started; val best so far **epoch 1**
    (L1 0.0252 / SSIM 0.7656 / PSNR 28.68 / score 0.0838); later epochs
    oscillate (SSIM 0.736–0.758), train L1 still falling → mild overfit.
  - reverse: epoch **8/20** started; val best so far **epoch 4**
    (L1 0.1469 / SSIM 0.5136 / PSNR 15.71 / score 0.2685), already better
    than the 1024 reverse baseline (L1 0.1505 / SSIM 0.4891 / PSNR 15.42).
  - Both ~33–36 min/epoch; expected terminal ~14:00–16:00 UTC.
- `2026-10-24 13:12 UTC` — **Stage 2.5b forward COMPLETE** (20/20, exit 0).
  Best **epoch 17**: val L1 0.0237 / SSIM 0.7771 / PSNR 28.86 / Pearson
  0.5851 / Dice 0.2018 (score 0.0794). Local n=50 eval: **FID 169.53 /
  LPIPS 0.2706**. vs 1024 ch2 baseline (215.38 / 0.3899 / SSIM 0.7733 /
  Dice 0.0828) it improves every metric; vs legacy 512 (146.27 / 0.2468 /
  SSIM 0.7662 / Dice 0.1638) it wins SSIM/Pearson/Dice but loses
  FID/LPIPS/PSNR/L1. Caveat: legacy is 512, new is 1024. Output pulled to
  `/home/bikash/kaggle_out_stage25b_forward` (best SHA `3850b8ba…`).
  Preflight was briefly blocked at C: 8094 MB; removed 262 MB of >1-day
  Windows temp files, preflight SAFE, then pulled.
- `2026-10-24 14:48 UTC` — **Stage 2.5b reverse COMPLETE** (20/20, exit 0).
  Best **epoch 20** (final epoch): val L1 0.1447 / SSIM 0.5181 / PSNR 15.84
  (score 0.2651); epoch 19 had regressed to 0.2979. Output pulled to
  `/home/bikash/kaggle_out_stage25b_reverse` (best SHA `652438e1…`); FID/LPIPS
  eval running locally. Structural metrics already beat baseline
  (0.1505 / 0.4891 / 15.42) and Stage 2.5 (0.1520 / 0.5024 / 15.24).
- `2026-10-25 05:00 UTC` — Fair 1024 forward comparison: legacy 512
  checkpoint run at 1024 gives **FID 129.99 / LPIPS 0.2977 / PSNR 28.25 /
  SSIM 0.6562 / Pearson 0.5846 / L1 0.0289 / Dice 0.4315**; Stage 2.5b 1024
  is **169.53 / 0.2706 / 28.86 / 0.7771 / 0.5851 / 0.0237 / 0.2018**. Legacy
  wins FID and nuclei Dice; Stage 2.5b wins LPIPS/PSNR/SSIM/L1. **Legacy 512
  remains the forward default.** JSON:
  `docs/metrics_legacy512_at1024_val50_local.json`.

## Stage 2.5b reverse extension — launched 2026-10-26 04:10 UTC
- **Kernel:** `bikasash/he-dapi-paired-stage25b-reverse-ext` (private, GPU T4)
- **Purpose:** 10 more clean reverse epochs (21-30) from the epoch-20 best.
- **Seed dataset:** `bikasash/he-dapi-dapi2he-stage25b-best-seed`,
  `checkpoint_last.pt` SHA `652438e1906a61994c1af8db5e8f0932b91184f2722deacdf1c751fd360fd108`.
- **Config:** `configs/hemit_dapi2he_1024_long.yaml` (lr 3e-5, no colour/grad,
  nuclei Dice 0, val 50/epoch, batch 1 x grad_accum 8).
- **Command:** `kaggle kernels push -p /home/bikash/kaggle_stage/stage25b_reverse_ext`
- **Output:** `/kaggle/working/checkpoints/dapi2he_quality/`; local pull target
  `/home/bikash/kaggle_out_stage25b_reverse_ext`.
- **Monitor:** `logs/kaggle_stage25b_reverse_ext_monitor.sh` PID `46882`,
  log `logs/kaggle_stage25b_reverse_ext_monitor.log`.
- **Resume:** rerun the same kernel push (seed is fixed), or locally resume
  `checkpoint_best.pt` with `--epochs 30` using the runner args in
  `scripts/kaggle/train_dapi_1024.py`.
