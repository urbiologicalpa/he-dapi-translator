# LIVE STATE — 2026-10-22 16:40 UTC

> **Canonical strategy/decisions: [`docs/PLAN.md`](PLAN.md) v2.0.**
> **Read this file first after any WSL crash, reboot, or "what is running?" question.**
> It is the source of truth for the active run. Update it **before** launching and
> **after** every exit/crash. Do not rely on memory or assumptions.
>
> Machine: WSL2, 3.7GB RAM, 1GB swap, no GPU, 4c/8t i5-10210U.
> Repo: `/home/bikash/he-dapi-translator`.

---

## 1. Current stage

| Item | State (verified 2026-10-22 16:24 UTC) |
|---|---|
| Roadmap phase | **Phase 2 — Modeling** |
| Sub-stage | **2.3 HEMIT-domain CycleGAN (DAPI→H&E) fair comparison** |
| Primary compute | Kaggle T4, kernel `bikasash/he-dapi-cyclegan-hemit-dapi2he-1024` |
| Primary run status | `KernelWorkerStatus.RUNNING` since **2026-10-22 14:25:30 UTC** |
| **Primary run validity** | **INVALID — do not use its results** (see §4) |
| Local CPU run | **STOPPED** at ~step 50; WSL protected (see §3) |
| Next stage | Cancel invalid Kaggle session → push corrected 256px run (§6) |

**Do not start another GPU kernel while the invalid session still holds the T4.**
**Do not restart the local CPU run while swap free is below ~300MB.**

---

## 2. What is running / not running now (verified 16:24 UTC)

### Running
- **Kaggle:** `bikasash/he-dapi-cyclegan-hemit-dapi2he-1024`
  - Created/pushed: `2026-10-22 14:25:30 UTC`
  - `kaggle kernels status` → `KernelWorkerStatus.RUNNING`
  - Expected output dir: `/kaggle/working/checkpoints/hemit_cyclegan_dapi2he_1024/`
  - Expected files: `checkpoint_epoch*.pt`, `checkpoint_last.pt`,
    `cyclegan_G_last.pt`, `generator_last.pt`, `eval_cyclegan.json`
  - **Known-invalid data mix; see §4.**

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

### Memory state at 16:24 UTC
```
Mem: 3807 total, 1941 used, 1866 available
Swap: 1024 total, 1022 used, ~1 free
```
Swap is effectively full and there is **no passwordless sudo**, so do not try
`swapoff/swapon` non-interactively. Local training must stay stopped until swap
free recovers or the box is rebooted.

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

## 4. Why the running Kaggle kernel is INVALID (evidence, not assumption)

The running kernel itself is fine, but the **code bundle it mounted is stale**.

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
- Labels: `ch0=DAPI`, `ch1=CD3`, `ch2=panCK`.

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

**Push command (do NOT run while the invalid session is RUNNING):**
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
   `kaggle kernels status bikasash/he-dapi-cyclegan-hemit-dapi2he-1024`.
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
- `2026-10-22 16:40 UTC` — Added `docs/PLAN.md` v2.0 (stages, gates, ADR log,
  run registry). Marked `WORKPLAN.md` v1.1 historical. Reconciled
  TRACKER/progress/README status. Invalid session still RUNNING; corrected run
  remains staged.
