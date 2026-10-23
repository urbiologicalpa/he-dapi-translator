# Progress tracker — 2026-10-23

> **Canonical plan: [`docs/PLAN.md`](PLAN.md) v2.0.** Live run and recovery:
> [`docs/LIVE_STATE.md`](LIVE_STATE.md). This file is the short status only.
> Machine-readable counterpart: `docs/progress.json`.

## BLOCKER — channel audit reverses the previous correction (2026-10-23)

The raw HEMIT RGB TIFF is **ch2=DAPI (blue), ch0=panCK (red), ch1=CD3
(green)**. Therefore the legacy 512 model, public HF weights, demo, and
`docs/metrics_eval.json` are the **real H&E→DAPI** model, not a panCK ablation.
The 1024 `ch0` forward/reverse models and R-006 are **panCK**, not DAPI.
All DAPI claims and runs are frozen. ADR-014 is accepted; the 256px data
rebuild is complete and verified. Legacy 512 is relabeled DAPI; the 1024 `ch0`
and R-006 labels are corrected. A corrected 1024/`ch2` DAPI baseline is still
pending. See [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md).

## Live runs

| Run | Status | Details |
|---|---|---|
| Local CPU, 512px ch2 | **Complete — real DAPI** | 200 epochs; full 630/945 metrics; relabeled from "panCK". |
| Forward paired 1024, ch0 | **Complete — panCK, not DAPI** | val/test SSIM 0.670/0.725, Pearson 0.255/0.611. Freeze as DAPI. |
| Reverse paired 1024, ch0 | **Complete — panCK→H&E, not DAPI** | val/test SSIM 0.437/0.593, Pearson 0.521/0.576. Do not present as DAPI. |
| Paired 1024, ch2, both | **RUNNING** | `bikasash/he-dapi-paired-ch2-1024-both` v1; 14@512 + 2×500@1024 each; ~4–5 h. |
| CycleGAN MHIST H&E | **Complete, negative/confounded** | FID 362.34 vs paired 342.98; target domain differed. |
| CycleGAN HEMIT (14:25) | **INVALID — cancelled** | `CANCEL_ACKNOWLEDGED`; stale code bundle mixed `data/examples`; do not use metrics. |
| CycleGAN HEMIT 256 (R-006) | **COMPLETE — panCK ch0; invalid for DAPI** | 10 epochs, exit 0; val-50 FID 367.97, LPIPS 0.5948, SSIM 0.2216; do not cite as DAPI. |
| Channel audit | **BLOCKER — awaiting owner** | Proposed ADR-014 reverses ch0/ch2; rebuild H5 on ch2 before any DAPI run. |

## Phases (evidence-gated; see PLAN.md)

| Stage | Status | Reality check |
|---|---|---|
| 0. Truth & infra | **DONE** | CI, Docker, guards, channel regression tests. |
| 1. Data contract | **DONE** | Corrected 1024 + derived 256 H5; no leakage; H5 authoritative. |
| 2. Model baselines | **BLOCKED** | Valid DAPI is the legacy 512 `ch2` model; the 1024 `ch0` models and R-006 are panCK. A corrected 1024/`ch2` baseline is pending. |
| 3. Validation | **PARTIAL** | Legacy 512 DAPI and panCK ablation metrics exist; corrected `ch2` DAPI run pending. |
| 4. Product integration | **ALPHA** | Router/API/UI/ONNX exist; legacy 512 DAPI weights need re-verification. |
| 5. Release | **ALPHA** | Frozen; no DAPI promotion until the legacy checkpoint is re-verified and a `ch2` DAPI baseline exists. |
| 6. Post-release | **NOT STARTED** | No monitoring/drift loop yet. |

The only near-paper-quality model is the legacy 512 **H&E→DAPI (`ch2`)** model.
The 1024 `ch0` forward/reverse models and R-006 are panCK; there is no valid
DAPI→H&E model.

## Next actions

1. ~~Accept ADR-014 (`ch2=DAPI`, `ch0=panCK`).~~ Done.
2. ~~Rebuild the 256 H5 with `dapi=mihc[...,2]`; update builder, loader, tests,
   and configs.~~ Done and verified.
3. Re-verify the legacy 512 DAPI checkpoint and record SHA-256 provenance.
4. Only if reverse DAPI→H&E is still required: train paired and CycleGAN on the
   rebuilt `ch2` data (do not reuse `ch0` checkpoints).
5. Rotate the exposed Kaggle token.

## Commands

```bash
# status snapshot (local + Kaggle)
bash scripts/status.sh

# tests
PYTHONPATH=. python -m pytest tests -q -m "not slow"

# corrected data (already built)
uv run --extra data python -m scripts.data.pack_hemit_1024 \
  --splits train val test --patch-size 1024 --compression gzip

# local unpaired resume (only if SwapFree > 300MB)
MAX_STEPS=300 EPOCHS=10 setsid nohup bash scripts/train_unpaired_safe.sh \
  >> logs/train_unpaired.log 2>&1 < /dev/null &

# corrected Kaggle run (after the invalid session is stopped)
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
.venv/bin/kaggle kernels push -p /tmp/kaggle_kernel_hemit256
```

Kaggle data/code upload, smoke, pretrain, and fine-tune history:
[`docs/KAGGLE_DAPI.md`](KAGGLE_DAPI.md).

## Milestones kept

- 2026-08-31: initial HEMIT/maxpmx/MHIST extraction.
- 2026-09-19: low-RAM streaming pass; 11 resource guards.
- 2026-10-18: v0.1.0 GHCR release, HF demo, PyPI alpha packages.
- 2026-10-20: DAPI channel-order correction; Kaggle T4 confirmed; docs
  rewritten to match implemented code.
- 2026-10-21: legacy 512 DAPI model + 1024 ch0 panCK run + full 630/945 eval;
  added bidirectional trainer and paired reverse run.
- 2026-10-22: reverse paired baseline eval; WSL host OOM during unpaired run;
  invalid stale-bundle Kaggle run identified; H5-authoritative loader,
  grad-checkpointing, 2.5GB cgroup launcher; PLAN v2.0 and LIVE_STATE added.
