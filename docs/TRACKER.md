# Progress tracker — 2026-10-25

> **Canonical plan: [`docs/PLAN.md`](PLAN.md) v2.0.** Live run and recovery:
> [`docs/LIVE_STATE.md`](LIVE_STATE.md). This file is the short status only.
> Machine-readable counterpart: `docs/progress.json`.

## Channel audit — RESOLVED (ADR-014, 2026-10-23)

The raw HEMIT RGB TIFF is **ch2=DAPI (blue), ch0=panCK (red), ch1=CD3
(green)**. Therefore the legacy 512 model, public HF weights, demo, and
`docs/metrics_eval.json` are the **real H&E→DAPI** model, not a panCK ablation.
The 1024 `ch0` forward/reverse models and R-006 are **panCK**, not DAPI.
ADR-014 is accepted; the 256px data rebuild is complete and verified. Legacy
512 is relabeled DAPI; the 1024 `ch0` and R-006 labels are corrected. The
corrected 1024/`ch2` baseline is complete but undertrained (see `RESULTS.md`).
R-007 (unpaired `ch2` 256) completed 12:13 UTC but is **INVALID for selection**
(no FID/LPIPS; Tanh/[0,1] mismatch). The paired U-Net remains default. Stage 2.5
reverse is complete (4/4 epochs); its Kaggle eval hung on the FID/LPIPS weight
download and was cancelled, so it was evaluated locally (FID 284.27 / LPIPS
0.5221 / SSIM 0.5024 vs baseline 282.92 / 0.5317 / 0.4891 — mixed). Stage 2.5b
forward (legacy 512 seed) and reverse (clean control) both completed 20-epoch
1024 fine-tunes on 2026-10-24. Forward best epoch 17 (FID 169.53 / LPIPS
0.2706); reverse best epoch 20 (FID 239.18 / LPIPS 0.4875). See
[`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md).

## Live runs

| Run | Status | Details |
|---|---|---|
| Local CPU, 512px ch2 | **Complete — real DAPI** | 200 epochs; full 630/945 metrics; relabeled from "panCK". |
| Forward paired 1024, ch0 | **Complete — panCK, not DAPI** | val/test SSIM 0.670/0.725, Pearson 0.255/0.611. Do not present as DAPI. |
| Reverse paired 1024, ch0 | **Complete — panCK→H&E, not DAPI** | val/test SSIM 0.437/0.593, Pearson 0.521/0.576. Do not present as DAPI. |
| Paired 1024, ch2, both | **COMPLETE — valid, undertrained** | `bikasash/he-dapi-paired-ch2-1024-both`; exit 0 08:55 UTC; 50-crop eval; checkpoints SHA-verified. |
| R-007 CycleGAN ch2 256 | **COMPLETE 12:13 UTC — INVALID for selection** | `bikasash/he-dapi-cyclegan-ch2-256-r-007` v1; 10/10 epochs, exit 0; no FID/LPIPS (kernel omitted `--metrics`), Tanh/[0,1] mismatch; paired U-Net remains default. |
| Stage 2.5 paired reverse 1024 | **COMPLETE ~15:00 UTC 2026-10-23; eval local** | `bikasash/he-dapi-paired-stage25-reverse` v1; 4/4 epochs, exit 0; output pulled (518 MB); best == last == epoch 4. Kaggle eval cancelled after hanging on FID/LPIPS download; local n=50 eval: FID 284.27 / LPIPS 0.5221 / SSIM 0.5024 / PSNR 15.24. |
| Stage 2.5b paired forward 1024 | **COMPLETE 13:12 UTC 2026-10-24** | `bikasash/he-dapi-paired-stage25b-forward`; 20/20 epochs, exit 0; best epoch 17. Val L1 0.0237 / SSIM 0.7771 / PSNR 28.86 / Pearson 0.5851 / Dice 0.2018; local FID 169.53 / LPIPS 0.2706. Beats the 1024 baseline everywhere, not legacy 512 on FID/LPIPS. |
| Stage 2.5b paired reverse 1024 | **COMPLETE 14:48 UTC 2026-10-24** | `bikasash/he-dapi-paired-stage25b-reverse`; 20/20 epochs, exit 0; best epoch 20. Val L1 0.1447 / SSIM 0.5181 / PSNR 15.84 / Pearson 0.6919; local FID 239.18 / LPIPS 0.4875. Beats the 1024 baseline and Stage 2.5 on every metric. |
| Host safety | **GREEN** | `C:` ~8.8 GB free; preflight `SAFE` at 2026-10-25 04:42 UTC. |
| CycleGAN MHIST H&E | **Complete, negative/confounded** | FID 362.34 vs paired 342.98; target domain differed. |
| CycleGAN HEMIT (14:25) | **INVALID — cancelled** | `CANCEL_ACKNOWLEDGED`; stale code bundle mixed `data/examples`; do not use metrics. |
| CycleGAN HEMIT 256 (R-006) | **COMPLETE — panCK ch0; invalid for DAPI** | 10 epochs, exit 0; val-50 FID 367.97, LPIPS 0.5948, SSIM 0.2216; do not cite as DAPI. |
| Channel audit | **RESOLVED — ADR-014 accepted** | H5 rebuilt on `ch2`; legacy 512 relabeled DAPI; `ch0`/R-006 marked panCK. |

## Phases (evidence-gated; see PLAN.md)

| Stage | Status | Reality check |
|---|---|---|
| 0. Truth & infra | **DONE** | CI, Docker, guards, channel regression tests. |
| 1. Data contract | **DONE** | Corrected 1024 + derived 256 H5; no leakage; H5 authoritative. |
| 2. Model baselines | **PAIRED SELECTED; 2.5 DONE; 2.5b RUNNING** | Valid forward/reverse `ch2` 1024 baselines exist but are undertrained; R-007 unpaired invalid; Stage 2.5 reverse complete and locally evaluated; Stage 2.5b forward + reverse long runs launched 03:10/03:56 UTC. |
| 3. Validation | **PARTIAL** | 50-crop ch2 metrics exist; selection memo closed (paired); full 630/945 val/test and CIs pending. |
| 4. Product integration | **ALPHA** | Router/API/UI/ONNX exist; legacy 512 DAPI weights verified (H5 ch2 + HF SHA match); demo/API SHA manifest pending. |
| 5. Release | **ALPHA** | Frozen; no promotion until the corrected `ch2` baseline passes full-split/CIs and the reverse model is release quality. |
| 6. Post-release | **NOT STARTED** | No monitoring/drift loop yet. |

The only near-paper-quality model is the legacy 512 **H&E→DAPI (`ch2`)** model.
The corrected 1024 `ch2` forward/reverse are valid but not release quality;
R-007 is invalid for selection; the 1024 `ch0` models and R-006 are panCK.

## Next actions

1. ~~Accept ADR-014 (`ch2=DAPI`, `ch0=panCK`).~~ Done.
2. ~~Rebuild the 256 H5 with `dapi=mihc[...,2]`.~~ Done and verified.
3. ~~**Free ≥10 GB on Windows `C:`**~~ Done 10:37 UTC; preflight `SAFE`.
4. ~~Push and finish R-007~~ Done 10:38–12:13 UTC; 10/10 epochs, exit 0.
   **Invalid for selection**: no FID/LPIPS and a Tanh/[0,1] mismatch. Do not
   rerun CycleGAN.
5. ~~Write the selection memo~~ Done 12:20 UTC; `docs/selection_reverse.md`
   selects the paired U-Net for Stage 2.5. No bootstrap CI is possible.
6. Rotate the exposed Kaggle token.
7. ~~Stage 2.5 reverse~~ Done 2026-10-23 ~15:00 UTC (4/4 epochs). Output
   preserved at `/home/bikash/kaggle_out_stage25_reverse/`.
8. ~~Stage 2.5 reverse eval~~ Done locally 03:52 UTC (Kaggle eval cancelled
   after hanging on FID/LPIPS download). Result is mixed: LPIPS/SSIM better,
   FID/L1/PSNR slightly worse than baseline.
9. ~~Stage 2.5b forward~~ Complete 13:12 UTC (20/20). Best epoch 17; local
   FID 169.53 / LPIPS 0.2706. Beats the 1024 baseline on every metric. Fair
   1024 comparison (2026-10-25): legacy 512 @1024 has FID 129.99 / Dice 0.4315
   vs Stage 2.5b FID 169.53 / Dice 0.2018. Legacy wins FID/Dice; Stage 2.5b
   wins LPIPS/PSNR/SSIM/L1. Legacy remains forward default.
10. ~~Stage 2.5b reverse~~ Complete 14:48 UTC (20/20). Best epoch 20: local
    FID 239.18 / LPIPS 0.4875, better than baseline (282.92/0.5317) and
    Stage 2.5 (284.27/0.5221) on every metric. Output pulled.
11. ~~Stage 2.5b reverse extension~~ Complete 2026-10-26 09:53 UTC (10/10).
    Best composite **epoch 22** (score 0.2622, L1 0.1393, PSNR 15.93), but
    local eval is mixed: seed **epoch 20** still wins FID 239.18 / LPIPS
    0.4875 / SSIM 0.5181 / Pearson 0.6919. **Stop extending reverse;** keep
    epoch 20 as perceptual default, epoch 22 as L1/PSNR alternative. Output
    pulled and evaluated. Forward needs a Dice-preserving objective.
12. Then full 630/945 val/test + bootstrap CIs on the selected checkpoints, and
    align the public/API/Docker checkpoint SHA before v0.2.0 (HF epoch169 vs
    API epoch200); ask before publishing.

## Commands

```bash
# status snapshot (local + Kaggle)
bash scripts/status.sh

# tests
PYTHONPATH=. python -m pytest tests -q -m "not slow"

# corrected data (already built)
uv run --extra data python -m scripts.data.pack_hemit_1024 \
  --splits train val test --patch-size 1024 --compression gzip

# host safety gate (must print RESULT: SAFE before any launch/pull)
bash scripts/crash_guard.sh --preflight

# R-007 (completed; status/artifacts only)
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
.venv/bin/kaggle kernels status bikasash/he-dapi-cyclegan-ch2-256-r-007

# Stage 2.5 reverse (complete; eval local, Kaggle eval deleted)
.venv/bin/kaggle kernels status bikasash/he-dapi-paired-stage25-reverse

# Stage 2.5b forward + reverse (running, ~10 h each)
KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)" \
  .venv/bin/kaggle kernels status bikasash/he-dapi-paired-stage25b-forward
KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)" \
  .venv/bin/kaggle kernels status bikasash/he-dapi-paired-stage25b-reverse
# monitor: tail -f logs/kaggle_stage25b_monitor.log
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
- 2026-10-23: ADR-014 `ch2` data rebuild + corrected paired 1024 both-direction
  run (complete, undertrained); WSL VM terminated by full Windows `C:`; added
  `crash_guard.sh` + cron forensics; reclaimed 58.3 GB and pushed R-007
  (`...-ch2-256-r-007` v1, COMPLETE 12:13 UTC but invalid for selection).
- 2026-10-23 12:45: Stage 2.5 prepared (val-based best-checkpoint selection,
  colour/gradient losses, FID/LPIPS eval, staged code/seed/kernels); tests 44
  passed. Uploaded private datasets and pushed
  `bikasash/he-dapi-paired-stage25-reverse` at 12:56 UTC; running/resumable.
