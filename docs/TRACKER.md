# Progress tracker — 2026-10-22

> **Canonical plan: [`docs/PLAN.md`](PLAN.md) v2.0.** Live run and recovery:
> [`docs/LIVE_STATE.md`](LIVE_STATE.md). This file is the short status only.
> Machine-readable counterpart: `docs/progress.json`.

## Critical correction (still in force)

HEMIT label TIFFs are **ch0=DAPI, ch1=CD3, ch2=panCK**. The previous pipeline
used `ch2`, so the epoch-199/200 CPU run, public HF weights, demo, and
`docs/metrics_eval.json` are an **H&E → panCK ablation**, not DAPI. The data
builders, loader, and tests are fixed; corrected 1024px H5 is built for all
splits. Do not present the public weights as DAPI.

## Live runs

| Run | Status | Details |
|---|---|---|
| Local CPU, 512px panCK | **Complete/ablation** | 200 epochs; full 630/945 panCK eval. Not DAPI. |
| Forward paired DAPI 1024 | **Complete, undertrained** | val/test SSIM 0.670/0.725, Pearson 0.255/0.611. |
| Reverse paired H&E 1024 | **Complete, undertrained** | val/test SSIM 0.437/0.593, Pearson 0.521/0.576. |
| CycleGAN MHIST H&E | **Complete, negative/confounded** | FID 362.34 vs paired 342.98; target domain differed. |
| CycleGAN HEMIT (14:25) | **INVALID — cancel** | Stale code bundle mixed `data/examples`; do not use metrics. |
| CycleGAN HEMIT corrected 256 | **Staged — launch next** | Code dataset `bikasash/he-dapi-code-hemit-256` READY; kernel `/tmp/kaggle_kernel_hemit256/`. |

## Phases (evidence-gated; see PLAN.md)

| Stage | Status | Reality check |
|---|---|---|
| 0. Truth & infra | **DONE** | CI, Docker, guards, channel regression tests. |
| 1. Data contract | **DONE** | Corrected 1024 + derived 256 H5; no leakage; H5 authoritative. |
| 2. Model baselines | **IN PROGRESS** | Forward/reverse paired done; fair unpaired comparison next; selection gate open. |
| 3. Validation | **PARTIAL** | Full 630/945 paired metrics; CIs, blinded audit, failure analysis open. |
| 4. Product integration | **ALPHA** | Router/API/UI/ONNX exist; public weights are still the ablation. |
| 5. Release | **ALPHA** | Staging/hand manual prod gate; corrected weights/manifest pending. |
| 6. Post-release | **NOT STARTED** | No monitoring/drift loop yet. |

Overall is intentionally not a single percentage: the only fully completed
model is the panCK ablation; corrected DAPI/H&E models are undertrained.

## Next actions

1. **Stop the invalid 14:25 Kaggle session**; launch the corrected 256 kernel;
   verify `HE: 0 files + H5 …`, `synthetic=False`, CUDA true.
2. Write `docs/selection_reverse.md` after the corrected run: paired vs
   unpaired on **val** FID/LPIPS + blinded qualitative; choose the default.
3. Spend the next GPU budget only on the selected architecture's quality pass
   (2.5), with val-based checkpoint selection.
4. Update the public README status (router and CycleGAN are implemented; public
   weights remain the ablation) before any new claim.
5. Rotate the Kaggle token exposed during auth debugging.

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
- 2026-10-21: corrected DAPI 512 pretrain + 1024 fine-tune + full 630/945 eval;
  added bidirectional trainer and paired reverse run.
- 2026-10-22: reverse paired baseline eval; WSL host OOM during unpaired run;
  invalid stale-bundle Kaggle run identified; H5-authoritative loader,
  grad-checkpointing, 2.5GB cgroup launcher; PLAN v2.0 and LIVE_STATE added.
