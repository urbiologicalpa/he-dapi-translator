# Progress tracker — 2026-10-21

> Canonical short status. Machine-readable counterpart: `docs/progress.json`.
> Detailed pre-2026-10-20 logs remain in git history and `MEMORY.md`.

## Critical correction

HEMIT label TIFFs are **ch0=DAPI, ch1=CD3, ch2=panCK**. The previous pipeline
used `ch2` as the target. Consequences:

- The epoch-199/200 CPU run is an **H&E → panCK ablation**.
- `docs/metrics_eval.json`, `generator_last.pt`, the HF weights, and the current
  demo are **not DAPI**.
- The bug is fixed in `scripts/data/pack_hemit.py`,
  `scripts/data/pack_hemit_1024.py`, and `src/models/datasets.py`; regression
  tests are in `tests/test_hemit_channels.py`.
- Corrected 1024px HEMIT H5 has been built and verified for all splits.

## Live runs

| Run | Status | Details |
|---|---|---|
| Local CPU, 512px | **Complete** | Epoch 200 finished 2026-10-20 19:34 UTC; full 630/945 panCK eval recorded. Do not treat as DAPI. |
| Kaggle T4 | DAPI baseline complete | 512 pretrain + 1024 fine-tune + full 630/945 eval done. Next: reverse paired. |
| Corrected 1024px DAPI | Baseline done, undertrained | val/test SSIM 0.670/0.725, Pearson 0.255/0.611. |
| DAPI → H&E | First baseline done | 20/20 @512 + 2×500 @1024; val/test SSIM 0.437/0.593, Pearson 0.521/0.576. Undertrained; quality pass next. |

## Phases

| Phase | Status | % | Reality check |
|---|---|---:|---|
| 0. Scoping / infra | Done | 100 | Repo, CI, Docker, packages exist |
| 1. Data pipeline | Done, corrected | 100 | 5,292 HEMIT pairs; 512px legacy data has wrong target; 1024 builder ready |
| 2. H&E → DAPI model | Baseline done | 60 | corrected DAPI baseline trained; undertrained; dual-branch not implemented |
| 3. App / API | Alpha | 60 | FastAPI + React + Docker; model behind it is the ablation |
| 4. Validation | Partial | 35 | Full 630/945 DAPI baseline + panCK ablation eval done; no CIs/qualitative audit yet |
| 5. Deployment | Alpha | 55 | GHCR staging, PyPI/HF alpha; corrected weights and CPU benchmark pending |

Overall is intentionally not presented as a single impressive percentage: the
only completed model is the panCK ablation; DAPI training is still in progress.

## Next actions

1. DONE — qualitative reverse audit (`docs/qualitative_dapi2he/`).
2. DONE (negative) — FID/LPIPS + colour/gradient quality pass ran; it did not
   improve the reverse model (FID 313→335, SSIM 0.496→0.458). Keep baseline.
3. IN PROGRESS — CycleGAN implemented; MHIST-domain run done but worse than
   paired reverse (FID 363 vs 343, confounded target domain). HEMIT-domain rerun
   `bikasash/he-dapi-cyclegan-hemit-dapi2he-1024` RUNNING for a fair comparison.
4. DONE — auto direction router (`detect_direction`, threshold 0.06) in CLI/streaming/API.
5. DONE (negative) — forward quality kernel completed; colour/gradient pass
   degraded FID/LPIPS vs the baseline (see RESULTS.md). Baseline remains default;
   val-based selection + dual-branch Swin/FMF still open.
6. DONE — CPU/ONNX/INT8 benchmark; ship ONNX FP32 (`docs/BENCHMARK.md`).

## Commands

```bash
# status
PYTHONPATH=. python -m src.utils.progress_tracker --check
PYTHONPATH=. python -m src.utils.progress_tracker --enforce
PYTHONPATH=. python -m pytest tests/test_hemit_channels.py tests/test_datasets.py -q

# corrected data
uv run --extra data python -m scripts.data.pack_hemit_1024 \
  --splits train val test --patch-size 1024 --compression gzip

# corrected DAPI training (on GPU, after data upload)
PYTHONPATH=. python -m src.models.train_paired \
  --config configs/hemit_dapi_512_pretrain.yaml --resume auto --optimize-for full
PYTHONPATH=. python -m src.models.train_paired \
  --config configs/hemit_dapi_1024.yaml --resume auto --optimize-for full
```

Kaggle data/code upload, smoke, pretrain, and fine-tune steps are in
[`docs/KAGGLE_DAPI.md`](KAGGLE_DAPI.md).

## Milestones kept

- 2026-08-31: initial HEMIT/maxpmx/MHIST extraction.
- 2026-09-19: low-RAM streaming pass; 11 resource guards.
- 2026-10-18: v0.1.0 GHCR release, HF demo, PyPI alpha packages.
- 2026-10-20: DAPI channel-order correction; Kaggle T4 availability confirmed;
  README/docs rewritten to match what is actually implemented.
- 2026-10-21: corrected DAPI 512 pretrain + 1024 fine-tune + full 630/945 eval.
  Baseline is undertrained. Added bidirectional trainer support and launched
  paired reverse DAPI → H&E on Kaggle.
