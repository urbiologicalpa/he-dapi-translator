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
| Kaggle T4 | 512 pretrain running | Kernel v5, epoch 10/20 at 12:43 UTC; ~26.5 min/epoch; checkpoints every epoch. |
| Corrected 1024px DAPI | Pretraining | 20-epoch 512 run in progress; then 2×500-step 1024 fine-tune. |
| DAPI → H&E | Not started | Trainer is a smoke-test skeleton; no real weights. |

## Phases

| Phase | Status | % | Reality check |
|---|---|---:|---|
| 0. Scoping / infra | Done | 100 | Repo, CI, Docker, packages exist |
| 1. Data pipeline | Done, corrected | 100 | 5,292 HEMIT pairs; 512px legacy data has wrong target; 1024 builder ready |
| 2. H&E → DAPI model | In progress | 50 | panCK ablation done; corrected DAPI 512 pretrain running; dual-branch not implemented |
| 3. App / API | Alpha | 60 | FastAPI + React + Docker; model behind it is the ablation |
| 4. Validation | Partial | 15 | Full 630/945 panCK ablation eval done; no DAPI evaluation yet |
| 5. Deployment | Alpha | 55 | GHCR staging, PyPI/HF alpha; corrected weights and CPU benchmark pending |

Overall is intentionally not presented as a single impressive percentage: the
only completed model is the panCK ablation; DAPI training is still in progress.

## Next actions

1. Run 20-epoch 512 pretrain on Kaggle, resuming across sessions.
2. Fine-tune 2×500 steps at 1024 from the 512 checkpoint.
3. Evaluate full 630 val + 945 test; publish DAPI metrics with the target channel.
4. Implement/vendor the dual-branch Swin-T + FMF generator.
5. Train reverse DAPI → H&E, then add a direction router for one plugin/API.
6. Benchmark 512px tiled/INT8 deployment on the 3.7 GB box.

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
