# Progress tracker — 2026-10-20

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
- Corrected 1024px HEMIT H5 has not been built yet.

## Live runs

| Run | Status | Details |
|---|---|---|
| Local CPU, 512px | Finishing panCK ablation | Epoch 199 saved; epoch 200 running at 17:09 UTC; ETA ~19:00 UTC 2026-10-20. Do not treat as DAPI. |
| Kaggle T4 | Available, idle | 2× Tesla T4 15.64 GB, torch 2.11.0+cu128. 0.01h used, 29.99h remaining; resets 2026-10-24. |
| Corrected 1024px DAPI | Blocked | Waiting for the ablation to finish, then build H5 and upload to Kaggle. |
| DAPI → H&E | Not started | Trainer is a smoke-test skeleton; no real weights. |

## Phases

| Phase | Status | % | Reality check |
|---|---|---:|---|
| 0. Scoping / infra | Done | 100 | Repo, CI, Docker, packages exist |
| 1. Data pipeline | Done, corrected | 100 | 5,292 HEMIT pairs; 512px legacy data has wrong target; 1024 builder ready |
| 2. H&E → DAPI model | In progress | 45 | panCK ablation done; corrected DAPI baseline not started; dual-branch not implemented |
| 3. App / API | Alpha | 60 | FastAPI + React + Docker; model behind it is the ablation |
| 4. Validation | Not started | 5 | No full 630/945 DAPI evaluation; only 30-image ablation metrics |
| 5. Deployment | Alpha | 55 | GHCR staging, PyPI/HF alpha; corrected weights and CPU benchmark pending |

Overall is intentionally not presented as a single impressive percentage: the
only trained model is the wrong target, so training progress is not DAPI
progress.

## Next actions

1. Let the panCK epoch 200 finish; archive it as an ablation.
2. Build corrected 1024px H5 with gzip (~10.4 GB total).
3. Upload to a private Kaggle dataset.
4. Run a 1-epoch 1024px U-Net sanity run on one T4 (batch 2), then train the
   corrected DAPI baseline with checkpoint/resume.
5. Evaluate full 630 val + 945 test; publish metrics with the target channel.
6. Implement/vendor the dual-branch Swin-T + FMF generator.
7. Plan reverse DAPI → H&E with honest "predicted H&E-like" labeling.
8. Benchmark 512px tiled/INT8 deployment on the 3.7 GB box.

## Commands

```bash
# status
PYTHONPATH=. python -m src.utils.progress_tracker --check
PYTHONPATH=. python -m src.utils.progress_tracker --enforce
PYTHONPATH=. python -m pytest tests/test_hemit_channels.py tests/test_datasets.py -q

# corrected data
uv run --extra data python -m scripts.data.pack_hemit_1024 \
  --splits train val test --patch-size 1024 --compression gzip

# corrected training (on GPU, after data upload)
PYTHONPATH=. python -m src.models.train_paired \
  --config configs/hemit_dual_branch.yaml --resume auto --optimize-for full
```

## Milestones kept

- 2026-08-31: initial HEMIT/maxpmx/MHIST extraction.
- 2026-09-19: low-RAM streaming pass; 11 resource guards.
- 2026-10-18: v0.1.0 GHCR release, HF demo, PyPI alpha packages.
- 2026-10-20: DAPI channel-order correction; Kaggle T4 availability confirmed;
  README/docs rewritten to match what is actually implemented.
