# H&E → DAPI Wiki — Machine-Aware Optimization (3.7GB RAM, CPU-only, WSL2)

> Generated 2026-09-19. Machine: Intel i5-10210U 8 vCPU, 3.7GB RAM + 1GB swap, no GPU, 867GB disk free.

## Philosophy: Spike Guard
Every pipeline step is wrapped with `ResourceMonitor` (`src/utils/resource_monitor.py:1`). If RAM > 80% or spike Δ > 500MB in <5s → throttle/batch-halve/stream-to-disk.

## Map (each folder = sub-agent skill)

| # | Folder | Skill | Spike Risk | Fix |
|---|--------|-------|------------|-----|
| 01 | `wiki/01-machine-profile/` | `machine_profiler` | baseline | CPU-only, RAM ceiling 2.6G avail |
| 02 | `wiki/02-data-extraction/` | `extraction_optimizer` | 10GB OOM on `np.concatenate` | streaming HDF5 copy, no concat |
| 03 | `wiki/03-datasets/` | `dataset_streamer` | `h5py.File` leak, preload | lazy per-worker, LRU, SWMR |
| 04 | `wiki/04-tiling/` | `tiling_streamer` | 4.8GB canvas float32 | chunked merge to zarr/tiff |
| 05 | `wiki/05-wsi-loader/` | `wsi_streamer` | `asarray()` loads 10GB WSI | tile-window reading |
| 06 | `wiki/06-training/` | `train_cpu_optimizer` | batch 8 OOM, VGG 400MB | batch 2 + grad-accum 4, no VGG on CPU |
| 07 | `wiki/07-inference/` | `infer_streamer` | full-image in RAM | tiled streaming to disk |
| 08 | `wiki/08-backend/` | `backend_throttler` | concurrent WSI uploads | queue + mem guard |
| 09 | `wiki/09-orchestration/` | `orchestrator` | multi-step crash | Makefile + resumable + checkpoints |

## How to use (skill invoke)

Each folder contains `SKILL.md` (agent prompt), `OPTIMIZATION.md` (before/after + benchmarks), and links to patched code.

```bash
# run spike-guarded pipeline
python -m src.utils.resource_monitor --check  # prints budget
make data  # resumable, stream-mode
python src/models/train_paired.py --config configs/hemit_dual_branch.yaml --optimize-for lowram
python src/models/infer.py --input data/raw/wsi/*.svs --stream --out /tmp/out.tiff
```

## Verification

`python tests/test_resource_guards.py` — checks every guard active.

## Data Extraction Status (you asked "are we done ri8?")

YES — Phase 1 DONE per `docs/DATA_EXTRACTION_REPORT.md:1` + verified Sep 6:
- `hemit_train_512.h5` 3717 pairs (3.7G), `hemit_val` 630, `hemit_test` 945 — from `data/raw/hemit/extracted` 5292 tifs via `scripts/data/pack_hemit.py` (streaming, resumable)
- `maxpmx_*` 4192 pairs (fallback), `mhist_256/512` 3152 patches, `codex/monuseg` extras
- Total `data/processed` 14G, `data/raw` ~72G (hemit 44G inside .7z)
- Next optimization: `hemit_512.h5` combined (5.3G) was previously OOM via `np.concatenate` — now fixed to chunked `h5py` copy in `scripts/data/pack_hemit.py:86`
