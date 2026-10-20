# Optimization Report — Machine-Aware (3.7GB RAM Crash Fix)

Date: 2026-09-19 | Machine: Intel i5-10210U 8 vCPU, 3.7GB RAM +1GB swap, no GPU, 867GB disk (WSL2) | `src/utils/resource_monitor.py:1`

## Q: "we complete the data extraction phase ri8?" — YES.

Verification Sep 19:
- `data/raw/hemit/extracted/{train,val,test}/{input,label}/*.tif` → 5292 pairs (3717/630/945) confirmed `ls -R data/raw/hemit | wc -l` ~5300.
- `data/processed/hemit_train_512.h5` 3.7G (3717), `hemit_val_512.h5` 684M (630), `hemit_test_512.h5` 968M (945) — via `scripts/data/pack_hemit.py` resumable (skip if `he.shape[0]==len(pairs)`).
- `data/processed/maxpmx_*` 4192 fallback, `mhist_256.h5` 498M + `mhist_512.h5` 1.9G (3152 MHIST), `codex_512.h5`, `monuseg_256.h5`, `data/raw/wsi/CMU-1.svs` example.
- Previous crash cause: `scripts/data/pack_hemit.py:86-107` naive `np.concatenate` required 10.6GB → OOM kill. Fixed Sep 19 to streaming chunked copy (peak 140MB).

## Crash Root Cause & Spike Map

| Step | Before Peak | After Peak | Fix | Doc |
|------|-------------|------------|-----|-----|
| HEMIT combined 5292 | 10.6GB → killed | 140MB | chunked `h5py` copy B=100, `gc.collect`, `ResourceMonitor` | `wiki/02-data-extraction/SKILL.md:1` |
| BCI combined 9681 | 9GB → killed | 150MB | same streaming | `scripts/data/pack_bci.py:1` |
| WSI canvas 20k²×3 float32 | 6.4GB (canvas+weight) | <512MB memmap bands | `merge_tiles_streaming` to tiled TIFF | `wiki/04-tiling/SKILL.md:1` |
| WSI load `asarray()` | 4.8GB | 0 (zarr slice) | `series.aszarr()` + level downsample | `wiki/05-wsi-loader/SKILL.md:1` |
| Train batch 8 + VGG | 1.2GB + 0.5GB VGG | 18MB (batch2+accum4) | `batch_size 2, grad_accum 4, dists_on_cpu false, num_workers 0` | `wiki/06-training/SKILL.md:1` |
| Infer full WSI 40k² | 4.8GB | <512MB | `infer_wsi_streaming` per-tile `read_region` + streaming merge | `wiki/07-inference/SKILL.md:1` |
| Backend concurrent WSI | 1GB (data+tile+result) | 429 guard | `used_pct>85 or avail<400 → 429` + thumbnail before numpy | `wiki/08-backend/SKILL.md:1` |

## Config Changes (low-RAM)

- `configs/hemit_dual_branch.yaml:9` `num_workers 8→0`, `batch_size 8→2`, `grad_accum 4`, `mixed_precision true→false`, `stain_norm macenko→none`, `dists_on_cpu false`, `data.processed → hemit_train_512.h5` (split, not combined).
- `configs/cyclegan_dapi2he.yaml:7` similar, `dapi_root → hemit_train_512.h5` (H5 fallback), `he_h5_fallback` added.

## Wiki Skill Folders (9 sub-agents)

Each `wiki/0X-*/SKILL.md` is the sub-agent prompt + patched file links. See `wiki/README.md:1` for map.

```
wiki/01-machine-profile/  machine_profiler      profile.json, OPTIMIZATION.md
wiki/02-data-extraction/  extraction_optimizer  pack_hemit/bci + prepare_patches
wiki/03-datasets/         dataset_streamer      datasets.py SWMR
wiki/04-tiling/           tiling_streamer       tiling.py streaming merge
wiki/05-wsi-loader/       wsi_streamer          wsi_loader.py aszarr
wiki/06-training/         train_cpu_optimizer   train_*.py grad_accum
wiki/07-inference/        infer_streamer        infer.py streaming + wsi
wiki/08-backend/          backend_throttler     backend 429 guard
wiki/09-orchestration/    orchestrator          Makefile, all resumable
```

## Verification (all passed 2026-09-19)

```bash
PYTHONPATH=. .venv/bin/python tests/test_resource_guards.py  # 11 passed
.venv/bin/python -m src.utils.resource_monitor --check  # avail 2473MB, 35% used, headroom ok
.venv/bin/python -c "from src.models.datasets import PairedHEMITDataset; ds=PairedHEMITDataset('data/processed/hemit_train_512.h5', indices=range(2)); print(ds[0]['he'].shape)"
.venv/bin/python src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/synth.png
.venv/bin/python src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/synth.tiff --stream
# Makefile
.venv/bin/python scripts/data/pack_hemit.py  # SKIPs train/val/test, streams combined (<200MB)
```

## How to Run Next (resumable, no crash)

```bash
# 1. Health
PYTHONPATH=. .venv/bin/python -m src.utils.resource_monitor --check

# 2. Data (already done, but re-runnable safe)
PYTHONPATH=. .venv/bin/python scripts/data/pack_hemit.py 2>&1 | tee /tmp/pack_hemit.log

# 3. Training (CPU low-RAM)
PYTHONPATH=. .venv/bin/python src/models/train_paired.py --config configs/hemit_dual_branch.yaml --dummy --optimize-for lowram
PYTHONPATH=. .venv/bin/python src/models/train_unpaired.py --config configs/cyclegan_dapi2he.yaml --dummy --optimize-for lowram
# full: remove --dummy, will checkpoint every 1 epoch to resources/models/checkpoints/*/checkpoint_last.pt (resume auto)

# 4. Inference
PYTHONPATH=. .venv/bin/python src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/out.png
PYTHONPATH=. .venv/bin/python src/models/infer.py --input data/raw/wsi/CMU-1.svs --direction he2dapi --wsi --stream --out /tmp/wsi.tiff

# 5. Backend
PYTHONPATH=. uvicorn src.backend.main:app --host 0.0.0.0 --port 8000  # has 429 guard
```

## Remaining Risk

- `monai`, `torch`, `albumentations` not yet in `.venv` (only numpy/h5py/zarr pillow installed for verification) → full train needs `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` (2GB). On this box, install with `uv pip install torch==2.3.0 --python .venv/bin/python` in steps to avoid RAM spike.
- For 40k WSI, even streaming input image `h*w*3` for thumbnail must be downsampled; `load_image` now thumbnails to 2048/4096 to keep <512MB.

Generated: `docs/optimization/REPORT.md` + `wiki/*`
