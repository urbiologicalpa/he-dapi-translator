# H&E ↔ DAPI Virtual Staining Translator

Bi-directional virtual staining for histopathology: **H&E (RGB) ↔ DAPI (grayscale)**, for single patches and whole-slide images (WSI). Paired Pix2Pix + unpaired CycleGAN, FastAPI backend, drag-and-drop frontend, Docker deploy.

![stack](https://img.shields.io/badge/PyTorch-2.14%20CPU-orange) [![guards](https://github.com/urbiologicalpa/he-dapi-translator/actions/workflows/ci.yml/badge.svg)](https://github.com/urbiologicalpa/he-dapi-translator/actions) [![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml) [![PyPI](https://img.shields.io/pypi/v/he-dapi?label=pip%20install%20he-dapi)](https://pypi.org/project/he-dapi/) [![napari plugin](https://img.shields.io/badge/napari-he--dapi-blueviolet)](https://pypi.org/project/napari-he-dapi/) ![trained on](https://img.shields.io/badge/trained%20on-3.7GB%20CPU-critical) ![building](https://img.shields.io/badge/status-actively%20building-orange) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](CONTRIBUTING.md)

> 🚧 **Actively building in public** — model training live (**181/200 epochs**, 90.5%), roadmap in [Issues](https://github.com/urbiologicalpa/he-dapi-translator/issues), milestones as [Releases](https://github.com/urbiologicalpa/he-dapi-translator/releases). Contributors welcome — start with [CONTRIBUTING.md](CONTRIBUTING.md).

> **Why this repo stands out:** the product-grade model (3,717 images, 512px, ngf64/9-block generator, 200 epochs) is being trained **on a 3.7GB RAM, CPU-only laptop** — via streaming-by-default engineering (10.6GB → 140MB pack peak, 4.8GB → 512MB WSI budget) with crash-resume safety across reboots. See [the constraint story](#engineering-under-constraints).

| H&E input | DAPI target |
|---|---|
| ![H&E](data/examples/he_example.png) | ![DAPI](data/examples/dapi_example.png) |

## 🔬 Live demo

Try the trained model **in your browser** — no install, inference runs client-side via ONNX Runtime Web:

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Space-blue)](https://huggingface.co/spaces/urbiologicalpa/he-dapi-translator) [![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow)](https://huggingface.co/urbiologicalpa/he-dapi-translator)

- **Space (demo):** https://huggingface.co/spaces/urbiologicalpa/he-dapi-translator
- **Weights (model card):** https://huggingface.co/urbiologicalpa/he-dapi-translator
- **Container:** `ghcr.io/urbiologicalpa/he-dapi-translator:latest` (also `:v0.1.0`)

## 📦 Install / Python API

> ⚠️ **Alpha / building:** the pip packages ship the `he2dapi` direction only;
> `dapi2he` lands with the unpaired model. APIs may change. Not for clinical use.

```bash
pip install he-dapi            # inference library + CLI
pip install napari-he-dapi     # napari plugin (in-viewer translation)
he-dapi translate -i he.png -o dapi.png -d he2dapi
```
```python
from he_dapi import translate
pred = translate("he.png", direction="he2dapi")   # -> uint8 HxW numpy array
```

## 🔨 Build progress (live training — 181/200 epochs)

| Metric | Value | Detail |
|---|---|---|
| Progress | **181/200 epochs (90.5%)** | resumable, per-epoch atomic checkpoints |
| G-loss | **2.2603 → 1.5302 (−32%)**, still falling | best-ep169, no divergence |
| D-loss | **~0.75 stable** | balanced GAN, no collapse/mode-collapse |
| Val PSNR | **29.3 dB** (peak 29.8) | offline eval, N=30 val |
| Nuclei Dice | 0.03 → 0.11 (late gain, still climbing) | StarDist pseudo-labels, thresh 0.1 |
| Pace | ~2.4h/epoch CPU | ETA full 200 ≈ Oct-20 |

Full history: [`docs/TRACKER.md`](docs/TRACKER.md) (human log) + [`docs/progress.json`](docs/progress.json) (machine-readable) — kept in sync after every milestone.

## Architecture

- **H&E→DAPI (paired):** dual-branch Pix2Pix (CNN-ResNet + Swin-Transformer + FMF fusion) + PatchGAN, loss = GAN + 100×L1 + SSIM + nuclei-Dice aux
- **DAPI→H&E (unpaired):** CycleGAN fallback for unaligned sections
- **Data:** HEMIT 5,292 pairs (3717 train / 630 val / 945 test, 512px) + MHIST/BCI for unpaired
- **Inference:** per-tile streaming WSI translation (no full-slide canvas), ONNX export

## Engineering under constraints (3.7GB box, no GPU)

Every stage profiled with `src/utils/resource_monitor.py`, every spike fixed by streaming, every fix locked by a regression guard (`tests/test_resource_guards.py`, 11 green):

| Spike | Fix |
|---|---|
| H5 pack 10.6GB concat OOM | chunked copy B=100, peak 140MB |
| WSI canvas 4.8GB | memmap `merge_tiles_streaming`, 512MB budget |
| VGG DISTS 400MB on CPU | disabled → L1+SSIM fallback |
| 8 workers × RAM duplication | `num_workers 0`, batch 2 + grad-accum 4 (eff. 8) |
| Reboots killing days of work | atomic `checkpoint_last.pt` + `--resume auto` + cron `boot_launch.sh` supervisor |

A Kaggle T4 offload was attempted, measured (CPU-only quota, 2.8h/ep — *slower* than local), and stood down with a written post-mortem instead of sunk cost. Process notes: [`wiki/`](wiki/) (9 optimization tracks), [`docs/optimization/REPORT.md`](docs/optimization/REPORT.md).

## Quick start

```bash
# env (CPU)
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install -r docker/requirements.txt  # or: pip install -e ".[lowram]"

# inference (patch + streaming WSI)
PYTHONPATH=. .venv/bin/python src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/synth.png
PYTHONPATH=. .venv/bin/python src/models/infer.py --input data/raw/wsi/CMU-1.svs --direction he2dapi --wsi --stream --out /tmp/wsi.tiff

# training (resumable — safe to kill and rerun)
PYTHONPATH=. .venv/bin/python -m src.models.train_paired --config configs/hemit_dual_branch.yaml --resume auto --optimize-for lowram

# tracker + guards
PYTHONPATH=. .venv/bin/python -m src.utils.progress_tracker --check
PYTHONPATH=. .venv/bin/python -m src.utils.progress_tracker --enforce
PYTHONPATH=. .venv/bin/python tests/test_resource_guards.py
```

Docker: `docker build -t he-dapi-translator -f docker/Dockerfile .` (see [`docker/`](docker/)). Staging image published to `ghcr.io/urbiologicalpa/he-dapi-translator:staging`.

## Layout

```
configs/        training configs (product + lowram demo)
src/models/     datasets, losses, Pix2Pix + CycleGAN trainers, inference
src/utils/      resource monitor, tiling, WSI loader, stain-norm, progress tracker
src/backend/    FastAPI inference server (429 throttling)
src/frontend/   drag-and-drop UI
pack_hemit.py / pack_bci.py / hemit_extract.py   streaming dataset builders (repo root)
he-dapi-product-gpu-train.py                     Kaggle GPU offload script (stood down, see post-mortem in TRACKER)
resources/      dataset scripts, model cards, papers
docs/           WORKPLAN, TRACKER, progress.json, metrics, reports
wiki/           per-stage optimization notes (the constraint story)
tests/          resource-guard regression tests (mirrored in CI)
.github/workflows/  ci.yml (guards) + cd-staging.yml + cd-production.yml (GHCR)
```

## Roadmap

- [x] Data pipeline (HEMIT/BCI/MHIST, streaming)
- [ ] Paired training to 200/200 (**181/200 now**)
- [ ] Full 630-val + test eval (SSIM/PSNR/Dice gate)
- [ ] Drag-and-drop tool polish (queue + websocket WSI progress)
- [ ] Docker Hub deploy
- [ ] Hugging Face Space live demo

## Author

**Bikash Kumar** — [@urbiologicalpa](https://github.com/urbiologicalpa) · [urbiologicalpaa](https://github.com/urbiologicalpaa) · Python/R/SQL, biotech × data. Open to collaboration and roles in ML / computational biology.
