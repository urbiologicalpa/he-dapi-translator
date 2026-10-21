# Workplan: H&E <-> DAPI Drag & Drop Translator
Version: 1.1 | Date: 2026-10-20 | Author: OpenCode + User
Location: `/home/bikash/he-dapi-translator`

## 0. Goal
Build a non-on-premise (cloud-ready, Dockerized) tool where user drags H&E image -> synthetic DAPI and vice versa. Must handle BOTH:
- Single patch (256x256 / 512x512 / 1024x1024) for quick demo
- Whole Slide Image (WSI, >1B pixels, OME-TIFF / SVS) via tiled inference

And BOTH user types: researchers + pathologists.

## 1. Phasing (12 weeks MVP)

### Phase 0 — Scoping & Infra (Week 1-2)
- [ ] Lock spec: inputs: H&E RGB, DAPI grayscale+metadata; outputs: synthetic counterpart + uncertainty heatmap + nuclei count
- [ ] Create repo structure (DONE in this scaffold) + Git init
- [ ] Decide stack:
  - Frontend: React + react-dropzone + OpenSeadragon for WSI viewer + slider compare
  - Backend: FastAPI + Python 3.10 + PyTorch + ONNX Runtime + openslide / tifffile + large_image
  - Training: PyTorch Lighting + MONAI
  - Deployment: Docker -> Docker Hub -> HuggingFace Space mirror
- [ ] Setup experiment tracking: MLflow or Weights & Biases
- [ ] Legal: non-on-premise but NO own data -> use only public datasets, add disclaimer "synthetic, not diagnostic"

### Phase 1 — Data (Week 2-4) — CRITICAL PATH
**Since no own data, source internet:**

Priority datasets (see `resources/datasets/README.md`):
1. **HEMIT (MICCAI 2024)** - 5,292 pairs (3,717/630/945), SAME slide, cellular-level aligned, H&E <-> DAPI/CD3/panCK, 1024x1024. BEST for paired training. Contact: ImmunoAIzer derived. Use central 512x512 crop only for the legacy CPU baseline; corrected DAPI training uses native 1024.
2. **BCI (Breast HER2)** - consecutive sections, not aligned -> for unpaired test.
3. **CODEX Virtual H&E examples (PMC8721868)** - DAPI + Eosin -> virtual H&E, has code + example data.
4. **Autofluorescence -> H&E Lung (medRxiv 2024)** - mIF panel includes DAPI.
5. Optional: Light-sheet C01/C02 dataset (2026 CycleGAN paper) - 256 fluorescence + MHIST 3152 H&E target.

Actions:
- [ ] Download HEMIT, create paired patch loader (512x512, stain norm Macenko)
- [ ] Build unpaired dataset: collect separate H&E (MHIST, TCGA) and DAPI (from HEMIT DAPI channel, plus nuclei datasets like MoNuSeg)
- [ ] Create `data/raw` download scripts + `data/processed` tiled HDF5/Zarr
- [ ] Add WSI example: 1 SVS from GDC + 1 OME-TIFF DAPI example
- [ ] Augmentation: flipping, affine, color jitter, stain augmentation, Gaussian blur for tone conditioning

### Phase 2 — Modeling (Week 4-8) — Dual Track
Train BOTH directions as separate models (not one bijective model):

**Track A: H&E -> DAPI (easier, segmentation task)**
- Baseline 1: Pix2Pix (U-Net generator + PatchGAN) -> L1 + cGAN + SSIM + DISTS loss (per 2024 FLIM paper)
- Baseline 2: Dual-branch Pix2Pix (CNN + Swin Transformer + FMF) from HEMIT paper -> DAPI-branch paper targets 0.815 SSIM / 0.716 Pearson / 28.610 dB PSNR (not composite mIHC numbers)
- Stretch: D-VST Diffusion Transformer (NeurIPS 2025) - tone-controllable, pathology-correct

**Track B: DAPI -> H&E (harder, generative)**
- Baseline 1: CycleGAN (ResNet generator) + identity loss -> unpaired, needs no alignment
- Baseline 2: CUT / SACSI-CUT (self-attention + cellular intensity consistency) from 2025 DAPI+Eosin paper -> better hallucination control
- Stretch: Conditional Diffusion (Palette/BBDM) + pathology encoder

Shared Tricks (from Nature Comms 2026 DGR):
- Decoupled Generation & Registration (DGR) -> robust to misalignment, +23.8% PSNR when misaligned
- Auxiliary nuclei segmentation loss: run pretrained StarDist/Cellpose on real vs fake DAPI, add Dice loss
- Stain normalization (Macenko/Reinhard) before inference

Deliverables:
- [ ] `src/models/train_paired.py` (Pix2Pix + DISTS)
- [ ] `src/models/train_unpaired.py` (CycleGAN/CUT)
- [ ] `src/models/infer.py` with tiled WSI inference (frequency-aware adaptive patch sampling from D-VST)
- [ ] Export to ONNX for docker

Metrics (see `resources/models/README.md`):
- Pixel: SSIM, PSNR, LPIPS, DISTS, FID/KID
- Bio: Dice nuclei segmentation, cell density correlation, MSE per channel (DAPI 0.12 vs 0.26 CycleGAN in mSIGHT)
- Human: pathologist blind test (52% distinguishability in DGR paper = indistinguishable)

### Phase 2.5 — Benchmark & release gates (after epoch 200)

**What ships after the 200-epoch run:** an **internal ablation benchmark**, not a
DAPI model and not a tagged product release. The run is H&E -> panCK because of
the old channel bug. Shipping it as DAPI would be false.

Deliverables:

- [x] Archive `checkpoint_epoch200.pt`, `generator_last.pt`, config, git SHA and
      checkpoint hash in `docs/ABLATION_PANCK.md`.
- [x] Run full panCK evaluation on **630 val + 945 test** using the legacy 512px
      H5; write `docs/metrics_panck_512.json`.
- [x] Fill `docs/BENCHMARK.md` with measured panCK numbers, clearly labeled.
- [x] Keep weights internal; publish only the engineering note and metrics in
      git. Do not tag a model release or publish a DAPI claim.
- [x] Keep PyPI/HF weights labeled as the ablation until corrected DAPI weights
      exist; no new DAPI package release from this run.
- [x] Production approval stays manual per `AGENTS.md`; do not auto-approve.

**Release gates for the eventual DAPI model (v0.2.0):**

- [ ] Corrected 1024px H5 verified (`dapi == mihc[...,0]`).
- [ ] Full 630 val + 945 test metrics recorded with target channel.
- [ ] Checkpoint manifest includes target, resolution, split, git SHA.
- [ ] Demo/package/HF/Docker all load the same corrected checkpoint.
- [ ] No panCK metrics or ONNX attached to the DAPI release.

### Phase 3 — Drag & Drop Tool / Plugin (Week 8-10)
**Frontend (React):**
- [ ] `src/frontend`: Dropzone (supports .png .tiff .svs), preview grid, WSI viewer (OpenSeadragon)
- [ ] Controls: Direction toggle H&E->DAPI / DAPI->H&E, model selector (fast vs quality), confidence overlay toggle
- [ ] Result: side-by-side slider, download TIFF, share link, watermark "SYNTHETIC"

**Backend (FastAPI):**
- [ ] `src/backend/main.py`: `POST /translate`, `POST /translate/wsi`, `GET /health`, `GET /models`
- [ ] Queue for WSI (Celery or background tasks), return tile progress via websocket
- [ ] Auto QC: check input H&E distribution vs training distribution (reject out-of-distribution), return confidence heatmap from discriminator

**Plugin variants (stretch, after web MVP):**
- [x] napari plugin: `napari-he-dapi` scaffolded in `packages/napari-he-dapi` (npe2 manifest + magicgui widget reusing `he-dapi`)
- [x] PyPI package: `he-dapi` (library + CLI) in `packages/he-dapi`; publish flow in `.github/workflows/publish-pypi.yml` + `docs/PUBLISHING.md`
- [ ] QuPath extension (Groovy) + ImageJ macro

### Phase 4 — Validation & Hardening (Week 10-11)
- [ ] Test on hold-out patients (never seen tissue), cross-tissue (breast -> colon) failure analysis
- [ ] Measure: DAPI nuclei count error <10%, H&E hallucination audit (no invented tumor)
- [ ] Add uncertainty: Monte Carlo dropout 5 passes -> variance map
- [ ] Add content filter + disclaimer for clinical misuse

### Phase 5 — Docker & Deployment (Week 11-12) — STAGED (CI -> Build -> Staging -> Production)
Standard ML deploy stages (Databricks MLOps: Data Validation -> Training -> Validation -> Registration -> Deployment -> Monitoring; deploy-code Challenger/Champion; Microsoft 2026 one-SHA-promoted-immutably; GitHub Environments with branch policies + reviewers):

| Stage | Trigger | Gate | Output |
|-------|---------|------|--------|
| 0 CI (`ci.yml`) | push `main`/`staging` + PR | `progress_tracker --enforce` + `test_resource_guards` (11) + backend `/health` smoke | green check |
| 1 Build (`cd-staging.yml`) | push `staging` | Docker build once, tag = git SHA | `ghcr.io/...:<sha>` + `:staging` |
| 2 Staging (`cd-staging.yml`, env `staging`) | auto after build | guards + `/health` + `/models` + dummy `/translate` smoke; human UAT (512px both directions, SYNTHETIC header, 429 guard, `/translate/wsi`) | UAT sign-off |
| 3 Production (`cd-production.yml`, env `production`) | `git tag vX.Y.Z` on `main` / Release published; needs staging SHA success + reviewer approval | promote SAME SHA `:staging -> :vX.Y.Z + :latest` (no rebuild); optional Docker Hub mirror; attach status docs only until corrected DAPI metrics/ONNX exist | prod release; rollback = re-tag previous `v*` |

- [ ] Dockerfile (see `docker/Dockerfile` scaffold): multi-stage, python slim + node build + cuda runtime + OCI labels
- [ ] docker-compose for local: frontend:3000, backend:8000, mlflow:5000
- [ ] Push to GHCR staging: auto on `staging` branch (`:sha` + `:staging`); prod promote on tag (`:vX.Y.Z` + `:latest`)
- [ ] Push to Docker Hub mirror: `docker push <user>/he-dapi-translator:latest` (optional, needs `DOCKERHUB_TOKEN` secret)
- [ ] Deploy: Hugging Face Space staging/prod (Gradio fallback, Zero-GPU like `timbrooks/instruct-pix2pix`, `biohub/dynacell` virtual-staining demo) + optional Modal/RunPod GPU endpoint (set `PROD_URL` repo variable for live health poll)
- [ ] CI: GitHub Actions -> auto build on push (done: `ci.yml` guards; `cd-staging.yml`/`cd-production.yml` added)
- [ ] Environments: `staging` (branch policy `staging`, no approval) + `production` (branch policy `main`/tags, required reviewers) — create via `gh api repos/{owner}/{repo}/environments/staging -X PUT` once `GITHUB_TOKEN` exported
- [ ] Branch flow: `feature/*` -> PR -> `staging` (UAT) -> PR -> `main` (prod via tag). Create: `git branch staging && git push -u origin staging` (needs token)

Refs: `junyanz/pytorch-CycleGAN-and-pix2pix` docker pattern (`taesungp` prebuilt + `nvidia-docker run`, `docs/docker.md`); HF Spaces `timbrooks/instruct-pix2pix` (1.5k likes, Zero GPU) + `biohub/dynacell` (staining demo); GitHub Environments docs (approval, branch policy, concurrency 1); Databricks ML CI/CD (version data+code+model, Challenger/Champion gates).

## 2. Novelty & Positioning
See `docs/NOVELTY_REPORT.md` (summary: H&E->DAPI well-published since 2024, but bidirectional drag-drop WSI tool + Docker product is still gap). Our USP:
- Bidirectional in one UI (most papers one-way)
- WSI + patch unified
- DGR misalignment-robust training + uncertainty map
- One-click Docker Hub, not just code dump

## 3. Risks & Revisions (Critique)
| Risk | Revision |
|------|----------|
| DAPI->H&E hallucinates cytoplasm | Label output "visualization only", add eosin-like prior from autofluorescence when available, limit to ROI |
| WSI OOM | Tiled inference 512px stride 256, frequency-aware sampling, stream output |
| Stain domain shift | Macenko norm + test-time augmentation + OOD detector |
| No paired data for DAPI->H&E | Use unpaired CUT + synthetic pairs via HEMIT reverse |
| Legal/ethical | Add watermark + terms, not for diagnosis |

## 4. Resources to Keep in `resources/`
- Papers PDFs + notes
- Dataset download scripts
- Model checkpoints (git-lfs or hf hub)

## 5. Immediate Next Steps (2026-10-20)
1. Wait for the panCK epoch-200 checkpoint; archive it as an ablation.
2. Run full `val`/`test` panCK evaluation into `docs/metrics_panck_512.json`
   using `src/models/eval_offline.py --out ... --label ... --target ...`.
3. Publish the internal ablation note and metrics in git (no release tag).
4. Build corrected 1024px HDF5 with `scripts/data/pack_hemit_1024.py`.
5. Upload to Kaggle and train the corrected U-Net DAPI baseline.

## 6. Timeline Gantt
W1-2 Infra+Data | W2-4 Data | W4-8 Modeling | W8-10 App | W10-11 Valid | W11-12 Docker

---
Saved: `/home/bikash/he-dapi-translator/docs/WORKPLAN.md`
