# Data Extraction Report - 2026-08-31
> Auth: Hugging Face token was used for gated downloads (since revoked/rotated — never commit tokens; use `HF_TOKEN` env var).

## Sources (WORKPLAN Phase 1)
| Dataset | Source | Raw Location | Processed | Count | Patch | Size |
|---------|--------|--------------|-----------|-------|-------|------|
| Paired H&E <-> DAPI | maxpmx/dataset_dapi-he (HF) | data/raw/maxpmx_dapi_he/images.zip (1.5G) + conditioning_images.zip (353M) | data/processed/maxpmx_{train,val,test,512}.h5 | 4192 pairs (3353/419/420 split 80/10/10) | 512x512 RGB + L | 2.2G train, 278M val/test, 2.8G combined |
| Unpaired H&E | NYCU-PCSxNTHU-MIS/MHIST (HF, 666M tar.gz) | data/raw/mhist/mhist.tar.gz -> data/raw/mhist/MHIST/images/*.png (3152) | data/processed/mhist_{256,512}.h5 | 3152 HP/SSA | 256/512 | 498M / 1.9G |

- HEMIT (MICCAI 2024, 5,292 pairs: 3,717/630/945) not on HF (404) - replaced by maxpmx/dataset_dapi-he for the initial MVP (4,192 pairs). The official HEMIT data is now extracted locally and is the primary source for corrected DAPI work.

## Extraction Pipeline
- HF auth: `HF_TOKEN` env var via `huggingface_hub` + `curl -H "Authorization: Bearer $HF_TOKEN"`
- Download: `curl -L -H Authorization: Bearer $HF_TOKEN https://huggingface.co/datasets/maxpmx/dataset_dapi-he/resolve/main/images.zip` etc. (1.5G + 353M + 666M = 2.5G raw)
- Extraction: `zipfile.ZipFile` / `tar -xzf` to `data/raw/.../images/`, verified 4192+4192 pairs, all train.jsonl entries exist (0 missing)
- Processing: `PIL.Image` resize to 512 (or 256), `h5py` chunked lzf compression to `data/processed/*.h5` with attrs `patch_size`, `num_pairs`, `source`
- Verification: `resources/datasets/verify_alignment.py` -> train HE mean 176.4 DAPI 21.2, val HE 227.2 DAPI 4.8 -> OK, no empty DAPI
- Examples: `data/examples/he_example.png` (512 RGB) + `dapi_example.png` (512 L) copied from first pair for drag-drop UI

## Modules Used
- `src/utils/stain_norm.py:7` MacenkoNormalizer (fit on HE example, transform test HE mean 126.9 -> 28.7)
- `src/utils/tiling.py:18` TiledInference (512/256 gaussian blending validated on 128x128 merge)
- `src/utils/wsi_loader.py:13` WSILoader (PIL backend for png, openslide for SVS)
- `src/models/datasets.py:22` PairedHEMITDataset / UnpairedDataset (h5 loading, torch-optional)
- `src/models/infer.py:19` patch + WSI tiled inference (dummy fallback works, torch pending)

## Next Steps (Phase 2 Modeling)
```bash
# Install torch (needs GPU, ~2GB)
/tmp/he-venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Paired training (H&E->DAPI)
python src/models/train_paired.py --config configs/hemit_dual_branch.yaml
# But use maxpmx data: update config data.processed to data/processed/maxpmx_train_512.h5
# Unpaired training (DAPI->H&E)
python src/models/train_unpaired.py --config configs/cyclegan_dapi2he.yaml --he-root data/raw/mhist/MHIST/images --dapi-root data/raw/maxpmx_dapi_he/conditioning_images
# Inference demo
python src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/out.png
# Backend
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

## Disk Usage
- data/raw 5.0G, data/processed 7.8G, total 12.8G (fits in 953G avail)

Generated: 2026-08-31 via /tmp/he-venv
