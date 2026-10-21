# Kaggle DAPI runbook and status

Last checked: **2026-10-21 12:45 UTC**

## Current status

- **No corrected DAPI training job exists on Kaggle yet.**
- Corrected `hemit_*_1024.h5` is built and verified locally:
  - train 11.5 GB, 3,717 pairs
  - val 2.1 GB, 630 pairs
  - test 3.0 GB, 945 pairs
- Private data datasets:
  - `bikasash/hemit-dapi-1024-ch0-macenko` — train + val ready
  - `bikasash/hemit-dapi-1024-ch0-test` — test split (separate, processing)
- Code dataset `bikasash/he-dapi-code-dapi-1024` is ready.
- Smoke kernel `bikasash/he-dapi-dapi-1024-corrected` passed:
  - T4, torch 2.11.0+cu128
  - 20 steps in ~39s at 512px, batch 2: **~1.96 s/step, ~1 sample/s**
  - ~61 min/epoch at 3,717 samples
- Revised target: 20 epochs at 512 (~9h), then 2×500 steps at 1024 (~2h).
- Test H5 is not attached; attach `hemit-dapi-1024-ch0-test` for final test eval.
- GPU quota: **~29.9h remaining**, refresh `2026-10-24T00:00:00`.
- Pretrain launched as kernel **v5** with `train_dapi_pretrain.py`; checkpoints save every epoch.

## Why the run is staged

A 100-epoch, native-1024px run is not realistic on a 30h T4 quota. Plan:

1. **Smoke** — done: ~1.96 s/step at 512, batch 2.
2. **Pretrain** — 20 epochs at 512; likely 12h + resume session.
3. **Fine-tune** — 2×500 steps at native 1024 from the 512 checkpoint.
4. **Evaluate** — full 630 val + 945 test at 1024; save labeled metrics.

If the smoke run is too slow, stop and reduce the stage targets rather than burn
quota silently.

## 1. Build corrected HDF5 (after local epoch 200)

```bash
cd /home/bikash/he-dapi-translator
PYTHONPATH=. .venv/bin/python -m scripts.data.pack_hemit_1024 \
  --splits train --compression gzip --compression-level 4
PYTHONPATH=. .venv/bin/python -m scripts.data.pack_hemit_1024 \
  --splits val --compression gzip --compression-level 4
PYTHONPATH=. .venv/bin/python -m scripts.data.pack_hemit_1024 \
  --splits test --compression gzip --compression-level 4
```

Verify channel integrity before upload:

```bash
PYTHONPATH=. .venv/bin/python - <<'PY'
import h5py, numpy as np
for split in ("train", "val", "test"):
    p = f"data/processed/hemit_{split}_1024.h5"
    with h5py.File(p, "r") as f:
        assert f.attrs["dapi_channel"] == 0
        assert f.attrs["patch_size"] == 1024
        i = len(f["dapi"]) // 2
        assert np.array_equal(f["dapi"][i], f["mihc"][i, ..., 0])
        print(split, f["he"].shape, f["dapi"].shape, f.attrs["macenko"])
PY
```

## 2. Upload private Kaggle dataset

```bash
mkdir -p /tmp/kaggle_dapi_1024
cp data/processed/hemit_train_1024.h5 /tmp/kaggle_dapi_1024/
cp data/processed/hemit_val_1024.h5 /tmp/kaggle_dapi_1024/
cp data/processed/hemit_test_1024.h5 /tmp/kaggle_dapi_1024/
cat > /tmp/kaggle_dapi_1024/dataset-metadata.json <<'JSON'
{
  "title": "HEMIT DAPI ch0 1024 Macenko",
  "id": "bikasash/hemit-dapi-1024-ch0-macenko",
  "licenses": [{"name": "other"}],
  "isPrivate": true
}
JSON
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/api_token)
.venv/bin/python -m kaggle datasets create -p /tmp/kaggle_dapi_1024 --dir-mode tar
```

## 3. Upload code dataset

After committing the DAPI configs/scripts:

```bash
mkdir -p /tmp/he-dapi-code
git archive --format=zip --output=/tmp/he-dapi-code/code_bundle.zip HEAD
cp scripts/kaggle/train_dapi_1024.py /tmp/he-dapi-code/
cat > /tmp/he-dapi-code/dataset-metadata.json <<'JSON'
{
  "title": "he-dapi code dapi 1024",
  "id": "bikasash/he-dapi-code-dapi-1024",
  "licenses": [{"name": "other"}],
  "isPrivate": true
}
JSON
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/api_token)
.venv/bin/python -m kaggle datasets create -p /tmp/he-dapi-code --dir-mode tar
```

## 4. Run stages

Attach both private datasets to a **GPU T4 x2** Kaggle script. Use only one GPU.

```bash
# Smoke: 1 epoch, 20 steps
DAPI_STAGE=smoke python /kaggle/input/he-dapi-code-dapi-1024/train_dapi_1024.py

# 40-epoch 512 pretrain
DAPI_STAGE=pretrain python /kaggle/input/he-dapi-code-dapi-1024/train_dapi_1024.py

# 20-epoch 1024 fine-tune
DAPI_STAGE=finetune python /kaggle/input/he-dapi-code-dapi-1024/train_dapi_1024.py
```

Checkpoints go to `/kaggle/working/checkpoints/`. After each session, version
them into a private checkpoint dataset so the next session can resume.

## 5. Evaluation

```bash
PYTHONPATH=. .venv/bin/python -m src.models.eval_offline \
  --checkpoint resources/models/checkpoints/he2dapi_dapi_1024/generator_last.pt \
  --h5 data/processed/hemit_val_1024.h5 --split val --num 630 \
  --patch-size 1024 --target "DAPI ch0" --label dapi-1024-unet \
  --out docs/metrics_dapi_1024_val.json

PYTHONPATH=. .venv/bin/python -m src.models.eval_offline \
  --checkpoint resources/models/checkpoints/he2dapi_dapi_1024/generator_last.pt \
  --h5 data/processed/hemit_test_1024.h5 --split test --num 945 \
  --patch-size 1024 --target "DAPI ch0" --label dapi-1024-unet \
  --out docs/metrics_dapi_1024_test.json
```

Note: `generator_last.pt` is a raw state dict, while `eval_offline.py` expects a
training checkpoint with a `"G"` key. Use `checkpoint_last.pt` or add a
state-dict fallback before evaluating.

## Tracking table

| Stage | Status | Notes |
|---|---|---|
| Local panCK epoch 200 | done | full 630/945 len eval recorded |
| Build 1024 H5 | done | train 11.5 GB, val 2.1 GB, test 3.0 GB |
| Upload data dataset | ready (train/val) | test split in separate dataset |
| Upload code dataset | ready | nested `code_bundle/` mount handled |
| Kaggle smoke | **done** | ~1.96 s/step at 512, batch 2 |
| 512 pretrain | **running** | kernel v5; epoch 10/20 at 12:43 UTC; checkpoints saving |
| 1024 fine-tune | pending | 2×500 steps |
| Full val/test eval | pending | 630 / 945 |
