# Kaggle DAPI runbook and status

Last checked: **2026-10-21 12:45 UTC**

## Current status

- Corrected `hemit_*_1024.h5` is built and verified locally:
  - train 11.5 GB, 3,717 pairs
  - val 2.1 GB, 630 pairs
  - test 3.0 GB, 945 pairs
- Private data datasets:
  - `bikasash/hemit-dapi-1024-ch0-macenko` — train + val
  - `bikasash/hemit-dapi-1024-ch0-test` — separate test split
- Code dataset `bikasash/he-dapi-code-dapi-1024` is ready.
- Smoke kernel `bikasash/he-dapi-dapi-1024-corrected` passed:
  - T4, torch 2.11.0+cu128
  - 20 steps in ~39s at 512px, batch 2: **~1.96 s/step, ~1 sample/s**
- Kernel **v5** is running the 512 pretrain with `train_dapi_pretrain.py`.
  - At `12:43 UTC`: **epoch 10/20**, ~1.7 s/step, **~26.5 min/epoch**
  - Checkpoints save every epoch under `/kaggle/working/checkpoints/dapi_pretrain/`
  - Pretrain ETA: **~17:25 UTC**
- Revised target: 20 epochs at 512 (~9h), then 2×500 steps at 1024 (~2h).
- Test H5 is **not attached** to the pretrain kernel; attach
  `bikasash/hemit-dapi-1024-ch0-test` for final test eval.
- GPU quota at launch: **~29.9h**, refresh `2026-10-24T00:00:00`.

## Why the run is staged

A 100-epoch, native-1024px run is not realistic on a 30h T4 quota. Plan:

1. **Smoke** — done: ~1.96 s/step at 512, batch 2.
2. **Pretrain** — 20 epochs at 512; ~9h at the measured ~26.5 min/epoch.
3. **Fine-tune** — 2×500 steps at native 1024 from the 512 checkpoint.
4. **Evaluate** — full 630 val + 945 test at 1024; save labeled metrics.

Measured throughput is the source of truth. If a later session is materially
slower, stop and revise the stage targets rather than burn quota silently.

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

## 2. Upload private Kaggle datasets

Train + val live in one dataset; test is separate because the combined upload
skipped it.

```bash
mkdir -p /tmp/kaggle_dapi_1024 /tmp/kaggle_dapi_test
cp data/processed/hemit_train_1024.h5 /tmp/kaggle_dapi_1024/
cp data/processed/hemit_val_1024.h5 /tmp/kaggle_dapi_1024/
cat > /tmp/kaggle_dapi_1024/dataset-metadata.json <<'JSON'
{
  "title": "HEMIT DAPI ch0 1024 Macenko",
  "id": "bikasash/hemit-dapi-1024-ch0-macenko",
  "licenses": [{"name": "other"}],
  "isPrivate": true
}
JSON

cp data/processed/hemit_test_1024.h5 /tmp/kaggle_dapi_test/
cat > /tmp/kaggle_dapi_test/dataset-metadata.json <<'JSON'
{
  "title": "HEMIT DAPI ch0 1024 test",
  "id": "bikasash/hemit-dapi-1024-ch0-test",
  "licenses": [{"name": "other"}],
  "isPrivate": true
}
JSON

export KAGGLE_API_TOKEN=$(cat ~/.kaggle/api_token)
.venv/bin/python -m kaggle datasets create -p /tmp/kaggle_dapi_1024 --dir-mode tar
.venv/bin/python -m kaggle datasets create -p /tmp/kaggle_dapi_test --dir-mode tar
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

# after later code changes:
.venv/bin/python -m kaggle datasets version -p /tmp/he-dapi-code -r skip -m "describe change"
```

Kaggle auto-extracts `code_bundle.zip`; the runner searches for
`src/models/train_paired.py` and handles the nested `code_bundle/` layout.

## 4. Run stages

Attach both data/code datasets to a **GPU T4 x2** Kaggle script. Use only one GPU.
Kaggle metadata has no environment field, so create a stage-specific code file
and point `code_file` at it:

```bash
K=/home/bikash/kaggle_kernel_dapi
cp scripts/kaggle/train_dapi_1024.py "$K/train_dapi_pretrain.py"
sed -i 's/STAGE = os.environ.get("DAPI_STAGE", "smoke").lower()/STAGE = "pretrain"/' \
  "$K/train_dapi_pretrain.py"
# set "code_file": "train_dapi_pretrain.py" in kernel-metadata.json
.venv/bin/python -m kaggle kernels push -p "$K"
```

Use the same pattern with `STAGE = "finetune"` and
`code_file: train_dapi_finetune.py`. Keep the repo default at `smoke`; the
stage-specific file lives in the kernel upload folder.

Checkpoints go to `/kaggle/working/checkpoints/dapi_{smoke,pretrain,finetune}/`.
Checkpoint filenames are per-epoch plus `checkpoint_last.pt`.

## 4a. Session handoff / resume

Kaggle stops GPU scripts at the session cap. Do this after a stop:

```bash
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/api_token)
mkdir -p /home/bikash/kaggle_out_pretrain
.venv/bin/python -m kaggle kernels output \
  bikasash/he-dapi-dapi-1024-corrected -p /home/bikash/kaggle_out_pretrain
```

The resumable file is
`checkpoints/dapi_pretrain/checkpoint_last.pt`. Upload **only that file** to a
private checkpoint dataset:

```bash
mkdir -p /tmp/kaggle_ckpt_pretrain
cp /home/bikash/kaggle_out_pretrain/checkpoints/dapi_pretrain/checkpoint_last.pt \
  /tmp/kaggle_ckpt_pretrain/
cat > /tmp/kaggle_ckpt_pretrain/dataset-metadata.json <<'JSON'
{
  "title": "he-dapi dapi pretrain checkpoint",
  "id": "bikasash/he-dapi-dapi-pretrain-ckpt",
  "licenses": [{"name": "other"}],
  "isPrivate": true
}
JSON
.venv/bin/python -m kaggle datasets create -p /tmp/kaggle_ckpt_pretrain --dir-mode tar
```

Add `bikasash/he-dapi-dapi-pretrain-ckpt` to `dataset_sources`, keep
`train_dapi_pretrain.py`, and push again. `find_input_checkpoint()` copies
`checkpoint_last.pt` into `/kaggle/working/checkpoints/dapi_pretrain/` and
`--resume auto` continues at the next epoch. Do not upload every epoch file;
the loader prefers `checkpoint_last.pt` and otherwise picks the highest epoch
number, not lexicographic order.

For fine-tune, attach the pretrain checkpoint dataset (or the fine-tune
checkpoint when resuming) and use `train_dapi_finetune.py`. The runner seeds
1024 from the pretrain checkpoint and runs 2×500 steps.

## 5. Evaluation

```bash
PYTHONPATH=. .venv/bin/python -m src.models.eval_offline \
  --checkpoint resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt \
  --h5 data/processed/hemit_val_1024.h5 --split val --num 630 \
  --patch-size 1024 --target "DAPI ch0" --label dapi-1024-unet \
  --out docs/metrics_dapi_1024_val.json

PYTHONPATH=. .venv/bin/python -m src.models.eval_offline \
  --checkpoint resources/models/checkpoints/he2dapi_dapi_1024/checkpoint_last.pt \
  --h5 data/processed/hemit_test_1024.h5 --split test --num 945 \
  --patch-size 1024 --target "DAPI ch0" --label dapi-1024-unet \
  --out docs/metrics_dapi_1024_test.json
```

`eval_offline.py` expects a training checkpoint with a `"G"` key, so use
`checkpoint_last.pt` (or add a state-dict fallback). For Kaggle test eval,
attach `bikasash/hemit-dapi-1024-ch0-test` plus the checkpoint dataset, copy
both into `/kaggle/working`, and run the same command with the mounted H5 path.

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
