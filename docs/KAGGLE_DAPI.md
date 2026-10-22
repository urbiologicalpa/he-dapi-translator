# Kaggle DAPI runbook and status

Last checked: **2026-10-22 06:03 UTC**

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
- **512 pretrain complete** at ~`13:09 UTC` (epoch 20/20).
  - Checkpoint: `checkpoints/dapi_pretrain/checkpoint_last.pt`
  - SHA-256: `be0b2892cf91f8e00ae844aa2ae349986414dd0a46da657d0ef6fa2aedf4abf4`
  - Uploaded as private dataset `bikasash/he-dapi-dapi-pretrain-ckpt`
- **1024 fine-tune complete** at ~`13:37 UTC` (2×500 steps).
  - Checkpoint: `checkpoints/dapi_finetune/checkpoint_last.pt`
  - SHA-256: `871560a14e54a8f9ba8a7b59b7b4a3e0a61e4d0c0936845fd3d938b5bfb4103d`
  - Uploaded as private dataset `bikasash/he-dapi-dapi-finetune-ckpt`
- **Full val/test eval complete** at `15:21 UTC` (kernel v8):
  - val 630: SSIM 0.6702, Pearson 0.2547, PSNR 28.77, Dice 0.1036
  - test 945: SSIM 0.7253, Pearson 0.6107, PSNR 26.60, Dice 0.4489
  - Metrics saved as `docs/metrics_dapi_1024_{val,test}.json`
- Test dataset `bikasash/hemit-dapi-1024-ch0-test` was ready and attached.
- This baseline is undertrained, not release quality.
- **Reverse paired DAPI → H&E training complete** in
  `bikasash/he-dapi-dapi2he-1024`:
  - 512 pretrain: 20/20 epochs, final G `14.8121`, D `0.5576`.
  - 1024 fine-tune: 2/2 epochs ×500 steps, final G `14.8727`, D `0.5629`.
  - Checkpoint: `checkpoints/dapi2he_finetune/checkpoint_last.pt`
    (SHA-256 `dbe6ebdef4d85400085a3740edeea05504d2e01387dbd96c4eb2e1e3e29f634d`).
  - Generator: `checkpoints/dapi2he_finetune/generator_last.pt`
    (SHA-256 `c5449c5b780be4efbe192f077a77777c393a85022eaa3e585d2aa2d37266b27b`).
  - Uploaded as `bikasash/he-dapi-dapi2he-finetune-ckpt`.
- **Reverse full eval complete** in `bikasash/he-dapi-dapi2he-1024-eval`:
  - val 630: SSIM `0.4367`, Pearson `0.5214`, PSNR `15.18`, L1 `0.1449`
  - test 945: SSIM `0.5930`, Pearson `0.5764`, PSNR `15.79`, L1 `0.1270`
  - Metrics saved as `docs/metrics_dapi2he_1024_{val,test}.json`.
  - Not release quality; low SSIM/PSNR indicates blur/color mismatch.
  - Stage `all`: 512 pretrain (20 epochs) then 1024 fine-tune (2×500 steps)
    in one kernel, so no intermediate checkpoint dataset is required.
  - Configs: `configs/hemit_dapi2he_512_pretrain.yaml`,
    `configs/hemit_dapi2he_1024.yaml`.
  - Code dataset `bikasash/he-dapi-code-dapi-1024` updated with direction
    support, reverse configs, and generic eval.
  - Local reverse smoke passed: 2 dummy epochs + ONNX export.
- Test H5 is **not attached** to the pretrain kernel; attach
  `bikasash/hemit-dapi-1024-ch0-test` for final test eval.
- GPU quota at launch: **~29.9h**, refresh `2026-10-24T00:00:00`.

## Why the run is staged

A 100-epoch, native-1024px run is not realistic on a 30h T4 quota. Plan:

1. **Smoke** — done: ~1.96 s/step at 512, batch 2.
2. **Pretrain** — 20 epochs at 512; ~2.5h at the measured ~7 min/epoch steady state.
3. **Fine-tune** — 2×500 steps at native 1024 from the 512 checkpoint.
4. **Evaluate** — full 630 val + 945 test at 1024; save labeled metrics.

## Reverse direction (DAPI → H&E)

The same paired trainer now takes `--direction dapi2he` (or the `data.direction`
config key). It swaps the generator input/target and builds a 1→3 channel UNet;
the PatchGAN still receives source + target (1+3 channels).

```bash
# local smoke
PYTHONPATH=. .venv/bin/python -m src.models.train_paired \
  --config configs/hemit_dapi2he_512_pretrain.yaml --dummy

# Kaggle: new kernel bikasash/he-dapi-dapi2he-1024, stage "all"
# entrypoint defaults to DAPI_DIRECTION=dapi2he and DAPI_STAGE=all
```

Reverse outputs land in `checkpoints/dapi2he_pretrain/` and
`checkpoints/dapi2he_finetune/`. Eval writes
`metrics_dapi2he_1024_{val,test}.json`; Dice is `null` for H&E targets because
the nuclei-Dice metric is DAPI-only.

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

`eval_offline.py` accepts `checkpoint_last.pt` (wrapped `"G"`), a bare generator
state dict, and CycleGAN `net.*` weights (auto-selects `ResnetGenerator`). For
Kaggle test eval, attach `bikasash/hemit-dapi-1024-ch0-test` plus the checkpoint
dataset, copy both into `/kaggle/working`, and run the same command with the
mounted H5 path.

### 5a. Quality, unpaired and CPU eval kernels

| Kernel | Purpose | GPU | Sources |
|---|---|---|---|
| `bikasash/he-dapi-dapi2he-quality-1024` | reverse quality fine-tune | yes | data, code, `he-dapi-dapi2he-finetune-ckpt` |
| `bikasash/he-dapi-he2dapi-quality-1024` | forward quality fine-tune | yes | data, code, `he-dapi-dapi-finetune-ckpt` |
| `bikasash/he-dapi-cyclegan-dapi2he-1024` | unpaired CycleGAN | yes | data, `mhist-he-256`, code |
| `bikasash/he-dapi-*-quality-eval-cpu` | FID/LPIPS on kernel output | no | data, code, `kernel_sources: <train kernel>` |

The quality kernels use `train_dapi_quality.py` with `STAGE="quality"` and
`DIRECTION` set to `dapi2he` or `he2dapi`; configs are
`configs/hemit_{dapi2he,dapi}_quality_1024.yaml`. The unpaired kernel runs
`train_unpaired.py` (`configs/cyclegan_dapi2he.yaml`) with MHIST H&E (256px) as
the external domain and evaluates the last generator on `hemit_val_1024.h5`.
Kaggle allows two concurrent GPU sessions: queue the unpaired kernel after a
quality kernel finishes.

The CPU eval kernels (`scripts/kaggle/eval_checkpoint.py`) depend on the
training kernel via `kernel_sources`, enable internet for Inception/LPIPS
weights, and write `/kaggle/working/metrics_eval.json`.

## Tracking table

| Stage | Status | Notes |
|---|---|---|
| Local panCK epoch 200 | done | full 630/945 len eval recorded |
| Build 1024 H5 | done | train 11.5 GB, val 2.1 GB, test 3.0 GB |
| Upload data dataset | ready (train/val) | test split in separate dataset |
| Upload code dataset | ready | nested `code_bundle/` mount handled |
| Kaggle smoke | **done** | ~1.96 s/step at 512, batch 2 |
| 512 pretrain | **done** | epoch 20/20; checkpoint sha `be0b2892…f4abf4` |
| 1024 fine-tune | **done** | 2×500 steps; checkpoint sha `871560a1…b4103d` |
| Full val/test eval | **done** | val/test SSIM 0.670/0.725; see metrics JSON |
