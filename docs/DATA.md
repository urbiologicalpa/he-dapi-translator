# HEMIT data

## Source

- Dataset: [BianChang/HEMIT-DATASET](https://github.com/BianChang/HEMIT-DATASET)
- License: **CC BY 4.0**
- Task: paired H&E → multiplex immunohistochemistry (mIHC)
- Resolution: 1024 × 1024 native patches
- Pairs: **5,292** total

| Split | Pairs |
|---|---:|
| Train | 3,717 |
| Val | 630 |
| Test | 945 |

Raw layout:

```
data/raw/hemit/extracted/
├── train/
│   ├── input/   # H&E RGB TIFFs
│   └── label/   # mIHC RGB TIFFs, same filenames
├── val/
│   ├── input/
│   └── label/
└── test/
    ├── input/
    └── label/
```

## Channel order (critical)

HEMIT label TIFF channels are:

| Channel | Marker | Use |
|---|---|---|
| `ch0` | **DAPI** | H&E → DAPI target |
| `ch1` | CD3 | optional auxiliary target |
| `ch2` | panCK | optional auxiliary target |

This was confirmed against the official `post_process.py` /
`tif_composite` mapping (ch0 → blue/DAPI, ch1 → green/CD3, ch2 → red/panCK).

> **Legacy bug.** The first 512px H5 files were built with `ch2` as the target.
> They are **panCK data**, not DAPI. Do not train or evaluate a DAPI model on
> `data/processed/hemit_*_512.h5`. The bug is fixed in
> `src/models/datasets.py` and `scripts/data/pack_hemit.py`; new work should use
> the corrected 1024px builder.

## Build corrected data

Install the optional data dependencies:

```bash
uv pip install --python .venv/bin/python "torchstain==1.4.1" imagecodecs
# or: uv sync --extra data
```

Build all splits:

```bash
uv run --extra data python -m scripts.data.pack_hemit_1024 \
  --splits train val test \
  --patch-size 1024 \
  --compression gzip --compression-level 4
```

Smoke test without touching the full dataset:

```bash
uv run --extra data python -m scripts.data.pack_hemit_1024 \
  --splits val --limit 2 --out-dir /tmp/hemit_smoke
```

Outputs:

```
data/processed/hemit_train_1024.h5
data/processed/hemit_val_1024.h5
data/processed/hemit_test_1024.h5
```

Each file contains:

| Dataset | Shape | Meaning |
|---|---|---|
| `he` | `(N, 1024, 1024, 3)` uint8 | H&E, optionally Macenko-normalized |
| `mihc` | `(N, 1024, 1024, 3)` uint8 | raw mIHC channels |
| `dapi` | `(N, 1024, 1024)` uint8 | `mihc[..., 0]` |

Attributes include `patch_size`, `num_pairs`, `source`, `channel_order`,
`dapi_channel=0`, `macenko`, and `macenko_reference`.

Expected storage with gzip level 4 is about **10.4 GB total** (~7.3 GB train,
~1.2 GB val, ~1.85 GB test). This fits a Kaggle dataset while remaining
streamable from HDF5.

## Verification

```bash
python - <<'PY'
import h5py
with h5py.File("data/processed/hemit_val_1024.h5", "r") as f:
    print(f.attrs["channel_order"], f.attrs["dapi_channel"])
    assert (f["dapi"][0] == f["mihc"][0, ..., 0]).all()
    assert not (f["dapi"][0] == f["mihc"][0, ..., 2]).all()
print("DAPI channel OK")
PY
```

Regression tests:

```bash
PYTHONPATH=. python -m pytest tests/test_hemit_channels.py -q
```

## Normalization and licensing

- Macenko normalization is applied to **H&E only** at pack time.
- The reference is `torchstain`'s default Macenko reference; record it in any
  experiment report.
- Keep HEMIT attribution and the CC BY 4.0 notice with redistributed data or
  weights. Do not commit raw data or H5 files to this repository.
