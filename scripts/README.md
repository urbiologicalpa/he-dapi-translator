# Scripts

Operational entry points that are not part of the installable `src/` package.

## `scripts/data/`

| Script | Purpose | Output |
|---|---|---|
| `pack_hemit_1024.py` | **Correct** HEMIT builder: native 1024px, `DAPI = ch0`, optional Macenko on H&E, gzip/lzf, resumable | `data/processed/hemit_{train,val,test}_1024.h5` |
| `pack_hemit.py` | Legacy 512px HEMIT builder (streaming, low-RAM) | `data/processed/hemit_{train,val,test}_512.h5` |
| `pack_bci.py` | BCI HER2 paired builder | `data/processed/bci*.h5` |
| `hemit_extract.py` | One-shot HEMIT archive extraction | `data/raw/hemit/extracted/` |

The 512px files currently on disk were built before the channel-order fix and are
**not valid DAPI data**. Rebuild with `pack_hemit_1024.py` (or rebuild the 512px
files from corrected code) before training H&E -> DAPI.

```bash
# corrected 1024px HEMIT, gzip level 4
uv run --extra data python -m scripts.data.pack_hemit_1024 \
  --splits train val test --patch-size 1024 --compression gzip --compression-level 4
```

## `scripts/kaggle/`

| Script | Purpose |
|---|---|
| `he-dapi-product-gpu-train.py` | Kaggle T4 training helper (kept for the corrected DAPI run) |

## Historical paths

Older notes in `wiki/` and `docs/optimization/` may still mention the scripts at
the repository root. Those are dated logs; the current paths are the ones above.
