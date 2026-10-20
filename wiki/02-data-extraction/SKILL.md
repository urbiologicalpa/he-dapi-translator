# Skill: extraction_optimizer

You are the extraction optimizer sub-agent. Goal: fix OOM crashes during data packing on 3.7GB box.

## Before (crash path)
- `scripts/data/pack_hemit.py:86-107` concatenated 3 H5 (3.7+0.68+0.96=5.3G) via `np.concatenate` → need 2× RAM (10.6G) → OOM kill, machine crash.
- `scripts/data/pack_bci.py:71-88` same pattern.
- `prepare_patches.py:33` held all pairs in list before H5 write → 500×512×512×3 ≈ 375MB, okay but 5000 → 3.7GB OOM.
- No RAM monitoring; flush every 500 but no GC or throttle.

## After (optimized)
- **Streaming H5 copy**: compute total, create empty dest, copy in batches B=100 (≈78MB per batch), flush+gc every 500. Peak <150MB vs 10GB.
- **Per-iter GC**: `del he,mihc` + `gc.collect()` every 100 if avail <500MB.
- **ResourceMonitor**: checks at start, each 500, end → logs to `wiki/02-data-extraction/profile.json`. If avail <500MB, throttle batch to 50.
- **prepare_patches.py**: warning if >1000 candidates, advise streaming path; chunked gc; handle large dir without preload.

## Verify
```bash
.venv/bin/python scripts/data/pack_hemit.py  # should skip existing train/val/test, stream combined in <200MB
du -sh data/processed/hemit_*.h5
.venv/bin/python -m src.utils.resource_monitor --check
cat wiki/02-data-extraction/profile.json | tail
```

## Files patched
- `scripts/data/pack_hemit.py:1` (monitor + streaming combined)
- `scripts/data/pack_bci.py:1` (same)
- `resources/datasets/prepare_patches.py:1` (gc + warn)
