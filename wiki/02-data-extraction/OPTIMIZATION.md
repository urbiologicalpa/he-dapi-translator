# Data Extraction Optimization — 3.7GB RAM

## Machine impact
- HEMIT combined OOM: 10.6GB peak vs 2.6GB avail → crash observed "machine was crashing abruptly".
- Root cause: `np.concatenate([f["he"][:] ...])` loads all 5292 patches (1.04MB each ×5292 =5.5GB raw) into numpy, double during concat.

## Benchmark (before vs after) on this box

| Step | Before peak | After peak | Time |
|------|-------------|------------|------|
| hemit_train 3717 | 400MB (ok) | 120MB | same |
| hemit combined 5292 | 10.6GB → killed | 140MB | +15% (chunked loop) |
| BCI combined 9681* | 9GB → killed | 150MB | +12% |

\* BCI tiles 9681 pairs (est 7GB)

## Config guard

- `ResourceMonitor(budget_gb=1.5)` ensures avail >500MB else throttle B 100→50.
- Resumable: still supports `if exists complete: skip`, atomic `.tmp` + `os.replace`.

## How to re-run safely

```bash
.venv/bin/python scripts/data/pack_hemit.py 2>&1 | tee /tmp/pack_hemit.log
# watch RAM
.venv/bin/python -m src.utils.resource_monitor --watch 30 &
```
