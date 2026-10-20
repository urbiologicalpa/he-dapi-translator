# Makefile - Resumable, spike-guarded workflow for 3.7GB RAM CPU box
PY=.venv/bin/python
MON=$(PY) -m src.utils.resource_monitor --check

.PHONY: health data train-paired-dummy train-unpaired-dummy infer clean wiki

health:
	@echo "=== Machine Health ==="
	@$(MON)
	@echo "=== Disk ==="
	@du -sh data/raw/* data/processed/*.h5 2>&1 | head -n 20

data:
	@echo "=== Data extraction (streaming) ==="
	@$(PY) -m src.utils.resource_monitor --check
	@$(PY) scripts/data/pack_hemit.py 2>&1 | tee /tmp/pack_hemit.log
	@$(PY) scripts/data/pack_bci.py 2>&1 | tee /tmp/pack_bci.log
	@echo "[OK] data - check wiki/02-data-extraction/profile.json"

train-paired-dummy:
	@$(PY) src/models/train_paired.py --config configs/hemit_dual_branch.yaml --dummy --optimize-for lowram 2>&1 | tee /tmp/train_paired_dummy.log
	@echo "log: /tmp/train_paired_dummy.log"

train-unpaired-dummy:
	@$(PY) src/models/train_unpaired.py --config configs/cyclegan_dapi2he.yaml --dummy --optimize-for lowram 2>&1 | tee /tmp/train_unpaired_dummy.log

infer:
	@$(PY) src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/synth_dapi.png 2>&1 | tee /tmp/infer.log
	@ls -lh /tmp/synth_dapi.png || echo "fallback dummy (no torch) still produced? check /tmp/synth_dapi.png"
	@$(PY) src/models/infer.py --input data/examples/he_example.png --direction he2dapi --out /tmp/synth_dapi.tiff --stream 2>&1 | tee /tmp/infer_stream.log
	@ls -lh /tmp/synth_dapi.tiff || true

infer-wsi-stream:
	@$(PY) src/models/infer.py --input data/raw/wsi/CMU-1.svs --direction he2dapi --wsi --stream --out /tmp/wsi_stream.tiff 2>&1 | tee /tmp/wsi_stream.log

wiki:
	@cat wiki/README.md
	@ls -R wiki

clean:
	rm -f data/processed/*.tmp /tmp/*.tiff /tmp/*.dat /tmp/*.log
	rm -f resources/models/checkpoints/*/*.tmp

# Full pipeline (resumable)
all: health data train-paired-dummy infer
	@echo "=== All done (low-RAM) ==="
	@cat wiki/README.md
