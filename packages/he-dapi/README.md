# he-dapi

![status](https://img.shields.io/badge/status-alpha%2Fbuilding-orange) ![PyPI](https://img.shields.io/pypi/v/he-dapi)

> ⚠️ **Alpha / channel audit in progress (2026-10-23).** Evidence now shows the
> raw HEMIT RGB TIFF stores **DAPI in `ch2` (blue)**, panCK in `ch0` (red), CD3
> in `ch1` (green). The legacy 512 weights in this package are therefore the
> **real H&E→DAPI** model, not a panCK ablation. The 1024 `ch0` models are
> panCK. DAPI claims are frozen while ADR-014 is implemented and the weights are re-verified.
> Research demo only — not for clinical use. See `docs/CHANNEL_AUDIT.md` in the
> source repository.

**H&E ↔ DAPI virtual staining toolkit — inference library + CLI (alpha; forward
baseline only).**

Intended product: one library/CLI for both H&E → DAPI and DAPI → H&E. The
current implementation exposes only the forward direction; the reverse trainer
is a skeleton. The current forward checkpoint was trained on the wrong HEMIT
channel and should be treated as the legacy H&E → DAPI model (provenance
re-verification pending). Weights are downloaded
automatically from the
[Hugging Face model repo](https://huggingface.co/urbiologicalpa/he-dapi-translator).

## Install
```bash
pip install he-dapi
# optional extras
pip install "he-dapi[wsi]"     # .svs/.tiff support
pip install "he-dapi[onnx]"    # ONNX Runtime
```

## CLI
```bash
he-dapi info
he-dapi translate -i he.png -o dapi.png -d he2dapi
```

## Python
```python
from he_dapi import translate, load_generator

pred = translate("he.png", direction="he2dapi")   # -> uint8 HxW numpy array
# or reuse a loaded model across calls:
model = load_generator("he2dapi")
pred  = translate("he.png", model=model)
```

Large images are processed in 512px tiles (stride 256) with overlap averaging.
Input is normalized to `[-1, 1]` (`uint8 / 127.5 - 1`); output is `[0, 1]`.

## Status
- `he2dapi` (paired Pix2Pix): **legacy 512 H&E→DAPI**; provenance re-verification pending
  weights are pending.
- `dapi2he` (unpaired CycleGAN): not available; the trainer is a smoke-test
  skeleton.
- Architecture: U-Net baseline, not the HEMIT dual-branch Swin/FMF model.
- Evaluation: no valid DAPI metrics yet. See the repository's `docs/RESULTS.md`.

## License
MIT. Research demonstration only — not validated for clinical use.
