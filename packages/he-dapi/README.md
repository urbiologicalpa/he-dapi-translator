# he-dapi

![status](https://img.shields.io/badge/status-alpha%2Fbuilding-orange) ![PyPI](https://img.shields.io/pypi/v/he-dapi)

> 🚧 **Early / work-in-progress (alpha).** Only the `he2dapi` direction is available today;
> `dapi2he` (unpaired CycleGAN) is pending training. APIs, weights and outputs may change
> between releases. **Research demo only — not for clinical use.**

**H&E ↔ DAPI virtual staining — inference library + CLI.**

Predicts a virtual **DAPI** fluorescence image from a brightfield **H&E** patch
(paired Pix2Pix). Weights are downloaded automatically from the
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
- `he2dapi` (paired Pix2Pix): **available**.
- `dapi2he` (unpaired CycleGAN): added when unpaired training completes.

## License
MIT. Research demonstration only — not validated for clinical use.
