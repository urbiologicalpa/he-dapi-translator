# napari-he-dapi

![status](https://img.shields.io/badge/status-alpha%2Fbuilding-orange) ![PyPI](https://img.shields.io/pypi/v/napari-he-dapi)

> ⚠️ **Alpha / channel correction implemented (2026-10-23).** Evidence shows the raw
> HEMIT RGB TIFF stores **DAPI in `ch2` (blue)**, panCK in `ch0`, CD3 in `ch1`.
> The legacy 512 `he2dapi` weights are therefore the **real H&E→DAPI** model,
> not a panCK ablation; the 1024 `ch0` models are panCK. DAPI claims are frozen
> while the legacy weights are re-verified. `dapi2he` has no valid DAPI
> checkpoint. **Do not use for clinical work.** Research demo only.

A [napari](https://napari.org) plugin for **H&E ↔ DAPI virtual staining**, powered by
[`he-dapi`](https://pypi.org/project/he-dapi/).

Open an H&E image in napari, launch the widget from **Plugins → H&E <-> DAPI**,
pick the direction, and the predicted virtual stain appears as a new layer.

## Install
```bash
pip install napari-he-dapi
```

## Use
1. Open napari and load an H&E image (`File → Open`).
2. `Plugins → H&E <-> DAPI Virtual Staining → H&E <-> DAPI`.
3. Choose **direction** (`he2dapi`), click **Run virtual staining**.

Weights download automatically from
[`urbiologicalpa/he-dapi-translator`](https://huggingface.co/urbiologicalpa/he-dapi-translator).

## Status
`he2dapi` is wired up but currently resolves to the legacy 512 DAPI checkpoint (provenance re-verification pending).
Corrected DAPI weights and `dapi2he` are not available yet. See the repository's
`docs/MODEL_CARD.md` for the exact limitations.

## License
MIT. Research demonstration only — not validated for clinical use.
