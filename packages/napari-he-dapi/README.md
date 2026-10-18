# napari-he-dapi

![status](https://img.shields.io/badge/status-alpha%2Fbuilding-orange) ![PyPI](https://img.shields.io/pypi/v/napari-he-dapi)

> 🚧 **Early / work-in-progress (alpha).** Only the `he2dapi` direction is wired up today;
> `dapi2he` (unpaired CycleGAN) lands when training completes. APIs may change.
> **Research demo only — not for clinical use.**

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
`he2dapi` (paired Pix2Pix) available; `dapi2he` (CycleGAN) when unpaired training completes.

## License
MIT. Research demonstration only — not validated for clinical use.
