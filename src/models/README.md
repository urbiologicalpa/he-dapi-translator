# `src/models`

Training and inference code.

| File | State | Notes |
|---|---|---|
| `train_paired.py` | Implemented | U-Net Pix2Pix baseline, PatchGAN, L1 + SSIM + optional DISTS + nuclei Dice |
| `train_unpaired.py` | Smoke-test skeleton | ResNet/CycleGAN dummy loop; no trained reverse model |
| `infer.py` | Implemented | Patch + tiled/streaming inference, optional ONNX path |
| `eval_offline.py` | Implemented | Low-RAM, read-only evaluation |
| `datasets.py` | Implemented | HEMIT H5 loader; `dapi = mihc[..., 0]` fallback |
| `losses.py` | Implemented | SSIM proxy, nuclei Dice |
| `export_onnx.py` | Implemented | ONNX export helper |

**Important:** `configs/hemit_dual_branch.yaml` is named for the planned HEMIT
dual-branch model, but `train_paired.py` currently instantiates `UNetGenerator`.
The dual-branch Swin-T + FMF implementation is still a roadmap item.

Current public checkpoints are the **H&E → panCK ablation**, not DAPI. See
`docs/MODEL_CARD.md` and `docs/RESULTS.md`.

```bash
# CPU smoke test (no real checkpoints touched)
PYTHONPATH=. python -m src.models.train_paired \
  --config configs/hemit_dual_branch.yaml --dummy --optimize-for lowram

# inference (current checkpoint is the ablation)
PYTHONPATH=. python src/models/infer.py \
  --input data/examples/he_example.png --direction he2dapi \
  --checkpoint resources/models/checkpoints/he2dapi/generator_last.pt \
  --out /tmp/synthetic.png
```
