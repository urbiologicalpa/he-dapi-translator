# Novelty Report: H&E <-> DAPI Translation
Date: 2026-08-30 | Search: web (arXiv, Nature, ScienceDirect, GitHub)

## Question
Has anyone built a drag-drop tool to convert H&E <-> DAPI and vice versa? Is our idea novel?

## Answer: PARTIALLY NOVEL — Science exists, Product gap exists.

### 1. What HAS been done (Published Science, 2019-2026)

**H&E -> DAPI (and multiplex) - MANY papers:**
- **HEMIT (Bian et al., MICCAI 2024 Workshops)** — *Direct hit*. Dataset + dual-branch Pix2Pix (ResNet CNN + Swin Transformer + Feature Map Fusion) for H&E -> mIHC including **DAPI** + CD3 + panCK. 5,292 pairs (3,717/630/945), same-slide cellular alignment. DAPI-branch paper numbers: SSIM 0.815, Pearson 0.716, PSNR 28.610 dB (composite mIHC metrics differ). Paper: https://arxiv.org/html/2403.18501v1 — This is exactly H&E -> DAPI with paired supervised learning.
- **DGR (Ma et al., Nature Communications 2026-03-27)** — Misalignment-resistant virtual staining, tested on 5 datasets including HEMIT. Decoupled Generation + Registration, +23.8% PSNR when misaligned (±5° rot, ±10% translation), pathologists 52% accuracy distinguishing real vs virtual (indistinguishable). https://www.nature.com/articles/s41467-026-71038-2
- **mSIGHT (Triple-Negative Breast Cancer, 2025, Sage)** — H&E -> multiplex IF (DAPI, panCK, CD3/CD4/CD8, CD20) with registration network (RegGAN-based). Outperformed Pix2Pix and CycleGAN: DAPI MSE 0.12 vs 0.27, FID 125 vs 321. Evaluated cell density + adjacency for neoadjuvant therapy prediction. https://sage.cnpereading.com/doi/10.1177/17588359251379411
- **Autofluorescence Virtual Staining (NCI, medRxiv 2024)** — Autofluorescence -> H&E + mIF (DAPI/PanCK/PD-L1/CD3/CD8) for NSCLC. Deep learning from label-free.
- **Scalable Virtual Multi-Staining (PLOS Comp Bio 2025)** — Single H&E encoder + 8 decoders (pediatric Crohn's, 480 WSIs). Trustworthy AI with confidence heatmaps.
- **D-VST (NeurIPS 2025)** — Diffusion Transformer for tone-controllable cross-dye WSI staining, frequency-aware patch sampling for billion-pixel WSI.

**DAPI/Fluorescence -> H&E (reverse direction):**
- **Virtual H&E from CODEX (PMC8721868, 2021)** — Commercial CODEX: DAPI + Eosin fluorescence -> virtual H&E via exponential transform + code: https://github.com/SimonsonLab/VirtualHE_examples — *Not deep learning but same transform*.
- **DAPI + Eosin -> Virtual H&E 3D via SACSI-CUT (SPIE 2025)** — DAPI+Eosin fluorescence double staining -> Virtual H&E with contrastive unpaired translation + self-attention.
- **FLIM -> H&E (Nature npj Imaging 2024)** — Label-free FLIM (lifetime + intensity) -> H&E via Pix2Pix + DISTS loss, 7 cell types signature.
- **Light-Sheet C01/C02 -> H&E via CycleGAN (arXiv 2601.08776, 2026)** — TO-PRO-3 nuclear + Eosin-like channels combined to RGB -> H&E using CycleGAN ResNet + PatchGAN, unsupervised, MHIST target 3152 H&E images.
- **Phase Contrast -> DAPI via VisCy (napari-tmidas)** — VisCy VSCyto3D predicts nuclei (DAPI/Hoechst-like) from transmitted light. napari plugin exists, but NOT H&E<->DAPI bidirectional.

**Tools/Plugins that ALMOST match:**
- `napari-tmidas` (VisCy virtual staining) — drag-free plugin for phase -> nuclei/membrane, not H&E.
- `multiplex2brightfield` (TristanWhitmarsh) — multiplex (IMC/CODEX) -> virtual brightfield (H&E/IHC) via channel mapping + LLM, OME-TIFF I/O, but rule-based color mapping, not GAN.
- `virtual_stain_flow` (WayScience) — framework for brightfield -> virtual stains (DAPI/tubulin/actin) with UNet, reproducible tracking.
- `LazySlide ROSIE` — H&E -> multiplex spatial gene expression (tile-level mean), not pixel-perfect DAPI.

### 2. What HAS NOT been done (Gap)

1. **No bidirectional drag-drop in one tool:** Every paper is one-way. No public tool lets user drop H&E *or* DAPI and get the other with same UI.
2. **No productized drag-drop WSI plugin:** Code is GitHub research scripts. No paper offers a polished web drag-drop + Docker Hub image + napari plugin unified. Closest is Hugging Face demos but single direction.
3. **No combined WSI + patch handling:** Papers handle patches (512-1024px). No unified tiled WSI inference with adaptive sampling + patch demo (D-VST proposes but not shipped).
4. **No uncertainty / trustworthy layer for H&E<->DAPI specifically:** PLOS 2025 has confidence heatmaps but for H&E->multi, not DAPI reverse hallucination control.

### 3. Novelty Assessment

| Aspect | Novelty | Evidence |
|--------|---------|----------|
| H&E -> DAPI science | LOW | HEMIT, DGR, mSIGHT already SOTA |
| DAPI -> H&E science | LOW-MEDIUM | CODEX, SACSI-CUT, light-sheet CycleGAN exist, but less explored than H&E->IHC |
| Bidirectional H&E <-> DAPI in single model/tool | MEDIUM-HIGH | No bidirectional product found |
| Drag-drop + WSI + patch + Docker Hub | HIGH | No such tool; gap is engineering/productization, not algorithm |

**Verdict:** Publishable as *tool paper* (like napari plugin / JOSS), not as core algorithm novelty unless you add DGR+diffusion hybrid + uncertainty.

### 4. How to make it still novel (Recommendations)

- **Angle 1 (Easiest, 3 weeks):** "First open bidirectional H&E<->DAPI virtual staining toolbox with WSI support" — ship docker + napari + web. Cite HEMIT/DGR, fine-tune their models, productize.
- **Angle 2 (Medium):** Add DGR decoupled registration + diffusion tone-control for DAPI->H&E hallucination reduction + uncertainty heatmaps. Claim "trustworthy bidirectional".
- **Angle 3 (Strongest, 3 months):** Train on HEMIT + add autofluorescence/FLIM channel as optional prior for DAPI->H&E to reduce hallucination, evaluate downstream task: Cellpose nuclei count error, T-cell density correlation (as in mSIGHT). This closes the ill-posed gap.

### 5. Sources Saved (in `resources/papers/`)

Will store PDFs/links:
- HEMIT arXiv 2403.18501
- DGR Nature Comms 2026
- mSIGHT Sage 2025
- Autofluorescence Lung medRxiv 2024
- FLIM npj Imaging 2024
- CODEX Virtual H&E PMC8721868
- SACSI-CUT SPIE 2025
- Light-sheet CycleGAN arXiv 2601.08776
- D-VST NeurIPS 2025
- PLOS Comp Bio 2025 multi-staining
- VisCy napari-tmidas docs
- multiplex2brightfield GitHub

## 6. Workplan Routing Given Findings

Your idea 1) "have to look internet what is available then reroute" — done. Reroute:

- Phase 1: Don't collect own data; reuse HEMIT (paired) + MHIST (unpaired) + CODEX examples. Fork HEMIT dual-branch repo as baseline.
- Phase 2: Train both directions separately, but share encoder (like PLOS 2025 single H&E encoder + multiple decoders) to make bidirectional efficient.
- Phase 3: Build drag-drop covering both directions from day 1 (toggle), with WSI tiled inference from D-VST paper.

Next: See `WORKPLAN.md` Phase 1 download scripts.

---
Search executed 2026-08-30 via live crawl. Full logs in tool outputs.
