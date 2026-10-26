# Launch Checklist — v0.2.0 (H&E ↔ DAPI Virtual Staining)

> **Status 2026-10-23:** the legacy 512 checkpoint is the real **H&E→DAPI
> (`ch2`)** model; its HF generator and H5 target are verified. The 1024 `ch0`
> models are the panCK ablation. The corrected 1024 `ch2` forward/reverse exist
> but are undertrained; R-007 (unpaired reverse) is invalid for selection, so
> the paired U-Net remains default. Stage 2.5 (longer 1024 paired, val-based
> selection, colour/gradient losses) is **prepared but not launched**; it needs
> the 2026-10-24 quota reset + approval. Do not launch bidirectional or
> release-quality claims until the v0.2.0 gate in [`PLAN.md`](PLAN.md) is met.
> This file is the launch process template.

Goal: one coordinated launch **after** training completes and the real backend+frontend demo
ships — not a trickle. This turns a 0-star repo into something shareable and citable.

Positioning (the hook):
> *A product-grade virtual-staining model (Pix2Pix + CycleGAN) trained to convergence on a **3.7GB RAM, CPU-only laptop** — with crash-safe, reboot-proof training and a live in-browser demo.*

---

## 0. Sequencing (do NOT launch before these)
- [x] **Corrected DAPI data** (`dapi = mihc[...,2]`) is built and verified.
- [ ] Corrected H&E→DAPI and DAPI→H&E pass the v0.2.0 gate: full 630 val + 945 test, CIs, same checkpoint SHA in API/UI/Docker/HF, and manual prod approval.
- [ ] **v1.0** release cut: real **backend (FastAPI)** + **frontend** demo, not the stopgap Space.
- [ ] Space weights refreshed with the **epoch-200** ONNX.
- [ ] README "Results" gallery shows real before/after (issue #8).
- [ ] Reproduction story is airtight: `docker run` one-liner + `python -m src.models.train_paired` documented.

## 1. Pre-launch repo polish
- [ ] Social preview image set (Settings → Social preview): before → after H&E/DAPI, repo name.
- [ ] 3–5 `good first issue` open and unassigned (#6–#10 done).
- [ ] `help wanted` on 1–2 meaty issues (#1, #3 done).
- [ ] A short `docs/RESULTS.md` (or HF model card) with the final metrics table + training curves.
- [ ] CHANGELOG v1.0 entry.
- [ ] Cite-able artifact: HF model card with metrics + a short write-up (blog/arXiv-style).
- [ ] Pages landing updated to link the v1.0 demo + weights + paper.

## 2. Launch-day copy (adapt per channel)

**Title (Reddit / HN):** `I trained a H&E→DAPI virtual staining model to convergence on a 3.7GB CPU-only laptop`
**Title (X / LinkedIn):** `H&E → DAPI virtual staining, trained to convergence on a 3.7GB CPU-only laptop — code, weights, and a live in-browser demo: <link>`

**Body (short, link at the end):**
> Virtual staining predicts a DAPI (fluorescence) image from brightfield H&E. I trained a paired
> Pix2Pix (+ CycleGAN for unpaired) model on HEMIT to 200 epochs — **on a 3.7GB RAM, CPU-only
> laptop**, using streaming I/O (10.6GB→140MB peak), per-epoch atomic checkpoints, and cron-based
> reboot recovery. Final: PSNR ~__ dB, SSIM __, nuclei Dice __.
> Try it in the browser (no install): <Space link> · Weights: <HF model> · Code: <repo>

## 3. Channels (post the same day, staggered)
| Channel | Notes |
|---|---|
| Reddit `r/computationalpathology`, `r/MachineLearning` (Showcase Sat), `r/MedicalImageAnalysis` | Best-fit audience; include image, not just link |
| X / Twitter | 3-tweet thread: (1) hook+gif, (2) constraint story, (3) links; tag histopath/ML accounts |
| LinkedIn | Frame as low-resource-ML + reproducibility |
| Hacker News (Show HN) | Tue–Thu ~13:00–15:00 UTC; constraint story is the draw |
| Hugging Face community / model card | Auto-surfaces in HF search |
| Papers with Code | Link model+repo once write-up exists |
| Relevant Discords/Slacks | Drop in `#showcase` channels |

## 4. Timing
- Best: **Tue–Thu, 13:00–16:00 UTC** (US morning + EU afternoon).
- Don't launch Fri/weekend. Don't cross-post identical text within minutes (looks spammy).
- Have the demo warm and the repo README final **before** the first post.

## 5. Post-launch follow-up (the part most people skip)
- [ ] Reply to **every** comment/issue within a few hours — engagement compounds reach.
- [ ] Convert feedback into issues (`triage`), assign `good first issue` where small.
- [ ] Track GitHub **Traffic → Views/Referrers** and HF Space likes; note what channel converted.
- [ ] Publish a short "what I learned" follow-up 1–2 weeks later (second reach wave).

## 6. Metrics to watch (baseline today)
- Repo: 0 stars / 0 forks / 1 watcher; 20 views, 1 unique (14d); referrers = github.com only.
- Success at +7 days: >25 stars, >5 forks, ≥3 external issues/PRs, first outside contributor.

## 7. Don'ts
- Don't claim clinical validity — keep the "research demo, not for diagnosis" disclaimer.
- Don't over-hype bare metrics; show the images and the constraint story instead.
- Don't launch the stopgap Space as if it's v1.0.
