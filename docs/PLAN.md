# PLAN v2.0 — canonical product + research plan

> **Channel audit RESOLVED 2026-10-23 — ADR-014 accepted.**
> The raw HEMIT RGB TIFF has **ch2=DAPI (blue), ch0=panCK (red), ch1=CD3
> (green)**. The legacy 512 model is the real DAPI model; the 1024 `ch0`
> forward/reverse models and R-006 are panCK and must not be presented as DAPI.
> See [`CHANNEL_AUDIT.md`](CHANNEL_AUDIT.md). The 256 H5 rebuild is complete and
> verified; the loader reads `mihc[...,2]`. The corrected `ch2` 1024 baseline is
> complete but undertrained (see `RESULTS.md`). **R-007** (Stage 2.3, unpaired
> `ch2` 256) completed 12:13 UTC but is **INVALID for selection** (no FID/LPIPS;
> Tanh/[0,1] mismatch). The paired U-Net remains default; Stage 2.5 is
> running now and will resume after the quota reset.

**Status:** authoritative as of **2026-10-23 12:57 UTC**. Supersedes
`docs/WORKPLAN.md` v1.1 (kept only as history).
**Owner:** user (`urbiologicalpa`) + coding agent.
**Read order for any LLM/new session:**
1. this file (strategy, gates, decisions),
2. `docs/LIVE_STATE.md` (what is running right now, exact commands),
3. `docs/RESULTS.md` (measured numbers only),
4. `docs/TRACKER.md` / `docs/progress.json` (short + machine-readable status),
5. checkpoint files under `resources/models/checkpoints/` (training ground truth).

If any document disagrees with this one, **this file wins** until explicitly
revised with a dated decision entry.

---

## 1. Product definition

**One-line:** a cloud-ready, Dockerized, research-use tool that accepts either
H&E or DAPI and synthesizes the counterpart for patches and whole-slide images
(WSI), with honest confidence/limitations.

**Users:** computational pathologists, researchers, and lab scientists who need
a quick cross-modality view, not a diagnostic device.

**Product thesis:** one bidirectional workflow (direction router + patch/WSI +
API/plugin/UI), not another one-way paper reproduction.

**Non-goals (explicit):**
- No clinical/diagnostic claim; output is synthetic and must be watermarked.
- No on-prem requirement; public data only.
- No single model that is bijective; the two directions remain separate models
  until a shared-encoder design is proven.
- No release of the panCK ablation as DAPI, and no mixing of panCK metrics with
  DAPI claims.

**Product success criteria (release gate, v0.2.0):**
1. Both directions have corrected, release-quality weights with full
   **630 val + 945 test** metrics and checkpoint SHA-256 manifests.
2. The same checkpoint SHA is used by API, UI, Docker, HF, and packages.
3. Direction router + patch + WSI paths pass a documented UAT.
4. CPU/ONNX inference benchmark is recorded and stays within the stated budget.
5. Model card states data, target channel, resolution, limitations, and
   "research use only"; production deploy is manually approved (AGENTS.md).

---

## 2. Stage/gate model (refined)

The old 12-week Gantt is replaced by **evidence-gated stages**. A stage is not
"done" because code exists; it is done when its gate artifact exists.

| Stage | Gate artifact | Current state (2026-10-23) |
|---|---|---|
| 0. Truth & infra | CI green, Docker build, resource guards, channel regression tests | **DONE** |
| 1. Data contract | `ch2=DAPI` proof, loader precedence, 256 H5 rebuilt/verified, no leakage, checksums | **DONE (re-closed 2026-10-23)** |
| 2. Model baselines | Valid `ch2` forward + reverse paired baselines; fair unpaired comparison; selection memo | **PAIRED SELECTED**; valid 1024 `ch2` both-direction baselines complete (50-crop), quality gate not passed; R-007 unpaired run invalid, so paired remains default; Stage 2.5 running (resumable) |
| 3. Validation | Full-split metrics, CIs, qualitative/blinded audit, failure analysis, limitations | **PARTIAL** |
| 4. Product integration | Router + API/UI + WSI + ONNX + model manifest, same checkpoint everywhere | **ALPHA** |
| 5. Release | Staging UAT, manual prod approval, model card, rollback | **ALPHA** |
| 6. Post-release | Monitoring, drift/feedback, quota/cost review | **NOT STARTED** |

### Stage 2 breakdown (current critical path)

| Sub-stage | Hypothesis / purpose | Gate |
|---|---|---|
| 2.1 Forward paired baseline | U-Net can learn H&E→DAPI (`ch2`) | Legacy 512 `ch2` verified (FID 146.27, SSIM 0.7662). Corrected 1024 `ch2` fine-tune exists (FID 215.38, SSIM 0.7733, Pearson 0.5515, 50 val crops) but **does not beat legacy on FID/LPIPS/PSNR/L1**; longer fine-tune needed. |
| 2.2 Reverse paired baseline | U-Net can learn DAPI→H&E | **Valid first baseline:** corrected 1024 `ch2` reverse, 50 val crops at **1024 px**, FID 282.92, LPIPS 0.5317, PSNR 15.42, SSIM 0.4891, Pearson 0.6644; the same checkpoint at **256 px** (the R-007 protocol) gives FID 285.29, LPIPS 0.5129. Not release quality. |
| 2.3 Fair unpaired comparison | CycleGAN on **HEMIT** H&E (same target domain, same 256px eval) is competitive with paired reverse | **COMPLETE 12:13 UTC, exit 0 — INVALID for selection.** The eval omitted `--metrics fid,lpips`, so no FID/LPIPS; raw metrics also used Tanh output against `[0,1]` targets. 10-epoch smoke, not a fair verdict. See `LIVE_STATE.md` §0.5. |
| 2.4 Reverse model selection | Pick paired vs unpaired using **val** FID/LPIPS + blinded qualitative audit + cost; test only once | **CLOSED 12:20 UTC: paired U-Net remains default.** Unpaired produced no valid comparison; `docs/selection_reverse.md` records the invalid run and the decision. |
| 2.5 Quality pass | Longer 1024 schedule, val-based checkpoint selection, perceptual/color losses, dual-branch forward | **RUNNING 12:56 UTC (resumable)** on `bikasash/he-dapi-paired-stage25-reverse`; quota will stop after ~1–2 epochs; resume after `2026-10-24 00:00 UTC`. Reverse first; forward after. |

**2.4 is closed; 2.5 is running.** A longer run on the wrong architecture would
waste GPU quota. A continued plain 1024 fine-tune of the corrected `ch2` seed
counts as completing 2.1/2.2, not as the 2.5 quality recipe.

### Stage 2.5 — running experiment contract (resumable)

**Goal:** test whether a longer paired 1024 fine-tune with stain-colour and
sharpness losses plus val-based selection beats the current reverse baseline
(FID 282.92 / LPIPS 0.5317 at 1024; 285.29 / 0.5129 at 256). Reverse first;
forward only after the reverse result is understood.

- **Seed:** `resources/models/checkpoints/dapi2he_ch2_1024/checkpoint_last.pt`
  SHA-256 `5f930960…ba870f98` (epoch 1). Do not reseed from CycleGAN.
- **Data:** `hemit_train_1024.h5` only; fixed 50-crop val from
  `hemit_val_1024.h5`; test 945 touched once after selection.
- **Config:** `configs/hemit_dapi2he_quality_1024.yaml` — 4 epochs, batch 1,
  grad-accum 8, lr 3e-5, fp32; GAN 3 / L1 50 / SSIM 10 / colour 20 / gradient
  25; DISTS 0 (offline VGG risk); `val_h5`, `val_num: 50`, `val_every: 1`.
- **Selection:** fixed val `score = L1 + 0.25 × (1 − SSIM)`; save
  `checkpoint_best.pt` + `generator_best.pt`; final FID/LPIPS on best and last
  at 1024, 50 metric crops; choose by FID then LPIPS.
- **Compute:** ~14,868 microbatches (~1,859 optimizer steps). Measured 1024
  fine-tune speed on the T4 was ~0.53 s/microbatch → **~2.2 h train + ~0.5 h
  eval**. Stop at 4 h, OOM, CUDA unavailable, or no epoch progress in 20 min;
  keep partial checkpoints.
- **Kernels:** `bikasash/he-dapi-paired-stage25-reverse` (train, internet off)
  and `bikasash/he-dapi-paired-stage25-eval` (internet on for FID/LPIPS).
  Staged under `/home/bikash/kaggle_stage/stage25_{code,seed,kernel,eval}/`.
- **Gate:** lower val FID **and** LPIPS than the baseline, no obvious colour
  collapse/hallucination; otherwise record a negative result and do not promote.
  Full 630/945 + CIs remain required for v0.2.0.

### Immediate stage (2.3) — exact experiment contract (CLOSED, kept as history)

**Result 2026-10-23 12:13 UTC:** R-007 completed but is invalid for selection
(no FID/LPIPS; Tanh/[0,1] mismatch). The contract below is historical; do not
relaunch it.

- **Hypothesis H0:** with H5-only HEMIT data and identical 256px evaluation,
  CycleGAN is competitive with the paired reverse baseline on FID/LPIPS.
- **Data:** `hemit_train_256.h5` only (train split); `hemit_val_256.h5` for
  evaluation. No `data/examples`, no MHIST, no val/test leakage.
- **Model:** CycleGAN, ResNet-9 generators, PatchGAN discriminators, as in
  `configs/cyclegan_hemit_dapi2he_kaggle.yaml`.
- **Compute:** Kaggle T4, 10 epochs, batch 4 × grad-accum 2, AMP on.
- **Primary metrics:** FID and LPIPS on the **first 50** val crops
  (`eval_offline --metric-num`, default 50), matching the paired reference;
  PSNR/SSIM/L1 cover all 630 rows. Use a bootstrap CI; a <1–2 % gap is not a
  win. **Secondary:** color/contrast stats and a qualitative grid.
- **Paired reference (corrected ch2):** `dapi2he_ch2_1024` at 256px, 50 val
  crops: **FID 285.29, LPIPS 0.5129, PSNR 15.45, SSIM 0.4772, L1 0.1548**
  (`docs/metrics_ch2_1024_reverse_256_val50.json`). The old
  `342.98 / 0.5904` reference is **ch0 panCK and must not be used**.
- **Decision rule:** if CycleGAN has lower FID **and** LPIPS (or a clearly
  better blinded qualitative result at similar FID) with no hallucination
  pattern, it advances to 2.4. Otherwise the paired U-Net remains default and
  the next GPU spend goes to 2.5 forward/reverse quality, not more CycleGAN.
- **Stop conditions:** data log shows example PNGs or `synthetic=True`; no
  epoch progress in 20 min; OOM; CUDA unavailable. Stop and document, do not
  silently relaunch.
- **Budget:** ≤1.5 GPU-hours for the crop + 10 epochs + eval.
- **Artifacts:** checkpoint SHA-256, `eval_cyclegan.json`, qualitative grid,
  `LIVE_STATE.md` entry with the exact kernel ref/time.

---

## 3. Current truth (verified, not assumed)

### 3.1 Data
- HEMIT: 5,292 aligned pairs, splits 3,717 / 630 / 945.
- **Channel audit (2026-10-23): the raw RGB TIFF is `ch2=DAPI` (blue),
  `ch0=panCK` (red), `ch1=CD3` (green). ADR-014 accepted. The old 512 H5 used
  ch2, so the legacy model/public weights are the real **H&E→DAPI** baseline;
  the 1024 `ch0` models are the panCK ablation.** The 256 H5 has been rebuilt
  and verified on ch2; the corrected 1024 ch2 run is complete.
- "Corrected" 1024 H5 (currently `ch0`/panCK): train 11.5GB, val 2.1GB, test
  3.0GB. Must be rebuilt with `dapi=mihc[...,2]` before any DAPI run.
- Derived 256 H5 (center crop, LZF): train 596MB, val 104MB, test 148MB.
- H5 is authoritative. `UnpairedDataset` must show
  `HE: 0 files + H5 …` and `synthetic=False`; any example fallback is a bug.

### 3.2 Measured baselines (full val/test unless noted)
**Frozen numbers below are `ch0`=panCK, not DAPI. Do not cite them as DAPI.**

| Run | Target | Val SSIM | Val Pearson | Val PSNR | Test SSIM | Test Pearson | Test PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| Forward paired 1024 | panCK `ch0` (invalid for DAPI) | 0.670 | 0.255 | 28.77 | 0.725 | 0.611 | 26.60 |
| Reverse paired 1024 | H&E from panCK `ch0` (invalid for DAPI) | 0.437 | 0.521 | 15.18 | 0.593 | 0.576 | 15.79 |
| MHIST CycleGAN 256 | MHIST H&E (DAPI side `ch0`=panCK; confounded) | 0.232 | — | 10.76 | — | — | — |

FID/LPIPS at 256 (50 val crops unless stated):
- Paired reverse (`ch0` panCK): FID 342.98, LPIPS 0.5904.
- MHIST CycleGAN: FID 362.34, LPIPS 0.6999 — **confounded target domain**, not
  a fair unpaired verdict.
- Forward baseline at 256 (`ch0` panCK): FID 48.52, LPIPS 0.1145.
- R-006 HEMIT CycleGAN 256 (`ch0` panCK): FID 367.97, LPIPS 0.5948. Invalid
  for DAPI.

### 3.3 Product surface
- FastAPI `/health`, `/models`, `/translate`, `/translate/wsi`; React UI;
  Docker staging; PyPI `he-dapi` + `napari-he-dapi`; HF demo.
- Direction router `detect_direction` (threshold 0.06) exists.
- ONNX FP32 CPU benchmark recorded (`docs/BENCHMARK.md`).
- **Verified (2026-10-23):** the legacy 512 public weights/demo are the real
  H&E→DAPI model (`ch2`): H5 target array-equal to `mihc[...,2]`, HF SHA
  `c87751…e2afdc`, 46/46 generator tensors match local epoch169. The 1024
  `ch0` models are panCK and must not be presented as DAPI.

### 3.4 Compute
- Local: 3.7GB RAM, 1GB swap, CPU only. Full 4-network training is unsafe.
  Local is for tests, data prep, smoke, evaluation of small checkpoints.
- Kaggle: T4 15.64GB verified. Use it for training. Quota is finite; every run
  needs a budget and a stop rule.
- The 2026-10-22 WSL crash was host-side RAM/commit exhaustion from
  batch 2 + 2 workers + 4 networks, not a Python exception.

---

## 4. Decision log (ADR-style)

Every material decision gets an ID, date, context, decision, why, and
consequences. Add new entries; do not rewrite history.

### ADR-001 — HEMIT channels: ch0=DAPI, ch2=panCK (2026-10-20) — **SUPERSEDED by ADR-014**
- **Status:** superseded 2026-10-23; retained for history. The decision below is
  wrong for the raw RGB TIFF.
- **Context:** the old builder stored `mihc[...,2]`; labels are mIHC.
- **Decision:** DAPI target is `ch0` in all builders/loaders/tests.
- **Why:** ch2 is panCK; using it invalidates every DAPI claim.
- **Consequences (historically recorded, now known false):** this ADR claimed
  the old 512 checkpoints were a panCK ablation and the 1024 `ch0` H5 was
  correct. In fact the 512 model is DAPI and the 1024 `ch0` is panCK; see
  ADR-014.

### ADR-002 — H5 is authoritative; no example fallback (2026-10-22)
- **Context:** the old unpaired loader fell back to `data/examples` and mixed
  ~50% repeated example PNGs when `he_root` was absent.
- **Decision:** when `he_h5`/`dapi_h5` is explicit, use only H5. The loader may
  not glob roots, synthesize `[None]*8`, or sample examples.
- **Why:** the Kaggle 14:23 code bundle silently contaminated a run with 2–3
  example images; comparisons would be meaningless.
- **Consequences:** regression test added; local log must show
  `HE: 0 files + H5 …`. Any run without that line is invalid.

### ADR-003 — 256px center crop for the unpaired fair comparison (2026-10-22)
- **Context:** the paired reverse reference is evaluated at 256px; native 1024
  gzip reads are ~6s/step on CPU and slow even on T4.
- **Decision:** derive 256px H5 by center crop (LZF) and train/eval CycleGAN at
  256px; keep 1024 H5 for the paper-level quality pass.
- **Why:** same target resolution as the paired reference; ~15× fewer bytes;
  fits the 1.5 GPU-hour budget.
- **Consequences:** results are a fair **256px** comparison, not a 1024px paper
  claim; this limitation must appear in the selection memo.

### ADR-004 — Kaggle T4 is the training plane; local CPU is not (2026-10-22)
- **Context:** repeated WSL host kills; no local GPU.
- **Decision:** train on Kaggle T4; use local CPU only for tests, data prep,
  smoke, and small evals.
- **Why:** 3.7GB RAM cannot hold 4 networks + activations; Kaggle T4 is verified.
- **Consequences:** every training run needs a Kaggle dataset/code bundle,
  provenance, and a `LIVE_STATE.md` entry.

### ADR-005 — Local safety profile if CPU training is unavoidable (2026-10-22)
- **Context:** batch 2 + 2 workers + 4 nets crashed the WSL VM.
- **Decision:** `batch_size=1`, `grad_accum=8`, `num_workers=0`,
  `mixed_precision=false`, `grad_checkpoint=true`, cgroup `MemoryMax=2.5G`,
  auto-resume with backoff. Never start when `SwapFree < 300MB`.
- **Why:** keeps effective batch 8 while cutting activation and worker copies.
- **Consequences:** ~16–18s/step, ~14h for 10 epochs; a signal run, not a
  publishable result.

### ADR-006 — Cancel the 2026-10-22 14:25 Kaggle run; do not use its metrics (2026-10-22)
- **Context:** it mounted the 14:23 code bundle with the old loader and
  `data/examples`, so both domains mix 2–3 example PNGs ~50% of the time.
- **Decision:** stop it and relaunch with the corrected code/dataset.
- **Why:** data provenance is not negotiable; a contaminated run is worse than
  no run.
- **Consequences:** ~2 GPU-hours sunk; corrected 256 kernel staged. Audit trail:
  script preserved in `scripts/kaggle/`, evidence in `LIVE_STATE.md` §4.

### ADR-007 — CycleGAN first for the reverse unpaired baseline (2026-10-22)
- **Context:** DAPI→H&E has no paired data in the real world; HEMIT gives a
  paired reference for a fair comparison.
- **Decision:** implement CycleGAN (ResNet-9 + PatchGAN) as the first unpaired
  baseline; keep CUT/SACSI-CUT as the next candidate only if hallucination or
  FID is unacceptable.
- **Why:** CycleGAN is the standard, well-understood baseline; CUT adds
  complexity that is hard to justify before a baseline exists.
- **Consequences:** if CycleGAN loses to paired, do not automatically try CUT;
  first decide whether the product needs unpaired at all.

### ADR-008 — Select on val; touch test once (2026-10-22)
- **Context:** the quality-pass experiments improved train loss while val
  FID/SSIM worsened; test was being reported casually.
- **Decision:** all architecture/loss/checkpoint choices use val only. Test is
  evaluated once for the selected model, with the same protocol.
- **Why:** prevents test-set overfitting and inflated claims.
- **Consequences:** selection memos must state the val metric used and the
  checkpoint SHA; test numbers appear only in the final results table.

### ADR-009 — Release gate requires corrected weights + one checkpoint everywhere (2026-10-22)
- **Context:** packages/demo still carry the panCK ablation; docs and weights
  diverged.
- **Decision:** no v0.2.0 release until both directions have corrected weights,
  full metrics, SHA manifests, and the same SHA in API/UI/Docker/HF/packages.
- **Why:** a product that ships the wrong target channel is a trust failure.
- **Consequences:** public status remains alpha; no new model release from
  uncorrected runs.

### ADR-010 — Research-use only, synthetic watermark, no diagnostic claim (2026-10-22)
- **Context:** DAPI→H&E is generative and can hallucinate.
- **Decision:** all outputs are labeled synthetic/research-use; UI/API add a
  watermark/disclaimer; no clinical claims.
- **Why:** legal/ethical risk and scientific honesty.
- **Consequences:** model card and UI copy must carry the warning; production
  promotion remains manual.

### ADR-011 — No arbitrary launches (2026-10-22)
- **Context:** runs were launched while stale, contaminated, or without a
  recovery plan.
- **Decision:** every long run needs, before launch: hypothesis, config path +
  hash, data provenance, compute/budget, stop conditions, expected artifacts,
  and a `LIVE_STATE.md` entry. After exit/crash: update the same entry.
- **Why:** reproducibility and crash recovery; any LLM must be able to resume.
- **Consequences:** a run without a registry entry is treated as invalid.

### ADR-012 — Git + provenance discipline (2026-10-22)
- **Context:** code and docs diverged; the running Kaggle bundle was stale.
- **Decision:** commit code and docs together; each run records git SHA, config
  hash, data file, checkpoint SHA-256, and Kaggle ref/version. Never commit
  secrets; tokens stay in `$HOME`.
- **Why:** the bundle, not the working tree, is what trains.
- **Consequences:** before any push, verify the bundle contents
  (`datasets.py` fixed, no `data/`), not just the local files.

### ADR-013 — A kernel wrapper is not a run until it has a main guard and logs (2026-10-22)
- **Context:** the corrected Kaggle wrapper defined `main()` but omitted
  `if __name__ == "__main__": main()`. v1/v2 showed `COMPLETE` in ~6s, no logs,
  no output, and no GPU usage. The status looked successful.
- **Decision:** every Kaggle wrapper must end with the main guard and be
  smoke-tested locally (`python -m py_compile`, run with `--help`/dry path).
  A `COMPLETE` status with no log/output/GPU delta is treated as a failed run.
- **Why:** status alone is not evidence; the queued/complete state can hide a
  wrapper that never called the work.
- **Consequences:** v3 added the guard and was confirmed by quota consumption;
  future kernel pushes check `kaggle quota` + logs/output, not just status.

### ADR-014 — HEMIT RGB TIFF: ch2=DAPI, ch0=panCK (ACCEPTED 2026-10-23)
- **Status:** accepted; reverses ADR-001 and supersedes it.
- **Context:** pixel morphology, channel/H&E correlation, the legacy 512 H5
  (`dapi == mihc[...,2]`), and the paper's colour convention ("DAPI blue, panCK
  red, CD3 green") all show the raw RGB TIFF stores DAPI in the blue channel
  (`ch2`). The official `post_process.py` comment (`ch0 #dapi`) is inconsistent
  with the raw files.
- **Decision:** use `DAPI_CHANNEL = 2`, `panCK = 0`, `CD3 = 1`; rebuild the
  1024/256 H5 `dapi` from `mihc[...,2]`; re-label the legacy 512 model as DAPI
  and the 1024 `ch0` models as the panCK ablation.
- **Why:** the current 1024 "DAPI" models and R-006 are actually panCK→H&E; the
  legacy 512 model's metrics (test SSIM 0.789) match the paper's DAPI target.
- **Consequences:** invalidate all ch0 DAPI claims; re-run the reverse
  comparison on ch2 if it is still a product requirement; correct public
  weights/model cards only after verifying their source checkpoints.

---

## 5. Run registry (append-only)

| ID | Date | Where | Purpose | Status | Evidence |
|---|---|---|---|---|---|
| R-001 | 2026-10-18/20 | local CPU | H&E→panCK 200 epochs (ablation) | complete, invalid target | `docs/ABLATION_PANCK.md` |
| R-002 | 2026-10-21 | Kaggle T4 | forward DAPI paired 512→1024 | complete, undertrained | `docs/RESULTS.md` |
| R-003 | 2026-10-21/22 | Kaggle T4 | reverse H&E paired 512→1024 | complete, undertrained | `docs/RESULTS.md` |
| R-004 | 2026-10-22 | Kaggle T4 | CycleGAN MHIST H&E | complete, negative/confounded | `docs/metrics_dapi2he_cyclegan_256.json` |
| R-005 | 2026-10-22 14:25 | Kaggle T4 | CycleGAN HEMIT (stale bundle) | **INVALID, cancelled** | `LIVE_STATE.md` §4 |
| R-006 | 2026-10-22 17:20 | Kaggle T4 | CycleGAN HEMIT 256 | **COMPLETE 18:55 UTC; INVALID target (ch0 panCK)** | `bikasash/he-dapi-cyclegan-hemit-dapi2he-256`; 10 epochs, exit 0; metrics not usable for DAPI |
| R-007 | 2026-10-23 10:38 UTC | Kaggle T4 | CycleGAN HEMIT **ch2** 256 fairness run | **COMPLETE 12:13 UTC; INVALID for selection** | kernel `bikasash/he-dapi-cyclegan-ch2-256-r-007`; 10/10 epochs, exit 0; no FID/LPIPS (kernel omitted `--metrics`), Tanh/[0,1] mismatch; `LIVE_STATE.md` §0.5 |

---

## 6. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| WSL host OOM | lost run / corrupt state | Kaggle for training; local cgroup + swap check; no local 4-net runs |
| Contaminated data (examples/leakage) | invalid science | H5-authoritative loader; log assertion; regression test; bundle audit |
| Kaggle quota exhaustion | schedule slip | budget per run; stop rules; cancel invalid sessions; 1 run at a time |
| Unpaired metrics misleading | wrong model choice | FID/LPIPS primary; blinded qualitative; val-only selection |
| Reverse hallucination | clinical/trust risk | research-use labels, watermark, qualitative audit, uncertainty map |
| 256px crop hides context | weaker product | 1024 quality pass after architecture selection |
| Stale code bundle | silent wrong run | build bundle from committed SHA; inspect zip before push |
| Public weights are panCK | credibility | keep alpha; no DAPI claim; replace at v0.2.0 gate |
| Token exposure | account risk | rotate Kaggle token; keep 600 files; never echo |

---

## 7. Git / documentation policy

- Canonical plan: this file. Current run: `docs/LIVE_STATE.md`. Results:
  `docs/RESULTS.md`. Short status: `docs/TRACKER.md` + `docs/progress.json`.
- Commit code + docs together with a message stating the stage and gate.
- Each run entry includes: git SHA, config path/hash, data files, Kaggle
  ref/version, checkpoint path + SHA-256, start/end UTC, stop reason.
- Do not commit `data/`, checkpoints, tokens, or Kaggle secrets.
- `AGENTS.md` rules still apply: manual production approval; no unprompted
  commits except when the user asks; never disturb a running job.

---

## 8. Next actions (in order)

1. ~~**Recover host disk first.**~~ Done 2026-10-23 10:37 UTC; C: ~9.9 GB
   free at 12:20 UTC (pagefile-dependent), preflight `SAFE`; see
   `LIVE_STATE.md` §0.4.
2. ~~**Push and finish R-007**~~ Done 10:38–12:13 UTC; 10/10 epochs, exit 0.
   The run is **invalid for selection**: no FID/LPIPS and a Tanh/[0,1] mismatch.
   Do not rerun CycleGAN; paired U-Net remains default.
3. ~~**Write the selection memo**~~ Done 12:20 UTC:
   `docs/selection_reverse.md` records R-007 as invalid and selects the paired
   reverse `dapi2he_ch2_1024` baseline for Stage 2.5. No bootstrap CI is
   possible without valid R-007 metrics.
4. **Stage 2.5 is running** (pushed 12:56 UTC): `train_paired.py` now does
   fixed-subset validation and best-checkpoint selection; eval requests
   FID/LPIPS. Let the first attempt save epoch checkpoints, then after the
   2026-10-24 00:00 UTC reset pull the output, create a resume dataset, and
   re-push `bikasash/he-dapi-paired-stage25-reverse`; then the eval kernel. Do
   not start CUT or another CycleGAN run by default.
5. Align the public/API/Docker checkpoint SHA before v0.2.0: the HF/package
   artifact is epoch169, while the API/Docker local fallback is epoch200.
   Recommended: publish epoch200 (`/home/bikash/hf_stage/epoch200/`) and update
   `WEIGHT_FILES`, with user approval. Until then, do not claim one model.
6. Update README status table (direction router and CycleGAN are implemented;
   public weights are the legacy 512 DAPI model; no bidirectional/release claim
   until v0.2.0) before any new claim.
7. Full 630/945 + CIs and SHA manifests are required for the v0.2.0 gate; the
   50-crop selection is not a release claim.
8. Rotate the Kaggle token that was exposed during auth debugging.
