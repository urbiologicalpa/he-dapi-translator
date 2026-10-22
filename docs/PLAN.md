# PLAN v2.0 — canonical product + research plan

**Status:** authoritative as of **2026-10-22 16:40 UTC**. Supersedes
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

| Stage | Gate artifact | Current state (2026-10-22) |
|---|---|---|
| 0. Truth & infra | CI green, Docker build, resource guards, channel regression tests | **DONE** |
| 1. Data contract | Corrected 1024 H5 for train/val/test, channel proof, no leakage, checksums | **DONE** |
| 2. Model baselines | Forward + reverse paired baselines; fair unpaired comparison; selection memo | **IN PROGRESS** |
| 3. Validation | Full-split metrics, CIs, qualitative/blinded audit, failure analysis, limitations | **PARTIAL** |
| 4. Product integration | Router + API/UI + WSI + ONNX + model manifest, same checkpoint everywhere | **ALPHA** |
| 5. Release | Staging UAT, manual prod approval, model card, rollback | **ALPHA** |
| 6. Post-release | Monitoring, drift/feedback, quota/cost review | **NOT STARTED** |

### Stage 2 breakdown (current critical path)

| Sub-stage | Hypothesis / purpose | Gate |
|---|---|---|
| 2.1 Forward paired baseline | U-Net can learn H&E→DAPI ch0 | **DONE, undertrained**; val SSIM 0.670, Pearson 0.255 |
| 2.2 Reverse paired baseline | U-Net can learn DAPI→H&E | **DONE, undertrained**; val SSIM 0.437, PSNR 15.18 |
| 2.3 Fair unpaired comparison | CycleGAN on **HEMIT** H&E (same target domain, same 256px eval) is competitive with paired reverse | **RUN NOW** (corrected 256 kernel) |
| 2.4 Reverse model selection | Pick paired vs unpaired using **val** FID/LPIPS + blinded qualitative audit + cost; test only once | Not started |
| 2.5 Quality pass | Longer 1024 schedule, val-based checkpoint selection, perceptual/color losses, dual-branch forward | Not started |

**Do not start 2.5 before 2.4.** A longer run on the wrong architecture is
wasted GPU quota.

### Immediate stage (2.3) — exact experiment contract

- **Hypothesis H0:** with H5-only HEMIT data and identical 256px evaluation,
  CycleGAN is competitive with the paired reverse baseline on FID/LPIPS.
- **Data:** `hemit_train_256.h5` only (train split); `hemit_val_256.h5` for
  evaluation. No `data/examples`, no MHIST, no val/test leakage.
- **Model:** CycleGAN, ResNet-9 generators, PatchGAN discriminators, as in
  `configs/cyclegan_hemit_dapi2he_kaggle.yaml`.
- **Compute:** Kaggle T4, 10 epochs, batch 4 × grad-accum 2, AMP on.
- **Primary metrics:** FID and LPIPS on 630 val crops (same protocol as the
  paired baseline). **Secondary:** PSNR/SSIM/L1 (diagnostic only for unpaired),
  color/contrast stats, qualitative grid.
- **Paired reference:** reverse 1024 baseline at 256px:
  FID 342.98, LPIPS 0.5904, PSNR 13.17, SSIM 0.4941, L1 0.2056.
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
- Labels: **ch0=DAPI, ch1=CD3, ch2=panCK**. The old 512 H5 used ch2 → the
  epoch-199/200 model, public HF weights, demo, and `metrics_eval.json` are a
  **panCK ablation**.
- Corrected 1024 H5: train 11.5GB, val 2.1GB, test 3.0GB.
- Derived 256 H5 (center crop, LZF): train 596MB, val 104MB, test 148MB.
- H5 is authoritative. `UnpairedDataset` must show
  `HE: 0 files + H5 …` and `synthetic=False`; any example fallback is a bug.

### 3.2 Measured baselines (full val/test unless noted)
| Run | Target | Val SSIM | Val Pearson | Val PSNR | Test SSIM | Test Pearson | Test PSNR |
|---|---|---:|---:|---:|---:|---:|---:|
| Forward paired 1024 | DAPI ch0 | 0.670 | 0.255 | 28.77 | 0.725 | 0.611 | 26.60 |
| Reverse paired 1024 | H&E | 0.437 | 0.521 | 15.18 | 0.593 | 0.576 | 15.79 |
| MHIST CycleGAN 256 | MHIST H&E | 0.232 | — | 10.76 | — | — | — |

FID/LPIPS at 256 (50 val crops unless stated):
- Paired reverse: FID 342.98, LPIPS 0.5904.
- MHIST CycleGAN: FID 362.34, LPIPS 0.6999 — **confounded target domain**, not
  a fair unpaired verdict.
- Forward baseline at 256: FID 48.52, LPIPS 0.1145.

### 3.3 Product surface
- FastAPI `/health`, `/models`, `/translate`, `/translate/wsi`; React UI;
  Docker staging; PyPI `he-dapi` + `napari-he-dapi`; HF demo.
- Direction router `detect_direction` (threshold 0.06) exists.
- ONNX FP32 CPU benchmark recorded (`docs/BENCHMARK.md`).
- **Caveat:** currently published weights/demo are the panCK ablation; must not
  be presented as DAPI. Corrected weights are internal and undertrained.

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

### ADR-001 — HEMIT channels: ch0=DAPI, ch2=panCK (2026-10-20)
- **Context:** the old builder stored `mihc[...,2]`; labels are mIHC.
- **Decision:** DAPI target is `ch0` in all builders/loaders/tests.
- **Why:** ch2 is panCK; using it invalidates every DAPI claim.
- **Consequences:** old 512 checkpoints/metrics/weights are a panCK ablation;
  corrected 1024 H5 was rebuilt and verified with `tests/test_hemit_channels.py`.

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

---

## 5. Run registry (append-only)

| ID | Date | Where | Purpose | Status | Evidence |
|---|---|---|---|---|---|
| R-001 | 2026-10-18/20 | local CPU | H&E→panCK 200 epochs (ablation) | complete, invalid target | `docs/ABLATION_PANCK.md` |
| R-002 | 2026-10-21 | Kaggle T4 | forward DAPI paired 512→1024 | complete, undertrained | `docs/RESULTS.md` |
| R-003 | 2026-10-21/22 | Kaggle T4 | reverse H&E paired 512→1024 | complete, undertrained | `docs/RESULTS.md` |
| R-004 | 2026-10-22 | Kaggle T4 | CycleGAN MHIST H&E | complete, negative/confounded | `docs/metrics_dapi2he_cyclegan_256.json` |
| R-005 | 2026-10-22 14:25 | Kaggle T4 | CycleGAN HEMIT (stale bundle) | **INVALID, cancelled** | `LIVE_STATE.md` §4 |
| R-006 | 2026-10-22 17:20 | Kaggle T4 | CycleGAN HEMIT corrected 256 | **RUNNING (v3)** | `bikasash/he-dapi-cyclegan-hemit-dapi2he-256`; code ds `bikasash/he-dapi-code-hemit-256`; v1/v2 no-op (missing main guard) |

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

1. **Stop R-005** (invalid Kaggle session) via the Kaggle UI, or confirm it
   exited. Do not launch R-006 while R-005 holds the T4.
2. **Launch R-006** (corrected 256 kernel) and verify the first log lines:
   `HE: 0 files + H5 …`, `synthetic=False`, `[LOWRAM]` absent on GPU,
   `cuda=True`.
3. After R-006: write `docs/selection_reverse.md` comparing paired vs unpaired
   on val FID/LPIPS + qualitative; pick the default; update RESULTS/TRACKER.
4. If paired wins: spend next GPU budget on 2.5 reverse quality (longer 1024,
   val-based selection). If unpaired wins: test CUT only if hallucination is the
   blocker, not by default.
5. Update README status table (direction router and CycleGAN are implemented;
   public weights remain the panCK ablation) before any new public claim.
6. Rotate the Kaggle token that was exposed during auth debugging.
