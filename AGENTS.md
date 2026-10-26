# AGENTS.md — operating rules for this repo

## Production deployments (IMPORTANT)
- **Do NOT auto-approve the `production` environment.** Leave every `cd-production`
  run in the "Waiting for approval" state so **urbiologicalpa** clicks
  *Actions → run → Review deployments → (tick) production → Approve and deploy*.
- The user prefers to retain manual approval of all production promotes/pushes.
- (One-time exception, 2026-10-18: v0.1.0 promote was approved via API to validate the
  fixed path. Do not repeat without an explicit request.)

## Conventions
- **Host safety gate:** before any training launch or Kaggle output pull, run
  `bash scripts/crash_guard.sh --preflight`. Exit `2` = do not launch/pull.
  WSL `df` is misleading: the real ceiling is free space on Windows `C:`
  (the sparse `ext4.vhdx` cannot grow when `C:` is full). After any reboot or
  crash, run `bash scripts/crash_guard.sh --report` and read the newest file in
  `logs/crash_reports/` before resuming work.
- No `gh` CLI on this box — use the GitHub REST API with the token at `~/.github_pat` (600).
- HF token at `~/.hf_token` (user `urbiologicalpa`). Kaggle T4 is **active** (verified
  2026-10-22); use `KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"` with the `kaggle` CLI
  (Basic auth with the raw KGAT token fails). The old "STAND DOWN" note is obsolete.
- PyPI token at `~/.pypi_token` (600) + `~/.pypirc`. Publish with
  `uv publish --token "$(cat ~/.pypi_token)" packages/<pkg>/dist/*` (build first with `uv build`).
  `he-dapi` and `napari-he-dapi` are published on PyPI (v0.1.1, **alpha** — `Development Status :: 3 - Alpha`).
  Bump `version` in `packages/*/pyproject.toml` (+ `__init__.__version__`) before republishing.
- Training ground truth = checkpoint files in `resources/models/checkpoints/he2dapi/`,
  not the tracker docs. Do not disturb a running training job (PID via `pgrep -f train_paired`).
- **Long runs must be documented before launch and after every exit/crash** in
  `docs/LIVE_STATE.md`: exact command, PID/session, log path, checkpoint path,
  config hash/bundle path, resume command, and reason for stopping. Run
  `bash scripts/status.sh` and paste/verify the snapshot instead of assuming.
  Never relaunch local training while `SwapFree < ~300MB` (WSL host OOM).
- Never commit secrets; keep tokens only in `$HOME` dotfiles.
- Only commit/push when explicitly asked.

## Context / token management (IMPORTANT)
- Treat context as a budget. Routine status/monitoring tasks should stay small;
  implementation tasks should checkpoint state and split if they grow.
- **State cache first:** read `docs/LIVE_STATE.md`, `docs/progress.json`, and
  `docs/TRACKER.md` before exploring the repo. Update them after each major step
  so future sessions do not re-derive state.
- **Targeted reads only:** use `rg -n`, `sed -n 'A,Bp'`, `git diff --stat`,
  `git status --short`, and `--file-pattern` pulls. Avoid full-file dumps and
  full training logs.
- **Filter long outputs:** for Kaggle use
  `kaggle kernels logs -f <kernel> | grep -E '\[Epoch|\[VAL\]|\[BEST\]|\[DONE\]|Error' | tail -20`.
  Never paste a full log or a 1000-line tool result into chat.
- **Parallelize independent checks** (status + forward log + reverse log) in one
  message; do not re-read files already read in the same session.
- **Summarize, don't dump:** report only the metrics/decisions needed for the
  next action; keep prose short.
- **Handoff before a long run:** write the command, seed SHA, config, output
  path, monitor PID/log, and resume command to `docs/LIVE_STATE.md`; then launch.
- **Soft budget:** if a task is still exploring after ~50k tokens of tool output,
  stop, write a state note, and ask or split the task instead of continuing.
