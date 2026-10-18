# AGENTS.md — operating rules for this repo

## Production deployments (IMPORTANT)
- **Do NOT auto-approve the `production` environment.** Leave every `cd-production`
  run in the "Waiting for approval" state so **urbiologicalpa** clicks
  *Actions → run → Review deployments → (tick) production → Approve and deploy*.
- The user prefers to retain manual approval of all production promotes/pushes.
- (One-time exception, 2026-10-18: v0.1.0 promote was approved via API to validate the
  fixed path. Do not repeat without an explicit request.)

## Conventions
- No `gh` CLI on this box — use the GitHub REST API with the token at `~/.github_pat` (600).
- HF token at `~/.hf_token` (user `urbiologicalpa`). Kaggle: **STAND DOWN** (`~/.kaggle/api_token`).
- PyPI token at `~/.pypi_token` (600) + `~/.pypirc`. Publish with
  `uv publish --token "$(cat ~/.pypi_token)" packages/<pkg>/dist/*` (build first with `uv build`).
  `he-dapi` and `napari-he-dapi` are published on PyPI (v0.1.1, **alpha** — `Development Status :: 3 - Alpha`).
  Bump `version` in `packages/*/pyproject.toml` (+ `__init__.__version__`) before republishing.
- Training ground truth = checkpoint files in `resources/models/checkpoints/he2dapi/`,
  not the tracker docs. Do not disturb a running training job (PID via `pgrep -f train_paired`).
- Never commit secrets; keep tokens only in `$HOME` dotfiles.
- Only commit/push when explicitly asked.
