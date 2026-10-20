# Skill: orchestrator

You coordinate all sub-agents and ensure no crash on this 3.7GB box.

## Workflow (resumable, spike-guarded)

1. **Data** (`02`): `scripts/data/pack_hemit.py` streaming combined (peak 140MB). Resumable: skip if `he.shape[0]==len(pairs)`, atomic `.tmp→replace`, flush 500.
2. **Dataset** (`03`): lazy H5 SWMR per worker, subset support.
3. **Tiling** (`04`) + **WSI** (`05`): streaming merge to TIFF, not full canvas.
4. **Training** (`06`): batch 2 + grad_accum 4, DISTS off on CPU, num_workers 0, OOM try/catch.
5. **Inference** (`07`): stream if est>512MB, thumbnail guard.
6. **Backend** (`08`): 429 on `used_pct>85`.

## Orchestration files
- `Makefile` (targets: data, train-paired, train-unpaired, infer, health, clean)
- `wiki/README.md` (map)
- `docs/optimization/REPORT.md` (summary)

## Commands
```bash
make data       # pack hemit/bci streaming
make train-paired-dummy  # 4 samples, batch 2, checks monitor
make infer      # patch demo
make health     # resource check
```

## Recovery
Every long step is `nohup + tee + checkpoint`:
```bash
nohup .venv/bin/python scripts/data/pack_hemit.py 2>&1 | tee /tmp/pack_hemit.log &
# resume: re-run same command (skip if complete)
```

## Token & push-auth map (reboot-safe — lesson 2026-10-14, do NOT re-break)
All tokens live as `600` files in `$HOME` (outside repo — CI secret-scan blocks
`ghp_/KGAT_` patterns in tracked files). Exact names matter: on 2026-10-14 a push
failed 3x because the agent looked for `~/.git_pat` while the real file is
`~/.github_pat`. Always `ls ~` first, never guess the filename.

| Token | File (`chmod 600`) | Used by |
|-------|--------------------|---------|
| GitHub PAT | `~/.github_pat` | `git push` via credential helper (below) |
| Kaggle API | `~/.kaggle/api_token` | `watch_runs.sh` via `KAGGLE_API_TOKEN=$(cat ~/.kaggle/api_token)` |
| HuggingFace | `~/.hf_token` | manual HF uploads only |

```bash
ls -l ~/.github_pat ~/.kaggle/api_token ~/.hf_token   # verify exist, mode 600
# one-time: wire GitHub PAT into git (survives reboot, never commit it)
git config credential.helper '!f() { echo username=urbiologicalpa; echo password=$(cat ~/.github_pat); }; f'
git push origin main   # must print "main -> main"; if "Authentication failed", re-check filename/perms
```

## Reboot recovery (week-mode relaunch order — lesson 2026-10-14)
0. **Auto (since 2026-10-17)**: cron `@reboot` + `*/5` runs `logs/boot_launch.sh` (idempotent, flock-guarded). It relaunches `keepalive_week.sh` → train + watch if dead, so reboots recover with no manual step. Install: `crontab <(printf '@reboot /bin/bash %s/logs/boot_launch.sh\n*/5 * * * * /bin/bash %s/logs/boot_launch.sh\n' "$PWD" "$PWD")`. Verify: `tail -5 logs/boot.log`.
1. Ground truth = `resources/models/checkpoints/he2dapi/checkpoint_epoch*.pt` count + `checkpoint_last.pt` (`torch.load` → `epoch`/`loss`); tracker files are secondary.
2. Manual fallback — relaunch train FIRST: `setsid nohup nice -n 10 env PYTHONPATH=. .venv/bin/python -u -m src.models.train_paired --config configs/hemit_dual_branch.yaml --resume auto --optimize-for lowram >> logs/train_paired.log 2>&1 < /dev/null &` — confirm `step20` line appears (~4 min). Or just run `bash logs/boot_launch.sh`.
3. Then watch (`setsid nohup bash watch_runs.sh ...`, pid → `logs/watch.pid`) + keepalive (`setsid nohup bash logs/keepalive_week.sh ...`, pid → `logs/keepalive.pid`).
4. Keepalive restarts BOTH train and watch if dead (watch-guard added 2026-10-14 after watch died silently 1h pre-reboot). After editing it, kill + relaunch keepalive to reload.
5. Sync triple: `docs/progress.json:runs` + `docs/TRACKER.md` (Live Runs + Daily Log) + `MEMORY.md` banner, then commit + push (auth above).

## Wiki skill invocation
Each folder `wiki/0X-*/SKILL.md` is agent prompt. To re-apply: `cat wiki/0X-*/SKILL.md` + edit linked file.

## CI/CD (GitHub Actions → GHCR) — audit/fix 2026-10-18
Repo `urbiologicalpa/he-dapi-translator`, image `ghcr.io/urbiologicalpa/he-dapi-translator`.

| Workflow | Trigger | Output |
|----------|---------|--------|
| `guards` (`ci.yml`) | push `main`/`staging`, PRs | secret-scan, low-RAM enforce, `tests/test_resource_guards.py`, backend `/health` smoke |
| `cd-staging` (`cd-staging.yml`) | push `main`/`staging` **paths-filtered** (`docker/**`,`src/**`,`resources/**`,`data/examples/**`,`.dockerignore`,workflow) + `workflow_dispatch` | build immutable `:$(git SHA)` + `:staging` → GHCR, smoke `/health`+`/models` |
| `cd-production` (`cd-production.yml`) | tag `v*.*.*`, release, `workflow_dispatch` | promote same SHA → `:version` + `:latest`; **requires `production` env approval** |

**Rules / lessons:**
- `.dockerignore` excludes `resources/models/checkpoints` (14GB!), `data/raw|processed`, `.git`, `.venv`, `logs`, `wiki`, `docs`, `*.pt/h5/svs/7z`. CI was only fast because checkpoints are gitignored — never remove this file or local/CI builds balloon.
- Training-sync commits (docs/ckpt only) do **not** trigger `cd-staging` (paths filter) → no wasted heavy builds. Code/docker changes do.
- `main` branch protection: force-push + deletion blocked, `resource-guards` check required for PRs (direct pushes from the training loop still work). Set via API 2026-10-18.
- Refresh staging image now: `curl -X POST -H "Authorization: Bearer $(cat ~/.github_pat)" .../actions/workflows/cd-staging.yml/dispatches -d '{"ref":"main"}'`. Verify tag: `GET /users/urbiologicalpa/packages/container/he-dapi-translator/versions`.
- Release: `git tag vX.Y.Z && git push origin vX.Y.Z`, then approve the `production` environment in the Actions UI. GHCR serves images; no registry creds in repo (`secrets.GITHUB_TOKEN`).
- No `gh` CLI on this box — use the REST API with `~/.github_pat` (`repo`+`workflow`+`write:packages` scopes).
