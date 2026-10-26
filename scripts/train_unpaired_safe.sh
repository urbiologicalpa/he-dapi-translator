#!/bin/bash
# train_unpaired_safe.sh — crash-safe launcher for the HEMIT-domain CycleGAN.
#
# Why this exists: on 2026-10-22 the old batch=2 / num_workers=2 profile
# hard-killed the whole WSL2 VM (4 networks + worker copies exceeded the 3.7GB
# RAM/commit). This launcher:
#   * runs batch=1 + grad_accum=8 + num_workers=0 (from the config),
#   * fences the trainer in a user cgroup (MemoryMax) so Linux kills the
#     trainer instead of the WSL VM if it ever exceeds the budget,
#   * resumes from checkpoint_last.pt after an OOM/reboot,
#   * backs off after repeated fast failures so a broken run cannot loop.
#
# Usage:
#   bash scripts/train_unpaired_safe.sh
#   MAX_STEPS=300 EPOCHS=10 bash scripts/train_unpaired_safe.sh
#
# Cron (idempotent, flock-guarded): @reboot and every 10 minutes.
set -u
cd "$(dirname "$0")/.." || exit 1

LOG=logs/train_unpaired.log
DONE=logs/unpaired_done.flag
LOCK=/tmp/he_train_unpaired_safe.lock
CFG=configs/cyclegan_hemit_dapi2he.yaml
CKPT=resources/models/checkpoints/hemit_cyclegan_dapi2he_256
MAX_STEPS="${MAX_STEPS:-300}"
EPOCHS="${EPOCHS:-}"

mkdir -p logs "$CKPT"
exec 9>"$LOCK"
flock -n 9 || exit 0

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

if [ -f "$DONE" ]; then
  echo "[$(ts)] DONE flag present -> exit" >> "$LOG"
  exit 0
fi
if pgrep -f "train_paired.*hemit_dual_branch" > /dev/null 2>&1; then
  echo "[$(ts)] paired training is running; refusing to launch unpaired" >> "$LOG"
  exit 0
fi
if pgrep -f "train_unpaired.*cyclegan_hemit_dapi2he" > /dev/null 2>&1; then
  echo "[$(ts)] unpaired trainer already running; exit" >> "$LOG"
  exit 0
fi

# systemd-run --user needs these in cron sessions too.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=$XDG_RUNTIME_DIR/bus}"

args=(--config "$CFG" --optimize-for lowram --resume auto --max-steps "$MAX_STEPS")
[ -n "$EPOCHS" ] && args+=(--epochs "$EPOCHS")

echo "[$(ts)] LAUNCH max_steps=$MAX_STEPS epochs=${EPOCHS:-config} cgroup=2500M/128M swap" >> "$LOG"
fails=0
while true; do
  if [ -f "$DONE" ]; then
    echo "[$(ts)] DONE flag present -> exit" >> "$LOG"
    exit 0
  fi
  if ! bash scripts/crash_guard.sh --preflight >> "$LOG" 2>&1; then
    echo "[$(ts)] BLOCKED by crash_guard preflight (unsafe host) -> retry in 300s" >> "$LOG"
    sleep 300
    continue
  fi

  start=$(date +%s)
  echo "[$(ts)] START $(printf '%q ' .venv/bin/python -u -m src.models.train_unpaired "${args[@]}")" >> "$LOG"
  systemd-run --user --quiet --scope \
    -p MemoryHigh=2200M -p MemoryMax=2500M -p MemorySwapMax=128M \
    -- bash -lc "cd '$PWD' && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=2 PYTHONPATH=. .venv/bin/python -u -m src.models.train_unpaired ${args[*]}" \
    >> "$LOG" 2>&1
  rc=$?
  dur=$(( $(date +%s) - start ))
  echo "[$(ts)] EXIT rc=$rc after ${dur}s" >> "$LOG"

  if [ "$rc" -eq 0 ]; then
    touch "$DONE"
    echo "[$(ts)] COMPLETE -> $DONE" >> "$LOG"
    exit 0
  fi

  # 137=SIGKILL (cgroup OOM), 143=SIGTERM. A run shorter than 10 min is a
  # crash-loop candidate; do not hammer it.
  if [ "$dur" -lt 600 ]; then
    fails=$((fails + 1))
  else
    fails=0
  fi
  if [ "$fails" -ge 3 ]; then
    echo "[$(ts)] ALERT: 3 fast failures (rc=$rc) -> 900s backoff" >> "$LOG"
    sleep 900
    fails=0
  else
    sleep 60
  fi
done
