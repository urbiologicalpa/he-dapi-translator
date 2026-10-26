#!/bin/bash
# boot_launch.sh — idempotent reboot/periodic bootstrap for the 1-week run.
# Invoked by cron @reboot AND every 5 min. Ensures the keepalive supervisor
# (which manages train_paired + watch_runs) is alive, so the run survives
# reboots/hangs without manual help. No sudo / linger required.
set -u
cd /home/bikash/he-dapi-translator || exit 1
LOG=logs/boot.log
exec 9>/tmp/he_boot_launch.lock
flock -n 9 || exit 0
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# finished run: never relaunch once training wrote its completion flag
if [ -f logs/train_done.flag ]; then
  exit 0
fi

# keepalive supervises train + watch; if it is alive, nothing to do
if pgrep -f "keepalive_week.sh" > /dev/null 2>&1; then
  exit 0
fi
echo "[$(ts)] BOOT: keepalive dead -> launching train+watch+keepalive" >> "$LOG"

# 0) hard host-safety gate: never relaunch into a full C:/swap/OOM state.
if ! bash scripts/crash_guard.sh --preflight >> "$LOG" 2>&1; then
  echo "[$(ts)] BOOT: BLOCKED by crash_guard preflight (disk/memory unsafe); see logs/crash_reports" >> "$LOG"
  exit 0
fi

# 1) train FIRST (resume auto from checkpoint_last.pt), week-mode low-RAM
if ! pgrep -f "train_paired.*hemit_dual_branch" > /dev/null 2>&1; then
  setsid nohup nice -n 10 env PYTHONPATH=. .venv/bin/python -u -m src.models.train_paired \
    --config configs/hemit_dual_branch.yaml --resume auto --optimize-for lowram \
    >> logs/train_paired.log 2>&1 < /dev/null 9>&- &
  echo "[$(ts)] BOOT: train PID $!" >> "$LOG"
  sleep 5
fi

# 2) watch (300s week-mode)
if ! pgrep -f "watch_runs.sh" > /dev/null 2>&1; then
  setsid nohup bash watch_runs.sh >> logs/monitor.log 2>&1 < /dev/null 9>&- &
  echo $! > logs/watch.pid
  echo "[$(ts)] BOOT: watch PID $!" >> "$LOG"
fi

# 3) keepalive LAST (supervises both train and watch every 300s)
setsid nohup bash logs/keepalive_week.sh >> logs/keepalive.log 2>&1 < /dev/null 9>&- &
echo $! > logs/keepalive.pid
echo "[$(ts)] BOOT: keepalive PID $!" >> "$LOG"
