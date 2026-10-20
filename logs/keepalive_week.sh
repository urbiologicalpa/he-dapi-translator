#!/bin/bash
# keepalive_week.sh — 1-week supervisor: restarts train_paired if dead, every 300s
cd /home/bikash/he-dapi-translator || exit 1
while true; do
  if [ -f logs/train_done.flag ]; then
    echo "[KEEPALIVE $(date -u +%Y-%m-%dT%H:%M:%SZ)] TRAIN DONE flag present -> exiting supervisor" >> logs/keepalive.log
    exit 0
  fi
  if ! pgrep -f "train_paired.*hemit_dual_branch" > /dev/null 2>&1; then
    echo "[KEEPALIVE $(date -u +%Y-%m-%dT%H:%M:%SZ)] TRAIN DEAD -> relaunching epoch resume auto" >> logs/keepalive.log
    nohup nice -n 10 env PYTHONPATH=. .venv/bin/python -u -m src.models.train_paired --config configs/hemit_dual_branch.yaml --resume auto --optimize-for lowram >> logs/train_paired.log 2>&1 < /dev/null 9>&- &
    echo "[KEEPALIVE] new PID $!" >> logs/keepalive.log
  fi
  if ! pgrep -f "watch_runs.sh" > /dev/null 2>&1; then
    echo "[KEEPALIVE $(date -u +%Y-%m-%dT%H:%M:%SZ)] WATCH DEAD -> relaunching watch_runs.sh (300s week-mode)" >> logs/keepalive.log
    setsid nohup bash watch_runs.sh >> logs/monitor.log 2>&1 < /dev/null &
    echo $! > logs/watch.pid
    echo "[KEEPALIVE] watch PID $!" >> logs/keepalive.log
  fi
  sleep 300
done
