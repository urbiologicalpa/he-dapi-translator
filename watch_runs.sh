#!/bin/bash
# watch_runs.sh — 300s loop eye on local + Kaggle runs. Logs to logs/monitor.log
# Stop with: kill $(cat logs/watch.pid)
# Override the Kaggle kernel with KAGGLE_KERNEL=owner/slug.
cd /home/bikash/he-dapi-translator || exit 1
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/api_token)
KERNEL="${KAGGLE_KERNEL:-bikasash/he-dapi-dapi2he-1024-eval}"
while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  if [ -f logs/train_done.flag ]; then
    echo "[$TS] TRAIN: COMPLETE ($(cat logs/train_done.flag)) | watcher exiting"
    exit 0
  fi
  KAGGLE_STATUS=$(timeout 20 .venv/bin/python -m kaggle kernels status "$KERNEL" 2>&1 | head -n 1)
  [ -z "$KAGGLE_STATUS" ] && KAGGLE_STATUS="KAGGLE_UNKNOWN_TIMEOUT"
  TRAIN_PID=$(pgrep -f "train_paired.*hemit_dual_branch" | head -n 1)
  if [ -n "$TRAIN_PID" ] && ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    LOCAL_STATE="PID${TRAIN_PID} alive $(ps -p "$TRAIN_PID" -o etime= | tr -d ' ')"
  else
    LOCAL_STATE="TRAIN DEAD (no train_paired process; no done flag)"
  fi
  LOCAL_TAIL=$(tail -n 1 logs/train_paired.log 2>/dev/null)
  echo "[$TS] KERNEL: $KERNEL | KAGGLE: $KAGGLE_STATUS | LOCAL: $LOCAL_STATE | LOG: $LOCAL_TAIL"
  sleep 300  # WEEK-MODE 2026-10-08: 50s->300s for 1-week continuous (6x fewer wakeups/clock jitter, 6x fewer Kaggle API calls)
done
