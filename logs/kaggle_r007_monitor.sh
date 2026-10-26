#!/bin/bash
set -u
cd /home/bikash/he-dapi-translator || exit 1
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
K=bikasash/he-dapi-cyclegan-ch2-256-r-007
LOG=logs/kaggle_r007_monitor.log
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  st=$(.venv/bin/kaggle kernels status "$K" 2>/dev/null | tail -1)
  q=$(.venv/bin/kaggle quota 2>/dev/null | awk '/^GPU/{print $3}')
  echo "$ts status=$st gpu_used=$q" >> "$LOG"
  case "$st" in
    *COMPLETE*|*ERROR*|*FAILED*|*CANCEL*)
      .venv/bin/kaggle kernels logs "$K" > logs/kaggle_r007_final.log 2>&1
      if bash scripts/crash_guard.sh --preflight >/dev/null 2>&1; then
        .venv/bin/kaggle kernels output "$K" -p /home/bikash/kaggle_r007_output > logs/kaggle_r007_pull.log 2>&1
        echo "$ts terminal -> logs+pull done" >> "$LOG"
      else
        echo "$ts terminal -> logs saved; PULL DEFERRED (preflight not SAFE)" >> "$LOG"
      fi
      break ;;
  esac
  sleep 120
done
