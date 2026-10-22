#!/usr/bin/env bash
# Append the corrected 256px Kaggle kernel status to a log until it exits.
# This is a status monitor only; it does not touch the GPU or local training.
set -u
cd "$(dirname "$0")/.." || exit 1

KERNEL="bikasash/he-dapi-cyclegan-hemit-dapi2he-256"
LOG="logs/kaggle_256_monitor.log"
mkdir -p logs

if [ ! -f "$HOME/.kaggle/api_token" ]; then
  echo "missing ~/.kaggle/api_token" >&2
  exit 1
fi
export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/api_token")"

while true; do
  ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  status="$(.venv/bin/kaggle kernels status "$KERNEL" 2>&1 | tail -1)"
  printf '%s %s\n' "$ts" "$status" >> "$LOG"
  case "$status" in
    *COMPLETE*|*ERROR*|*FAILED*|*CANCEL*) exit 0 ;;
  esac
  sleep 120
done
