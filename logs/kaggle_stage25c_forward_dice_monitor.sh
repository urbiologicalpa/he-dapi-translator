#!/bin/bash
# Monitor Stage 2.5c soft-Dice probe; filtered pull only after preflight is SAFE.
set -u
cd /home/bikash/he-dapi-translator || exit 1
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
LOG=logs/kaggle_stage25c_forward_dice_monitor.log
SLUG=bikasash/he-dapi-paired-stage25c-forward-dice
OUT=/home/bikash/kaggle_out_stage25c_forward_dice
PATTERN='.*(checkpoint_best|checkpoint_last|checkpoint_epoch|generator_best|generator_last|\.log$).*'
DONE=0
mkdir -p "$OUT"

while [ "$DONE" = 0 ]; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  st=$(.venv/bin/kaggle kernels status "$SLUG" 2>/dev/null | tail -1)
  echo "$ts stage25c=$st" >> "$LOG"
  case "$st" in
    *COMPLETE*|*ERROR*|*FAILED*|*CANCEL*)
      .venv/bin/kaggle kernels logs "$SLUG" > logs/kaggle_stage25c_final.log 2>&1
      if bash scripts/crash_guard.sh --preflight >/dev/null 2>&1; then
        if .venv/bin/kaggle kernels output "$SLUG" -p "$OUT" --file-pattern "$PATTERN" \
             > logs/kaggle_stage25c_pull.log 2>&1; then
          echo "$ts stage25c terminal -> logs+filtered pull done" >> "$LOG"
          DONE=1
        else
          echo "$ts stage25c terminal -> pull failed; retrying in 300s" >> "$LOG"
        fi
      else
        echo "$ts stage25c terminal -> PULL DEFERRED (preflight); retrying in 300s" >> "$LOG"
      fi ;;
  esac
  [ "$DONE" = 0 ] && sleep 300
done
