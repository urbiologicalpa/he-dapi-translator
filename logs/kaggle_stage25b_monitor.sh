#!/bin/bash
# Monitor Stage 2.5b forward (H&E->DAPI) and reverse (DAPI->H&E) long runs.
# Pulls filtered checkpoints/logs when each kernel reaches a terminal state.
# The Stage 2.5 reverse eval kernel was cancelled (hung on FID/LPIPS weight
# download); its metrics were computed locally instead.
set -u
cd /home/bikash/he-dapi-translator || exit 1
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
LOG=logs/kaggle_stage25b_monitor.log
FWD=bikasash/he-dapi-paired-stage25b-forward
REV=bikasash/he-dapi-paired-stage25b-reverse
FWD_OUT=/home/bikash/kaggle_out_stage25b_forward
REV_OUT=/home/bikash/kaggle_out_stage25b_reverse
FWD_DONE=0
REV_DONE=0
PATTERN='.*(checkpoint_best|checkpoint_last|generator_best|generator_last|\.log$).*'

watch_one() {
  local name=$1 slug=$2 out=$3 donevar=$4
  local st ts
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  st=$(.venv/bin/kaggle kernels status "$slug" 2>/dev/null | tail -1)
  echo "$ts $name=$st" >> "$LOG"
  case "$st" in
    *COMPLETE*|*ERROR*|*FAILED*|*CANCEL*)
      .venv/bin/kaggle kernels logs "$slug" > "logs/kaggle_${name}_final.log" 2>&1
      if bash scripts/crash_guard.sh --preflight >/dev/null 2>&1; then
        mkdir -p "$out"
        .venv/bin/kaggle kernels output "$slug" -p "$out" --file-pattern "$PATTERN" \
          > "logs/kaggle_${name}_pull.log" 2>&1
        echo "$ts $name terminal -> logs+filtered pull done" >> "$LOG"
      else
        echo "$ts $name terminal -> logs saved; PULL DEFERRED (preflight)" >> "$LOG"
      fi
      eval "$donevar=1" ;;
  esac
}

while true; do
  [ "$FWD_DONE" = 0 ] && watch_one fwd "$FWD" "$FWD_OUT" FWD_DONE
  [ "$REV_DONE" = 0 ] && watch_one rev "$REV" "$REV_OUT" REV_DONE
  [ "$FWD_DONE" = 1 ] && [ "$REV_DONE" = 1 ] && break
  sleep 300
done
