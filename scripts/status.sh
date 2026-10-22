#!/usr/bin/env bash
# One-shot live-state snapshot. Read docs/LIVE_STATE.md for context.
set -u
cd "$(dirname "$0")/.." || exit 1

echo "=== $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
echo "--- boot ---"
uptime
echo
echo "--- memory ---"
free -m
echo
echo "--- training processes ---"
pgrep -af 'train_unpaired|train_paired|train_dapi' || echo "none"
echo
echo "--- unpaired checkpoints (256) ---"
ls -lt resources/models/checkpoints/hemit_cyclegan_dapi2he_256/ 2>/dev/null | head -8 || echo "missing"
echo
echo "--- local unpaired log (tail) ---"
tail -6 logs/train_unpaired.log 2>/dev/null || echo "missing"
echo
echo "--- cron (training lines) ---"
crontab -l 2>/dev/null | grep -E 'train|boot_launch' || echo "none"
echo
echo "--- kaggle kernels ---"
if [ -f "$HOME/.kaggle/api_token" ]; then
  for k in \
    bikasash/he-dapi-cyclegan-hemit-dapi2he-1024 \
    bikasash/he-dapi-cyclegan-hemit-dapi2he-256; do
    KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/api_token")" \
      .venv/bin/kaggle kernels status "$k" 2>&1 | tail -1
  done
else
  echo "no ~/.kaggle/api_token"
fi
