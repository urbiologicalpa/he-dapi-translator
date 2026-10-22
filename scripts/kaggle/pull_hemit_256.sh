#!/usr/bin/env bash
# Pull the finished corrected HEMIT 256px CycleGAN output and verify provenance.
# Usage: bash scripts/kaggle/pull_hemit_256.sh [output_dir]
set -euo pipefail
cd "$(dirname "$0")/../.." || exit 1

KERNEL="bikasash/he-dapi-cyclegan-hemit-dapi2he-256"
OUT="${1:-/tmp/k256out}"
mkdir -p logs

if [ ! -f "$HOME/.kaggle/api_token" ]; then
  echo "missing ~/.kaggle/api_token" >&2
  exit 1
fi
export KAGGLE_API_TOKEN="$(cat "$HOME/.kaggle/api_token")"

status="$(.venv/bin/kaggle kernels status "$KERNEL" 2>&1 | tail -1)"
echo "$status"
case "$status" in
  *COMPLETE*) ;;
  *) echo "not ready to pull (expect COMPLETE)" >&2; exit 2 ;;
esac

rm -rf "$OUT"
mkdir -p "$OUT"
.venv/bin/kaggle kernels output "$KERNEL" -p "$OUT"
.venv/bin/kaggle kernels logs "$KERNEL" > logs/kaggle_256_final.log 2>&1 || true

eval_json="$(find "$OUT" -name eval_cyclegan.json -print -quit)"
if [ -z "$eval_json" ]; then
  echo "FATAL: eval_cyclegan.json not found under $OUT" >&2
  find "$OUT" -maxdepth 4 -type f | sed -n '1,40p' >&2
  exit 3
fi

echo "--- eval JSON ---"
cat "$eval_json"
echo
echo "--- checkpoints ---"
find "$OUT" -name '*.pt' -print0 | sort -z | xargs -0 -r sha256sum | tee "$OUT/CHECKSUMS.sha256"
echo
echo "eval_json=$eval_json"
echo "checksums=$OUT/CHECKSUMS.sha256"
echo "logs=logs/kaggle_256_final.log"
