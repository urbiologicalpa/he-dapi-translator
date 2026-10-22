#!/bin/bash
# run_tests.sh - run pytest under a hard address-space cap.
#
# The 3.7GB WSL box has only 1GB of swap. Without a cap, a runaway test can
# trigger the kernel OOM killer and restart the whole VM. Slow training smokes
# are excluded by default (pyproject addopts); pass `-m slow` to include them.
#
# Usage:
#   bash scripts/run_tests.sh -q
#   bash scripts/run_tests.sh -m slow -q
#   HE_DAPI_TEST_MEM_MB=3000 bash scripts/run_tests.sh -q
set -u
cd "$(dirname "$0")/.." || exit 1

MEM_MB="${HE_DAPI_TEST_MEM_MB:-2500}"
ulimit -v $((MEM_MB * 1024)) || {
  echo "could not set ulimit -v ${MEM_MB}MB" >&2
  exit 1
}

export PYTHONPATH="${PYTHONPATH:-.}"
exec .venv/bin/python -m pytest "$@"
