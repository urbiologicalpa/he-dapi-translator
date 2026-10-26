#!/bin/bash
set -u
cd /home/bikash/he-dapi-translator || exit 1
export KAGGLE_API_TOKEN="$(cat ~/.kaggle/api_token)"
exec timeout 10800 .venv/bin/kaggle kernels logs bikasash/he-dapi-cyclegan-ch2-256-r-007 --follow
