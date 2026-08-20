#!/bin/bash

LL_ROOT=${LL_ROOT:-"$HOME/ladder-lite"}; LL_CAMPAIGN=${LL_CAMPAIGN:-"ll_$(date -u +%Y%m%d)"}
LL_PYTHON=${LL_PYTHON:-/home/nc437/evsp_env/bin/python3.12}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || exit 1
"$LL_PYTHON" -B "$REPO/src/summarize_scale_ladder_lite.py" \
  --campaign-root "$LL_ROOT/campaign/$LL_CAMPAIGN" --out-dir "$LL_ROOT/normalized"
