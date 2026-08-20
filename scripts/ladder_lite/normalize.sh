#!/bin/bash

LL_ROOT=${LL_ROOT:-"$HOME/ladder-lite"}
LL_PYTHON=${LL_PYTHON:-/home/nc437/evsp_env/bin/python3.12}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || exit 1
"$LL_PYTHON" -B "$REPO/src/summarize_scale_ladder.py" \
  --campaign-root "$LL_ROOT/campaign" --out-dir "$LL_ROOT/normalized" \
  --execution-mode ladder-lite-direct
