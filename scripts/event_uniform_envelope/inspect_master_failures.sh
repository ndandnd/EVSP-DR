#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 2 ]] || evsp_die "usage: $0 [SOURCE_ROOT] [RESUME_DIRECTORY]"
SOURCE_ROOT="${1:-$HOME/ladder-lite/small_threshold_event_20260903_44b6d5}"
RESUME_DIRECTORY="${2:-cg_resume48h_20260904}"
ROOT=$(cd "$SOURCE_ROOT/$RESUME_DIRECTORY" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"

"$PYTHON_BIN" "$SCRIPT_DIR/inspect_master_failures.py" \
  --resume-root "$ROOT" --expected 3 \
  --output "$ROOT/master_failure_summary.csv"
