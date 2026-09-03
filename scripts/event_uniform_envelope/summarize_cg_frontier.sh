#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 CAMPAIGN_ROOT"
ROOT="$1"
[[ -f "$ROOT/medium_event_summary.csv" ]] \
  || evsp_die "audit the campaign before summarizing it: $ROOT"
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_cg_frontier.py" "$ROOT"
