#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT PANEL_B_ROOT"
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
"$PYTHON_BIN" "$SCRIPT_DIR/triage_long_cluster_fill.py" \
  --panel-a "$1" --panel-b "$2"
