#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 PANEL_A_ROOT"

A_ROOT=$(cd "$1" && pwd)
bash "$SCRIPT_DIR/audit_panel_a.sh" "$A_ROOT"
bash "$SCRIPT_DIR/resubmit_panel_a_integer.sh" "$A_ROOT"
bash "$SCRIPT_DIR/submit_panel_b.sh" "$A_ROOT"

echo
echo "=== newly queued work ==="
squeue -r --me -h -o '%j|%T' |
  awk -F'|' '$1 ~ /^(eua24_mipr|eua24_tfr|eub24_cg|eub24_frz)$/ {count[$1 FS $2]++} END {for (x in count) print count[x], x}' |
  sort
