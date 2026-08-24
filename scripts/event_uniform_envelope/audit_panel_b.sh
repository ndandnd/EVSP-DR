#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 PANEL_B_ROOT"
ROOT=$(cd "$1" && pwd)
[[ -f "$ROOT/jobs.tsv" ]] || evsp_die "missing $ROOT/jobs.tsv"

mapfile -t JOB_IDS < <(awk -F'\t' 'NR > 1 {print $2}' "$ROOT/jobs.tsv")
JOB_LIST=$(IFS=,; echo "${JOB_IDS[*]}")
sacct -j "$JOB_LIST" -n -P \
  -o JobIDRaw,JobName%40,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,ReqMem,NodeList \
  > "$ROOT/slurm_accounting.psv"
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_panel_b.py" \
  --root "$ROOT" --sacct "$ROOT/slurm_accounting.psv"
sha256sum "$ROOT/panel_b_summary.csv" "$ROOT/slurm_accounting.psv" \
  > "$ROOT/SUMMARY_SHA256SUMS"
