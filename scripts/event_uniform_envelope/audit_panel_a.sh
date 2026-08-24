#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn

[[ $# == 1 ]] || evsp_die "usage: $0 PANEL_A_ROOT"
ROOT=$(cd "$1" && pwd)
[[ -f "$ROOT/jobs.tsv" ]] || evsp_die "missing $ROOT/jobs.tsv"
[[ -f "$ROOT/matrix.tsv" ]] || evsp_die "missing $ROOT/matrix.tsv"

JOB_FILES=("$ROOT/jobs.tsv")
[[ ! -f "$ROOT/integer_recovery_jobs.tsv" ]] \
  || JOB_FILES+=("$ROOT/integer_recovery_jobs.tsv")
mapfile -t JOB_IDS < <(
  awk -F'\t' 'FNR > 1 {print $2}' "${JOB_FILES[@]}" | sort -u
)
[[ ${#JOB_IDS[@]} -gt 0 ]] || evsp_die "jobs.tsv contains no job IDs"
JOB_LIST=$(IFS=,; echo "${JOB_IDS[*]}")

sacct -j "$JOB_LIST" -n -P \
  -o JobIDRaw,JobName%40,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,ReqMem,NodeList \
  > "$ROOT/slurm_accounting.psv"

PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_panel_a.py" \
  --root "$ROOT" --sacct "$ROOT/slurm_accounting.psv"

sha256sum \
  "$ROOT/matrix.tsv" \
  "$ROOT/jobs.tsv" \
  "$ROOT/panel_a_summary.csv" \
  "$ROOT/panel_a_stage_counts.csv" \
  "$ROOT/slurm_accounting.psv" \
  "$ROOT/stderr_inventory.csv" \
  "$ROOT/panel_b_gate.json" \
  > "$ROOT/SUMMARY_SHA256SUMS"

echo "CSV: $ROOT/panel_a_summary.csv"
