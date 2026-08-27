#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT PANEL_B_ROOT"
A_ROOT=$(cd "$1" && pwd)
B_ROOT=$(cd "$2" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
A_JOBS="$A_ROOT/highs_disagreement_retry_jobs.tsv"
B_JOBS="$B_ROOT/highs_disagreement_retry_jobs.tsv"
[[ -f "$A_JOBS" ]] || evsp_die "missing $A_JOBS"
[[ -f "$B_JOBS" ]] || evsp_die "missing $B_JOBS"
cmp -s "$A_JOBS" "$B_JOBS" \
  || evsp_die "Panel A/B retry job records differ"
mapfile -t JOB_IDS < <(
  awk -F'\t' 'FNR > 1 && $2 != "" {print $2}' "$A_JOBS" | sort -u
)
[[ ${#JOB_IDS[@]} == 2 ]] || evsp_die "expected exactly two retry array IDs"
JOB_LIST=$(IFS=,; echo "${JOB_IDS[*]}")
SACCT="$A_ROOT/highs_retry7200_slurm_accounting.psv"
sacct -j "$JOB_LIST" -n -P \
  -o JobID%48,JobIDRaw,JobName%40,State,ExitCode,Elapsed,Timelimit,TotalCPU,AllocCPUS,MaxRSS,MaxVMSize,ReqMem,NodeList \
  > "$SACCT"
cp "$SACCT" "$B_ROOT/highs_retry7200_slurm_accounting.psv"

"$PYTHON_BIN" "$SCRIPT_DIR/audit_highs_disagreement_retry.py" \
  --panel-a "$A_ROOT" --panel-b "$B_ROOT" --sacct "$SACCT"

sha256sum \
  "$A_JOBS" "$SACCT" \
  "$A_ROOT/backend_retry7200.csv" \
  "$A_ROOT/backend_retry7200_unresolved.csv" \
  > "$A_ROOT/HIGHS_RETRY7200_SUMMARY_SHA256SUMS"
sha256sum \
  "$B_JOBS" "$B_ROOT/highs_retry7200_slurm_accounting.psv" \
  "$B_ROOT/backend_retry7200.csv" \
  "$B_ROOT/backend_retry7200_unresolved.csv" \
  > "$B_ROOT/HIGHS_RETRY7200_SUMMARY_SHA256SUMS"

echo "Panel A CSV: $A_ROOT/backend_retry7200.csv"
echo "Panel B CSV: $B_ROOT/backend_retry7200.csv"
