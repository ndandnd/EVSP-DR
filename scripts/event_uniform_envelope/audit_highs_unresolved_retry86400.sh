#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT PANEL_B_ROOT"
A_ROOT=$(cd "$1" && pwd)
B_ROOT=$(cd "$2" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
A_JOBS="$A_ROOT/highs_unresolved_retry86400_jobs.tsv"
B_JOBS="$B_ROOT/highs_unresolved_retry86400_jobs.tsv"
[[ -f "$A_JOBS" && -f "$B_JOBS" ]] || evsp_die "missing 24-hour job record"
cmp -s "$A_JOBS" "$B_JOBS" || evsp_die "Panel A/B 24-hour job records differ"
mapfile -t IDS < <(
  awk -F'\t' 'FNR > 1 && $2 != "" {print $2}' "$A_JOBS" | sort -u
)
[[ ${#IDS[@]} -ge 1 ]] || evsp_die "no 24-hour array IDs"
JOB_LIST=$(IFS=,; echo "${IDS[*]}")
if squeue --me -h -j "$JOB_LIST" | grep -q .; then
  evsp_die "24-hour jobs are still active"
fi
SACCT="$A_ROOT/highs_retry86400_slurm_accounting.psv"
sacct -j "$JOB_LIST" -n -P \
  -o JobID%48,JobIDRaw,JobName%40,State,ExitCode,Elapsed,Timelimit,TotalCPU,AllocCPUS,MaxRSS,MaxVMSize,ReqMem,NodeList \
  > "$SACCT"
cp "$SACCT" "$B_ROOT/highs_retry86400_slurm_accounting.psv"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_highs_unresolved_retry86400.py" \
  --root "$A_ROOT" --panel A --sacct "$SACCT"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_highs_unresolved_retry86400.py" \
  --root "$B_ROOT" --panel B --sacct "$SACCT"
sha256sum "$A_JOBS" "$SACCT" \
  "$A_ROOT/backend_retry86400.csv" \
  "$A_ROOT/backend_retry86400_unresolved.csv" \
  > "$A_ROOT/HIGHS_RETRY86400_SUMMARY_SHA256SUMS"
sha256sum "$B_JOBS" "$B_ROOT/highs_retry86400_slurm_accounting.psv" \
  "$B_ROOT/backend_retry86400.csv" \
  "$B_ROOT/backend_retry86400_unresolved.csv" \
  > "$B_ROOT/HIGHS_RETRY86400_SUMMARY_SHA256SUMS"
echo "Panel A CSV: $A_ROOT/backend_retry86400.csv"
echo "Panel B CSV: $B_ROOT/backend_retry86400.csv"
