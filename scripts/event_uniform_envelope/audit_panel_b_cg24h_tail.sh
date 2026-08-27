#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 PANEL_B_ROOT"
B_ROOT=$(cd "$1" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
RESUME="$B_ROOT/cg_certification24h_13596d0"
mapfile -t IDS < <(
  awk -F'\t' 'FNR > 1 && $2 != "" {print $2}' \
    "$RESUME"/jobs_*.tsv | sort -u
)
[[ ${#IDS[@]} -gt 0 ]] || evsp_die "no Panel B 24h CG job IDs"
JOB_LIST=$(IFS=,; echo "${IDS[*]}")
SACCT="$RESUME/slurm_accounting.psv"
sacct -j "$JOB_LIST" -n -P \
  -o JobID%48,JobName%40,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,ReqMem,NodeList \
  > "$SACCT"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_cg_resume.py" \
  --resume-root "$RESUME" --sacct "$SACCT"
sha256sum \
  "$RESUME/execution_plan.json" "$RESUME/matrix.tsv" \
  "$RESUME/resume_summary.csv" "$SACCT" \
  > "$RESUME/CG24H_SUMMARY_SHA256SUMS"
echo "CSV: $RESUME/resume_summary.csv"
