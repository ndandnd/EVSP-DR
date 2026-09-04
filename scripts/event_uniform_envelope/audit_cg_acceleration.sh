#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 CAMPAIGN_ROOT"
ROOT="$1"
[[ -f "$ROOT/matrix.tsv" && -f "$ROOT/execution_plan.json" ]] \
  || evsp_die "not a CG acceleration campaign: $ROOT"
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
mapfile -t JOB_FILES < <(
  find "$ROOT" -maxdepth 1 -type f \
    \( -name 'jobs.tsv' -o -name 'jobs_recovery_*.tsv' \) | sort
)
mapfile -t IDS < <(
  awk -F'\t' 'FNR > 1 {print $3}' "${JOB_FILES[@]}" | sort -u
)
[[ ${#IDS[@]} -ge 7 ]] || evsp_die "expected at least seven campaign arrays"
JOB_LIST=$(IFS=,; echo "${IDS[*]}")
if squeue --me -h -j "$JOB_LIST" | grep -q .; then
  evsp_die "CG acceleration arrays are still active"
fi
SACCT="$ROOT/slurm_accounting.psv"
sacct -j "$JOB_LIST" -n -P \
  -o JobID%48,JobIDRaw,JobName%40,State,ExitCode,Elapsed,Timelimit,TotalCPU,AllocCPUS,MaxRSS,MaxVMSize,ReqMem,NodeList \
  > "$SACCT"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_cg_acceleration.py" \
  "$ROOT" --sacct "$SACCT"
sha256sum "$ROOT/matrix.tsv" "$ROOT/execution_plan.json" \
  "${JOB_FILES[@]}" "$SACCT" "$ROOT/cg_acceleration_rows.csv" \
  "$ROOT/cg_acceleration_by_arm_scale.csv" \
  > "$ROOT/AUDIT_SHA256SUMS"
