#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 CAMPAIGN_ROOT"
ROOT=$(cd "$1" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
[[ -f "$ROOT/jobs.tsv" ]] || evsp_die "missing medium event jobs.tsv"
mapfile -t IDS < <(awk -F'\t' 'FNR > 1 {print $2}' "$ROOT/jobs.tsv" | sort -u)
[[ ${#IDS[@]} == 3 ]] || evsp_die "expected three medium event array IDs"
JOB_LIST=$(IFS=,; echo "${IDS[*]}")
if squeue --me -h -j "$JOB_LIST" | grep -q .; then
  evsp_die "medium event arrays are still active"
fi
SACCT="$ROOT/slurm_accounting.psv"
sacct -j "$JOB_LIST" -n -P \
  -o JobID%48,JobIDRaw,JobName%40,State,ExitCode,Elapsed,Timelimit,TotalCPU,AllocCPUS,MaxRSS,MaxVMSize,ReqMem,NodeList \
  > "$SACCT"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_medium_event_legacy.py" \
  --root "$ROOT" --sacct "$SACCT"
sha256sum "$ROOT/matrix.tsv" "$ROOT/execution_plan.json" "$ROOT/jobs.tsv" \
  "$SACCT" "$ROOT/medium_event_summary.csv" > "$ROOT/AUDIT_SHA256SUMS"
