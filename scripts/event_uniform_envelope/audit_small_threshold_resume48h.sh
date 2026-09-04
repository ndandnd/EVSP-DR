#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [SMALL_THRESHOLD_ROOT]"
SOURCE_ROOT="${1:-$HOME/ladder-lite/small_threshold_event_20260903_44b6d5}"
ROOT=$(cd "$SOURCE_ROOT/cg_resume48h_20260904" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
mapfile -t IDS < <(
  awk -F'\t' 'FNR > 1 {print $2}' "$ROOT"/jobs_*.tsv | sort -u
)
[[ ${#IDS[@]} -ge 1 ]] || evsp_die "no cumulative-48h jobs"
JOB_LIST=$(IFS=,; echo "${IDS[*]}")
if squeue --me -h -j "$JOB_LIST" 2>/dev/null | grep -q .; then
  evsp_die "cumulative-48h jobs are still active"
fi
SACCT="$ROOT/slurm_accounting.psv"
sacct -j "$JOB_LIST" -n -P \
  -o JobID%48,JobIDRaw,JobName%40,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,NodeList \
  > "$SACCT"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_cg_resume.py" \
  --resume-root "$ROOT" --sacct "$SACCT"
sha256sum "$ROOT/execution_plan.json" "$ROOT/matrix.tsv" \
  "$ROOT"/jobs_*.tsv "$SACCT" "$ROOT/resume_summary.csv" \
  > "$ROOT/AUDIT_SHA256SUMS"
echo "Cumulative-48h CSV: $ROOT/resume_summary.csv"
