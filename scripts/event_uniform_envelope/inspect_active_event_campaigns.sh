#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die \
  "usage: $0 ACCELERATION_ROOT SMALL_THRESHOLD_ROOT"
ACCEL=$(cd "$1" && pwd)
SMALL=$(cd "$2" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"

mapfile -t ACCEL_JOBS < <(
  find "$ACCEL" -maxdepth 1 -type f \
    \( -name 'jobs.tsv' -o -name 'jobs_recovery_*.tsv' \) -print0 |
    xargs -0 awk -F'\t' 'FNR > 1 {print $3}' | sort -u
)
mapfile -t SMALL_JOBS < <(
  find "$SMALL" -maxdepth 1 -type f \
    \( -name 'jobs.tsv' -o -name 'jobs_recovery_*.tsv' \) -print0 |
    xargs -0 awk -F'\t' 'FNR > 1 {print $2}' | sort -u
)
IDS=("${ACCEL_JOBS[@]}" "${SMALL_JOBS[@]}")
[[ ${#IDS[@]} -gt 0 ]] || evsp_die "no campaign job IDs"
JOB_LIST=$(IFS=,; echo "${IDS[*]}")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT="$ACCEL/progress_snapshots/$STAMP"
SACCT=$(mktemp)
trap 'rm -f "$SACCT"' EXIT
sacct -j "$JOB_LIST" -n -P \
  -o JobID%48,JobIDRaw,JobName%40,State,ExitCode,Elapsed,Timelimit,TotalCPU,AllocCPUS,MaxRSS,MaxVMSize,ReqMem,NodeList \
  > "$SACCT"
"$PYTHON_BIN" "$SCRIPT_DIR/inspect_active_event_campaigns.py" \
  --acceleration-root "$ACCEL" --small-root "$SMALL" \
  --sacct "$SACCT" --output-dir "$OUTPUT"
squeue -r --me -h -o '%i|%j|%P|%T|%M|%l|%R' \
  > "$OUTPUT/active_queue.psv"
cp "$SACCT" "$OUTPUT/slurm_accounting.psv"
sha256sum "$OUTPUT"/*.csv "$OUTPUT"/*.psv > "$OUTPUT/SHA256SUMS"
