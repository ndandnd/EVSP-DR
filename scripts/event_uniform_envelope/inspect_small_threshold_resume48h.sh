#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 2 ]] || evsp_die "usage: $0 [SOURCE_ROOT] [RESUME_DIRECTORY]"
SOURCE_ROOT="${1:-$HOME/ladder-lite/small_threshold_event_20260903_44b6d5}"
RESUME_DIRECTORY="${2:-cg_resume48h_20260904}"
ROOT=$(cd "$SOURCE_ROOT/$RESUME_DIRECTORY" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
mapfile -t IDS < <(
  awk -F'\t' 'FNR > 1 {print $2}' "$ROOT"/jobs_*.tsv | sort -u
)
[[ ${#IDS[@]} -ge 1 ]] || evsp_die "no cumulative-48h jobs"
JOB_LIST=$(IFS=,; echo "${IDS[*]}")
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_DIR="$ROOT/progress_snapshots/$STAMP"
mkdir -p "$OUTPUT_DIR"
SACCT="$OUTPUT_DIR/slurm_accounting.psv"
sacct -j "$JOB_LIST" -n -P \
  -o JobID%48,JobIDRaw,JobName%40,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,NodeList \
  > "$SACCT"
"$PYTHON_BIN" "$SCRIPT_DIR/inspect_small_threshold_resume48h.py" \
  --resume-root "$ROOT" --sacct "$SACCT" \
  --output "$OUTPUT_DIR/resume_progress.csv"
squeue -r --me -h -j "$JOB_LIST" -o '%i|%j|%P|%T|%M|%l|%R' \
  > "$OUTPUT_DIR/active_queue.psv"
sha256sum "$OUTPUT_DIR"/*.csv "$OUTPUT_DIR"/*.psv \
  > "$OUTPUT_DIR/SHA256SUMS"
