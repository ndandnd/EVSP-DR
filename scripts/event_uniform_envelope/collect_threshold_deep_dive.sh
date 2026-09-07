#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 BASELINE_ROOT RESUME_ROOT"
BASELINE_ROOT="$1"
RESUME_ROOT="$2"
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTPUT_ROOT="$RESUME_ROOT/deep_dive_$STAMP"

"$PYTHON_BIN" "$SCRIPT_DIR/collect_threshold_deep_dive.py" \
  --source-root "$BASELINE_ROOT" \
  --resume-root "$RESUME_ROOT" \
  --output-root "$OUTPUT_ROOT"

ARCHIVE="$RESUME_ROOT/threshold_deep_dive_$STAMP.tar.gz"
tar -C "$RESUME_ROOT" -czf "$ARCHIVE" "$(basename "$OUTPUT_ROOT")"
cp "$ARCHIVE" "$RESUME_ROOT/threshold_deep_dive_latest.tar.gz"
printf '%s\n' "$OUTPUT_ROOT" > "$RESUME_ROOT/LATEST_DEEP_DIVE"
printf 'Latest deep-dive directory: %s\n' "$OUTPUT_ROOT"
printf 'Copy-ready archive: %s\n' "$RESUME_ROOT/threshold_deep_dive_latest.tar.gz"
