#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [SMALL_THRESHOLD_ROOT]"
SOURCE_ROOT="${1:-$HOME/ladder-lite/small_threshold_event_20260903_44b6d5}"
ROOT=$(cd "$SOURCE_ROOT/k5_raw_mip36h_20260905" && pwd)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
mapfile -t IDS < <(
  awk -F'\t' 'FNR > 1 {print $2}' "$ROOT"/jobs_*.tsv | sort -u
)
[[ ${#IDS[@]} == 1 ]] || evsp_die "expected exactly one k=5 MIP array"
if squeue --me -h -j "${IDS[0]}" 2>/dev/null | grep -q .; then
  evsp_die "k=5 RAW 36h MIP array is still active"
fi
SACCT="$ROOT/slurm_accounting.psv"
sacct -j "${IDS[0]}" -X -n -P \
  -o JobIDRaw,JobName%24,State,ExitCode,Elapsed,TotalCPU,MaxRSS,MaxVMSize,NodeList \
  > "$SACCT"
"$PYTHON_BIN" "$SCRIPT_DIR/audit_k5_raw_mip36h.py" \
  --root "$ROOT" --sacct "$SACCT"
sha256sum "$ROOT/snapshot_manifest.csv" "$ROOT"/jobs_*.tsv \
  "$SACCT" "$ROOT/k5_raw_mip36h_summary.csv" \
  > "$ROOT/AUDIT_SHA256SUMS"
