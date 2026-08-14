#!/bin/bash
# Read-only status summary for one controlled k40 factorial campaign.

set -u

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CAMPAIGN=${1:-}
if [ -z "$CAMPAIGN" ]; then
  BASE="$ROOT/src/results/k40_factorial"
  CAMPAIGN=$(find "$BASE" -mindepth 1 -maxdepth 1 -type d \
    -exec basename {} \; 2>/dev/null | sort | tail -n 1)
  if [ -z "$CAMPAIGN" ]; then
    echo "no k40 factorial campaign exists under $BASE" >&2
    exit 2
  fi
fi

RESULT_DIR="$ROOT/src/results/k40_factorial/$CAMPAIGN"
MANIFEST="$RESULT_DIR/launch.tsv"
[ -f "$MANIFEST" ] || {
  echo "campaign manifest not found: $MANIFEST" >&2
  exit 2
}

echo "=== campaign $CAMPAIGN ==="
column -t -s $'\t' "$MANIFEST" 2>/dev/null || cat "$MANIFEST"

JOB_IDS=$(awk -F '\t' 'NR > 1 {print $2}' "$MANIFEST" | paste -sd, -)
if [ -n "$JOB_IDS" ]; then
  echo "=== live jobs ==="
  squeue -j "$JOB_IDS" -o '%.14i %.15j %.2t %.10M %R' 2>/dev/null || true
  echo "=== accounting ==="
  sacct -X -j "$JOB_IDS" \
    --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS 2>/dev/null || true
fi

report_states() {
  local title=$1
  local suffix=$2
  echo "=== $title ==="
  printf 'arm\tstop\thours\titers\tcolumns\tweight\tartificials\tmin_rc\n'
  for arm in CA CS PA PS; do
    result="$RESULT_DIR/k40r2_flat_${arm}${suffix}"
    if [ ! -s "$result" ]; then
      printf '%s\tPENDING\n' "$arm"
      continue
    fi
    jq -r --arg arm "$arm" '
      (.final_lp // .final // {}) as $f
      | [
          $arm,
          (.stop_reason // "unknown"),
          ((.wall_s // 0) / 3600),
          (.iterations // 0),
          (.columns // 0),
          ($f.route_weight // "NA"),
          ($f.artificial_total // $f.artificials // "NA"),
          (.final.min_rc // "NA")
        ] | @tsv
    ' "$result"
  done
}

report_states "historical 22-hour comparison snapshots" ".m1320.snapshot.json"
report_states "primary 24-hour snapshots" ".m1440.snapshot.json"
report_states "live or final canonical states" ".json"
