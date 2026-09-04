#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
ROOT="${1:-$HOME/ladder-lite/small_threshold_event_20260903_44b6d5}"
ROOT=$(cd "$ROOT" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
SOLVER_COMMIT=$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["solver_commit"])' \
    "$ROOT/execution_plan.json"
)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")

if squeue --me -h -o '%j' | grep -q '^st04_k9r$'; then
  evsp_die "small-threshold recovery is already active"
fi
INDEX_FILE=$(mktemp)
trap 'rm -f "$INDEX_FILE"' EXIT
"$PYTHON_BIN" "$SCRIPT_DIR/select_missing_preempted_event_indices.py" \
  --root "$ROOT" --expected-scale 9 > "$INDEX_FILE"
[[ -s "$INDEX_FILE" ]] || evsp_die "no audited missing PREEMPTED rows"
[[ "$(wc -l < "$INDEX_FILE" | tr -d ' ')" == 3 ]] \
  || evsp_die "expected exactly three missing PREEMPTED rows"
INDICES=$(paste -sd, "$INDEX_FILE")

QUARANTINE="$ROOT/quarantine_preempted_20260904"
mkdir -p "$QUARANTINE"
QUARANTINE_LOG="$QUARANTINE/files.tsv"
if [[ ! -e "$QUARANTINE_LOG" ]]; then
  printf 'index\tsource\tdestination\tsha256\n' > "$QUARANTINE_LOG"
fi
while read -r index; do
  LINE=$(sed -n "$((index + 1))p" "$ROOT/matrix.tsv")
  IFS=$'\t' read -r observed cell _rest <<< "$LINE"
  [[ "$observed" == "$index" ]]
  OUT="$ROOT/cg/M__${cell}__event_2p5_event5.json"
  [[ ! -e "$OUT" ]] || evsp_die "status appeared for index $index"
  for sidecar in \
    "$OUT.columns.jsonl" "$OUT.iters.csv" \
    "$OUT.phase-telemetry.jsonl" "$OUT.lock"; do
    if [[ -e "$sidecar" ]]; then
      destination="$QUARANTINE/$(basename "$sidecar").pre-recovery"
      [[ ! -e "$destination" ]] \
        || evsp_die "quarantine destination exists: $destination"
      digest=$(sha256sum "$sidecar" | awk '{print $1}')
      mv "$sidecar" "$destination"
      printf '%s\t%s\t%s\t%s\n' \
        "$index" "$sidecar" "$destination" "$digest" \
        >> "$QUARANTINE_LOG"
    fi
  done
done < "$INDEX_FILE"

COMMON="EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_TIME_MODEL=event,EVSP_EVENT_ARC_MODE=lazy,EVSP_PYTHON=$PYTHON_BIN"
JOB=$(evsp_submit_and_resolve st04_k9r \
  --array="$INDICES%3" -p default_partition -c 1 --mem=48G \
  -t 12:15:00 --requeue --open-mode=append --signal=B:TERM@180 \
  --export="ALL,$COMMON" \
  -o "$ROOT/logs/st04_k9r_%A_%a.out" \
  -e "$ROOT/logs/st04_k9r_%A_%a.err" \
  "$SCRIPT_DIR/medium_event_cg.sub")
RECORD="$ROOT/jobs_recovery_${JOB}.tsv"
{
  printf 'scale\tarray_job_id\tindices\tpartition\tmem\tcpus\ttimelimit\twrapper_commit\tsolver_commit\tinput_commit\ttime_model\tevent_arc_mode\n'
  printf '9\t%s\t%s\tdefault_partition\t48G\t1\t12:15:00\t%s\t%s\tNA\tevent\tlazy\n' \
    "$JOB" "$INDICES" "$WRAPPER_COMMIT" "$SOLVER_COMMIT"
} > "$RECORD"
sha256sum "$ROOT/matrix.tsv" "$ROOT/execution_plan.json" \
  "$RECORD" "$QUARANTINE_LOG" > "$ROOT/RECOVERY_${JOB}_SHA256SUMS"
echo "Small-threshold preemption recovery: $JOB (indices $INDICES)"
echo "Quarantined unauthenticated sidecars, if any: $QUARANTINE_LOG"
