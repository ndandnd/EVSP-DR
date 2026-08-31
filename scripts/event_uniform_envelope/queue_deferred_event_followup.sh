#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 2 ]] || evsp_die "usage: $0 [MEDIUM_ROOT] [EXTENSION_ROOT]"

MEDIUM_ROOT="${1:-$HOME/ladder-lite/medium_event_corrected_20260831_44b6d5}"
EXTENSION_ROOT="${2:-$HOME/ladder-lite/event_extension_corrected_20260831_44b6d5}"
MEDIUM_ROOT=$(cd "$MEDIUM_ROOT" && pwd)
EXTENSION_ROOT=$(cd "$EXTENSION_ROOT" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
WRAPPER_REPO=$(evsp_execution_checkout "$REPO" "$WRAPPER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"

validate_plan() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
plan = json.loads((root / "execution_plan.json").read_text())
expected = {
    "solver_commit": "44b6d5030a78ddca9c74f582d70ad87572e61794",
    "time_model": "event",
    "event_arc_mode": "lazy",
    "wall_limit_s": 43200,
}
observed = {key: plan.get(key) for key in expected}
if observed != expected:
    raise SystemExit(f"source execution-plan mismatch: {observed}")
PY
}
validate_plan "$MEDIUM_ROOT"
validate_plan "$EXTENSION_ROOT"

mapfile -t IDS < <(
  awk -F'\t' 'FNR > 1 {print $2}' \
    "$MEDIUM_ROOT/jobs.tsv" "$EXTENSION_ROOT/jobs.tsv" | sort -u
)
[[ ${#IDS[@]} == 6 ]] || evsp_die "expected six corrected source arrays"
DEPENDENCY=$(IFS=:; echo "${IDS[*]}")

GATE_ROOT="$HOME/ladder-lite/deferred_event_followup_20260831_${WRAPPER_COMMIT:0:7}"
[[ ! -e "$GATE_ROOT" ]] || evsp_die "deferred follow-up already exists: $GATE_ROOT"
mkdir -p "$GATE_ROOT/logs"
EXPORTS="ALL,EVSP_WRAPPER_REPO=$WRAPPER_REPO,EVSP_WRAPPER_COMMIT=$WRAPPER_COMMIT,EVSP_MEDIUM_ROOT=$MEDIUM_ROOT,EVSP_EXTENSION_ROOT=$EXTENSION_ROOT,EVSP_PYTHON=$PYTHON_BIN"
JOB=$(evsp_submit_and_resolve ev31_gate \
  --dependency="afterany:$DEPENDENCY" \
  -p default_partition -c 1 --mem=2G -t 00:30:00 --no-requeue \
  --export="$EXPORTS" \
  -o "$GATE_ROOT/logs/controller_%j.out" \
  -e "$GATE_ROOT/logs/controller_%j.err" \
  "$WRAPPER_REPO/scripts/event_uniform_envelope/deferred_event_followup.sub")

{
  echo -e "controller_job_id\tdependency_type\tparent_array_ids\twrapper_commit\tmedium_root\textension_root"
  echo -e "$JOB\tafterany\t$DEPENDENCY\t$WRAPPER_COMMIT\t$MEDIUM_ROOT\t$EXTENSION_ROOT"
} > "$GATE_ROOT/controller_job.tsv"
sha256sum "$GATE_ROOT/controller_job.tsv" > "$GATE_ROOT/SUBMISSION_SHA256SUMS"
echo "Deferred event follow-up controller: $JOB"
echo "Dependencies: $DEPENDENCY"
echo "Controller record: $GATE_ROOT/controller_job.tsv"
