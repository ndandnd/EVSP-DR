#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
ROOT="${1:-$HOME/ladder-lite/transition_gap_event_20260902_44b6d5}"
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
SOLVER_COMMIT="44b6d5030a78ddca9c74f582d70ad87572e61794"

REMOTE=$(git -C "$REPO" ls-remote --heads origin 'cursor/event-based-pricer-2969*') \
  || evsp_die "git ls-remote failed for event-pricer branch"
printf '%s\n' "$REMOTE" >&2
[[ "$(printf '%s\n' "$REMOTE" | awk 'NF {n++} END {print n+0}')" == 1 ]] \
  || evsp_die "expected exactly one event-pricer branch"
[[ "$(printf '%s\n' "$REMOTE" | awk '{print $2}')" == \
  "refs/heads/cursor/event-based-pricer-2969" ]] \
  || evsp_die "unexpected event-pricer ref"
AGENT_SHA=$(printf '%s\n' "$REMOTE" | awk '{print $1}')
git -C "$REPO" fetch origin \
  'refs/heads/cursor/event-based-pricer-2969:refs/remotes/origin/cursor/event-based-pricer-2969'
git -C "$REPO" merge-base --is-ancestor "$SOLVER_COMMIT" "$AGENT_SHA" \
  || evsp_die "reviewed solver is not an ancestor of event-pricer tip"

if squeue --me -h -o '%j' | grep -qE '^tg02_k(3|4|6|7)$'; then
  evsp_die "transition-gap event campaign already active"
fi
[[ ! -e "$ROOT" ]] || evsp_die "campaign root already exists: $ROOT"

SOLVER_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
INPUT_REPO=$(evsp_execution_checkout "$REPO" "$WRAPPER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
"$PYTHON_BIN" \
  "$INPUT_REPO/scripts/event_uniform_envelope/validate_transition_gap_inputs.py" \
  --repo "$INPUT_REPO"
"$PYTHON_BIN" "$SOLVER_REPO/src/exact_pricer_expanded.py" --help \
  | grep -q -- '--time-model {uniform,event}' \
  || evsp_die "reviewed solver does not expose event time"
"$PYTHON_BIN" \
  "$INPUT_REPO/scripts/event_uniform_envelope/prepare_transition_gap_event.py" \
  --input-repo "$INPUT_REPO" --solver-repo "$SOLVER_REPO" \
  --root "$ROOT" --input-commit "$WRAPPER_COMMIT" \
  --solver-commit "$SOLVER_COMMIT" --wrapper-commit "$WRAPPER_COMMIT"

COMMON="EVSP_EXECUTION_REPO=$SOLVER_REPO,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_TIME_MODEL=event,EVSP_EVENT_ARC_MODE=lazy,EVSP_PYTHON=$PYTHON_BIN"
RECORD="$ROOT/jobs.tsv"
echo -e "scale\tarray_job_id\tindices\tpartition\tmem\tcpus\ttimelimit\twrapper_commit\tsolver_commit\tinput_commit\ttime_model\tevent_arc_mode" > "$RECORD"

submit_scale() {
  local scale="$1" name="$2" indices="$3" memory="$4"
  local job
  job=$(evsp_submit_and_resolve "$name" \
    --array="$indices%9" -p default_partition -c 1 --mem="$memory" \
    -t 12:15:00 --requeue --open-mode=append --signal=B:TERM@180 \
    --export="ALL,$COMMON" \
    -o "$ROOT/logs/${name}_%A_%a.out" \
    -e "$ROOT/logs/${name}_%A_%a.err" \
    "$INPUT_REPO/scripts/event_uniform_envelope/medium_event_cg.sub")
  echo -e "$scale\t$job\t$indices\tdefault_partition\t$memory\t1\t12:15:00\t$WRAPPER_COMMIT\t$SOLVER_COMMIT\t$WRAPPER_COMMIT\tevent\tlazy" >> "$RECORD"
  echo "$name: $job"
}

submit_scale 3 tg02_k3 0-8 24G
submit_scale 4 tg02_k4 9-17 24G
submit_scale 6 tg02_k6 18-26 32G
submit_scale 7 tg02_k7 27-35 32G
sha256sum "$ROOT/matrix.tsv" "$ROOT/execution_plan.json" "$RECORD" \
  > "$ROOT/SUBMISSION_SHA256SUMS"
echo "Transition-gap event campaign: $ROOT"
echo "Machine job record: $RECORD"
echo "Audit after completion:"
echo "bash $INPUT_REPO/scripts/event_uniform_envelope/audit_medium_event_legacy.sh $ROOT"
