#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
ROOT="${1:-$HOME/ladder-lite/medium_event_corrected_20260831_44b6d5}"
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
SOLVER_COMMIT="44b6d5030a78ddca9c74f582d70ad87572e61794"
INPUT_COMMIT="ff7fb2ba93cf13a31171e1e4aeb2d28dc8aeee20"
AGENT_BRANCH="cursor/event-based-pricer-2969"
INPUT_BRANCH="cursor/ladder-lite-20260819-2969"

verify_producer() {
  local branch="$1" required="$2" observed count ref sha
  observed=$(git -C "$REPO" ls-remote --heads origin "${branch}*") \
    || evsp_die "git ls-remote failed for $branch"
  printf '%s\n' "$observed" >&2
  count=$(printf '%s\n' "$observed" | awk 'NF {n++} END {print n+0}')
  [[ "$count" == 1 ]] || evsp_die "expected exactly one branch matching $branch*"
  ref=$(printf '%s\n' "$observed" | awk '{print $2}')
  sha=$(printf '%s\n' "$observed" | awk '{print $1}')
  [[ "$ref" == "refs/heads/$branch" ]] || evsp_die "unexpected ref $ref"
  git -C "$REPO" fetch origin "refs/heads/$branch:refs/remotes/origin/$branch"
  git -C "$REPO" merge-base --is-ancestor "$required" "$sha" \
    || evsp_die "$required is not an ancestor of $branch tip"
}
verify_producer "$AGENT_BRANCH" "$SOLVER_COMMIT"
verify_producer "$INPUT_BRANCH" "$INPUT_COMMIT"

if squeue --me -h -o '%j' | grep -qE '^(me31_k8|me31_k13|me31_k20)$'; then
  evsp_die "medium event campaign already active"
fi
[[ ! -e "$ROOT" ]] || evsp_die "campaign root already exists: $ROOT"
SOLVER_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
INPUT_REPO=$(evsp_execution_checkout "$REPO" "$INPUT_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
"$PYTHON_BIN" "$SOLVER_REPO/src/exact_pricer_expanded.py" --help \
  | grep -q -- '--time-model {uniform,event}' \
  || evsp_die "reviewed solver does not expose the event time model"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_medium_event_legacy.py" \
  --input-repo "$INPUT_REPO" --solver-repo "$SOLVER_REPO" \
  --root "$ROOT" --input-commit "$INPUT_COMMIT" \
  --solver-commit "$SOLVER_COMMIT" --wrapper-commit "$WRAPPER_COMMIT"

COMMON="EVSP_EXECUTION_REPO=$SOLVER_REPO,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_TIME_MODEL=event,EVSP_EVENT_ARC_MODE=lazy,EVSP_PYTHON=$PYTHON_BIN"
RECORD="$ROOT/jobs.tsv"
echo -e "scale\tarray_job_id\tindices\tpartition\tmem\tcpus\ttimelimit\twrapper_commit\tsolver_commit\tinput_commit\ttime_model\tevent_arc_mode" > "$RECORD"
submit_scale() {
  local scale="$1" name="$2" indices="$3" memory="$4"
  local job
  job=$(evsp_submit_and_resolve "$name" \
    --array="$indices%6" -p default_partition -c 1 --mem="$memory" \
    -t 12:15:00 --requeue --open-mode=append --signal=B:TERM@180 \
    --export="ALL,$COMMON" \
    -o "$ROOT/logs/${name}_%A_%a.out" -e "$ROOT/logs/${name}_%A_%a.err" \
    "$SCRIPT_DIR/medium_event_cg.sub")
  echo -e "$scale\t$job\t$indices\tdefault_partition\t$memory\t1\t12:15:00\t$WRAPPER_COMMIT\t$SOLVER_COMMIT\t$INPUT_COMMIT\tevent\tlazy" >> "$RECORD"
  echo "$name: $job"
}
submit_scale 8 me31_k8 0-5 32G
submit_scale 13 me31_k13 6-11 64G
submit_scale 20 me31_k20 12-17 96G
sha256sum "$ROOT/matrix.tsv" "$ROOT/execution_plan.json" "$RECORD" \
  > "$ROOT/SUBMISSION_SHA256SUMS"
echo "Medium event campaign: $ROOT"
echo "Machine job record: $RECORD"
