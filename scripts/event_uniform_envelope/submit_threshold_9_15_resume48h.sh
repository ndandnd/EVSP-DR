#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [THRESHOLD_ROOT]"
SOURCE_ROOT="${1:-$HOME/ladder-lite/threshold_9_15_event_20260904_9bdbb17}"
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
SOLVER_COMMIT=$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["solver_commit"])' \
    "$SOURCE_ROOT/execution_plan.json"
)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")

mapfile -t SOURCE_JOBS < <(
  awk -F'\t' 'FNR > 1 && $1=="cg" {print $3}' "$SOURCE_ROOT/jobs.tsv" | sort -u
)
[[ ${#SOURCE_JOBS[@]} == 1 ]] || evsp_die "expected one source CG array"
if squeue --me -h -j "${SOURCE_JOBS[0]}" 2>/dev/null | grep -q .; then
  evsp_die "source k9--k15 CG array is still active"
fi

PARENT_CAP=43200
CHILD_CAP=172800
RESUME_ROOT="$SOURCE_ROOT/cg_resume48h_20260906"
if [[ ! -s "$RESUME_ROOT/STAGING_COMPLETE" ]]; then
  RESUME_ARGS=()
  [[ ! -e "$RESUME_ROOT" ]] || RESUME_ARGS=(--resume-incomplete)
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_threshold_9_15_resume48h.py" \
    --source-root "$SOURCE_ROOT" --out-root "$RESUME_ROOT" \
    --solver-commit "$SOLVER_COMMIT" \
    --parent-wall-limit-s "$PARENT_CAP" --wall-limit-s "$CHILD_CAP" \
    --expected-cells 49 "${RESUME_ARGS[@]}"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/repair_cg_resume_telemetry.py" \
  --resume-root "$RESUME_ROOT"
PENDING=$(mktemp)
trap 'rm -f "$PENDING"' EXIT
"$PYTHON_BIN" "$SCRIPT_DIR/select_cg_resume_indices.py" \
  --resume-root "$RESUME_ROOT" --expected-commit "$SOLVER_COMMIT" \
  --expected-wall-limit-s "$CHILD_CAP" > "$PENDING"
mapfile -t INDICES < "$PENDING"

if [[ ${#INDICES[@]} == 0 ]]; then
  echo "No k9--k15 cumulative-48h continuations remain"
  exit 0
fi
active=$(
  squeue --me -h -o '%A|%j' |
    awk -F'|' '$2=="th48cg" {print $1}' | sort -u
)
[[ -z "$active" ]] || evsp_die "th48cg continuation already active: $active"
ARRAY=$(IFS=,; echo "${INDICES[*]}")
COUNT=${#INDICES[@]}
EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$RESUME_ROOT,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CUMULATIVE_WALL_LIMIT_S=$CHILD_CAP,EVSP_PYTHON=$PYTHON_BIN"
JOB=$(evsp_submit_and_resolve th48cg \
  --array="$ARRAY%36" -p default_partition -c 1 --mem=96G \
  -t 1-12:30:00 --requeue --open-mode=append --signal=B:TERM@180 \
  --export="$EXPORTS" \
  -o "$RESUME_ROOT/logs/%x_%A_%a.out" \
  -e "$RESUME_ROOT/logs/%x_%A_%a.err" \
  "$SCRIPT_DIR/threshold_9_15_resume48h.sub")
{
  printf 'stage\tarray_job_id\ttasks\tindices\tpartition\tconcurrency\tmem\ttimelimit\twrapper_commit\tsolver_commit\tcumulative_wall_limit_s\n'
  printf 'threshold_9_15_resume48h\t%s\t%s\t%s\tdefault_partition\t36\t96G\t1-12:30:00\t%s\t%s\t%s\n' \
    "$JOB" "$COUNT" "$ARRAY" "$WRAPPER_COMMIT" "$SOLVER_COMMIT" "$CHILD_CAP"
} > "$RESUME_ROOT/jobs_${JOB}.tsv"
sha256sum "$RESUME_ROOT/execution_plan.json" "$RESUME_ROOT/matrix.tsv" \
  "$RESUME_ROOT"/jobs_*.tsv > "$RESUME_ROOT/SUBMISSION_INPUT_SHA256SUMS"
echo "k9--k15 cumulative-48h continuation: $JOB ($COUNT tasks, max 36 concurrent)"
echo "Resume root: $RESUME_ROOT"
