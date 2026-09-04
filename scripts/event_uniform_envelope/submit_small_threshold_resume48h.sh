#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [SMALL_THRESHOLD_ROOT]"
SOURCE_ROOT="${1:-$HOME/ladder-lite/small_threshold_event_20260903_44b6d5}"
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

PARENT_CAP=43200
CHILD_CAP=172800
RESUME_ROOT="$SOURCE_ROOT/cg_resume48h_20260904"
if [[ ! -e "$RESUME_ROOT" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_wall_capped_event_resume.py" \
    --source-root "$SOURCE_ROOT" --out-root "$RESUME_ROOT" \
    --solver-commit "$SOLVER_COMMIT" \
    --parent-wall-limit-s "$PARENT_CAP" --wall-limit-s "$CHILD_CAP" \
    --expected-cells 23
fi

"$PYTHON_BIN" "$SCRIPT_DIR/repair_cg_resume_telemetry.py" \
  --resume-root "$RESUME_ROOT"

COUNTS=$(
  awk -F'\t' 'FNR > 1 {n[$4]++} END {
    printf "5=%d,8=%d,9=%d,10=%d", n[5]+0,n[8]+0,n[9]+0,n[10]+0
  }' "$RESUME_ROOT/matrix.tsv"
)
[[ "$COUNTS" == "5=4,8=8,9=4,10=7" ]] \
  || evsp_die "unexpected staged scale counts: $COUNTS"

PENDING=$(mktemp)
trap 'rm -f "$PENDING"' EXIT
"$PYTHON_BIN" "$SCRIPT_DIR/select_cg_resume_indices.py" \
  --resume-root "$RESUME_ROOT" --expected-commit "$SOLVER_COMMIT" \
  --expected-wall-limit-s "$CHILD_CAP" > "$PENDING"

submitted=0
for scale in 5 8 9 10; do
  mapfile -t indices < <(
    awk -F'\t' -v scale="$scale" \
      'NR==FNR {pending[$1]=1; next}
       FNR>1 && pending[$1] && $4==scale {print $1}' \
      "$PENDING" "$RESUME_ROOT/matrix.tsv"
  )
  [[ ${#indices[@]} -gt 0 ]] || continue
  name="st48k${scale}"
  active=$(
    squeue --me -h -o '%A|%j' |
      awk -F'|' -v name="$name" '$2==name {print $1}' | sort -u
  )
  [[ "$(printf '%s\n' "$active" | awk 'NF {n++} END {print n+0}')" -le 1 ]] \
    || evsp_die "multiple active $name arrays"
  if [[ -n "$active" ]]; then
    echo "k$scale continuation already active: $active"
    continue
  fi
  array=$(IFS=,; echo "${indices[*]}")
  count=${#indices[@]}
  exports="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$RESUME_ROOT,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CUMULATIVE_WALL_LIMIT_S=$CHILD_CAP,EVSP_MAX_ITERS=50000,EVSP_TIME_MODEL=event,EVSP_EVENT_ARC_MODE=lazy,EVSP_PYTHON=$PYTHON_BIN"
  job=$(evsp_submit_and_resolve "$name" \
    --array="$array%$count" -p default_partition -c 1 --mem=48G \
    -t 1-12:30:00 --requeue --open-mode=append --signal=B:TERM@180 \
    --export="$exports" \
    -o "$RESUME_ROOT/logs/${name}_%A_%a.out" \
    -e "$RESUME_ROOT/logs/${name}_%A_%a.err" \
    "$SCRIPT_DIR/cg_resume_extended.sub")
  {
    printf 'stage\tarray_job_id\ttasks\tindices\tscale\tpartition\tmem\ttimelimit\twrapper_commit\tsolver_commit\tcumulative_wall_limit_s\n'
    printf 'small_threshold_resume48h\t%s\t%s\t%s\t%s\tdefault_partition\t48G\t1-12:30:00\t%s\t%s\t%s\n' \
      "$job" "$count" "$array" "$scale" "$WRAPPER_COMMIT" \
      "$SOLVER_COMMIT" "$CHILD_CAP"
  } > "$RESUME_ROOT/jobs_${job}.tsv"
  submitted=$((submitted + count))
  echo "k$scale cumulative-48h continuation: $job ($count tasks)"
done

if compgen -G "$RESUME_ROOT/jobs_*.tsv" > /dev/null; then
  sha256sum "$RESUME_ROOT/execution_plan.json" \
    "$RESUME_ROOT/matrix.tsv" "$RESUME_ROOT"/jobs_*.tsv \
    > "$RESUME_ROOT/SUBMISSION_INPUT_SHA256SUMS"
fi
echo "Submitted continuation tasks: $submitted"
echo "Resume root: $RESUME_ROOT"
