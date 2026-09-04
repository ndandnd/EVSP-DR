#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
ROOT="${1:-$HOME/ladder-lite/cg_acceleration_20260903}"
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
INPUT_COMMIT=$(
  "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["input_commit"])' \
    "$ROOT/execution_plan.json"
)
SOLVER_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")

INDEX=9
LINE=$(sed -n "$((INDEX + 1))p" "$ROOT/matrix.tsv")
IFS=$'\t' read -r observed cell scale replicate trips csv_path csv_sha rep \
  soc block wall <<< "$LINE"
[[ "$observed" == "$INDEX" && "$cell" == "k20_s4" ]]
[[ "$(sha256sum "$csv_path" | awk '{print $1}')" == "$csv_sha" ]]
CACHE="$ROOT/network_cache/M__${cell}__${rep}.pkl"
[[ ! -e "$CACHE" && ! -e "$CACHE.manifest.json" ]] \
  || evsp_die "cache already exists for $cell"
if squeue --me -h -o '%j' | grep -qE '^ca04_'; then
  evsp_die "acceleration recovery already active"
fi

COMMON="EVSP_EXECUTION_REPO=$SOLVER_REPO,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_PYTHON=$PYTHON_BIN"
CACHE_JOB=$(evsp_submit_and_resolve ca04_net \
  --array='9%1' -p default_partition -c 1 --mem=96G \
  -t 24:30:00 --requeue --open-mode=append --signal=B:TERM@180 \
  --export="ALL,$COMMON" \
  -o "$ROOT/logs/cache/ca04_net_%A_%a.out" \
  -e "$ROOT/logs/cache/ca04_net_%A_%a.err" \
  "$SCRIPT_DIR/event_network_cache.sub")

RECORD="$ROOT/jobs_recovery_${CACHE_JOB}.tsv"
printf 'stage\tarm\tarray_job_id\tindices\tpartition\tmem\ttimelimit\tdependency\tcolumns_per_iter\tselection\tdiversity_weight\twrapper_commit\tsolver_commit\tinput_commit\n' \
  > "$RECORD"
printf 'cache_retry\tNA\t%s\t9\tdefault_partition\t96G\t24:30:00\tNA\tNA\tNA\tNA\t%s\t%s\t%s\n' \
  "$CACHE_JOB" "$WRAPPER_COMMIT" "$SOLVER_COMMIT" "$INPUT_COMMIT" \
  >> "$RECORD"

submit_arm() {
  local name="$1" arm="$2" columns="$3" selection="$4" weight="$5"
  local exports job
  exports="$COMMON,EVSP_ARM=$arm,EVSP_COLUMNS_PER_ITER=$columns,EVSP_COLUMN_SELECTION=$selection,EVSP_COLUMN_DIVERSITY_WEIGHT=$weight"
  job=$(evsp_submit_and_resolve "$name" \
    --array='9%1' -p default_partition -c 1 --mem=96G \
    -t 12:15:00 --requeue --open-mode=append --signal=B:TERM@180 \
    --dependency="aftercorr:$CACHE_JOB" --export="ALL,$exports" \
    -o "$ROOT/logs/cg/${name}_%A_%a.out" \
    -e "$ROOT/logs/cg/${name}_%A_%a.err" \
    "$SCRIPT_DIR/cg_acceleration.sub")
  printf 'cg\t%s\t%s\t9\tdefault_partition\t96G\t12:15:00\taftercorr:%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$job" "$CACHE_JOB" "$columns" "$selection" "$weight" \
    "$WRAPPER_COMMIT" "$SOLVER_COMMIT" "$INPUT_COMMIT" >> "$RECORD"
  echo "$name: $job"
}

submit_arm ca04_30r b030_reduced 30 reduced_cost 0.0
submit_arm ca04_60r b060_reduced 60 reduced_cost 0.0
submit_arm ca04_12r b120_reduced 120 reduced_cost 0.0
submit_arm ca04_20r b200_reduced 200 reduced_cost 0.0
submit_arm ca04_12d b120_complementary 120 complementary 0.5
submit_arm ca04_20d b200_complementary 200 complementary 0.5

STALE=()
while IFS=$'\t' read -r stage arm job _rest; do
  [[ "$stage" == "cg" ]] || continue
  task="${job}_9"
  state=$(squeue --me -h -j "$task" -o '%T|%R')
  if [[
    "$state" != 'PENDING|DependencyNeverSatisfied'
    && "$state" != 'PENDING|(DependencyNeverSatisfied)'
  ]]; then
    evsp_die "stale task is not dependency-blocked: $task $state"
  fi
  STALE+=("$task")
done < <(tail -n +2 "$ROOT/jobs.tsv")
[[ ${#STALE[@]} -eq 6 ]] || evsp_die "expected six stale dependent tasks"
scancel "${STALE[@]}"

sha256sum "$ROOT/matrix.tsv" "$ROOT/execution_plan.json" "$RECORD" \
  > "$ROOT/RECOVERY_${CACHE_JOB}_SHA256SUMS"
echo "k20_s4 cache recovery: $CACHE_JOB"
echo "Cancelled six superseded DependencyNeverSatisfied tasks."
echo "Recovery record: $RECORD"
