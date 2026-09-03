#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
ROOT="${1:-$HOME/ladder-lite/cg_acceleration_20260903}"
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
SOLVER_COMMIT="$WRAPPER_COMMIT"
INPUT_COMMIT="ff7fb2ba93cf13a31171e1e4aeb2d28dc8aeee20"
INPUT_BRANCH="cursor/ladder-lite-20260819-2969"

REMOTE=$(git -C "$REPO" ls-remote --heads origin "${INPUT_BRANCH}*") \
  || evsp_die "git ls-remote failed for $INPUT_BRANCH"
printf '%s\n' "$REMOTE" >&2
[[ "$(printf '%s\n' "$REMOTE" | awk 'NF {n++} END {print n+0}')" == 1 ]] \
  || evsp_die "expected exactly one branch matching $INPUT_BRANCH*"
[[ "$(printf '%s\n' "$REMOTE" | awk '{print $2}')" == \
  "refs/heads/$INPUT_BRANCH" ]] || evsp_die "unexpected input branch"
INPUT_TIP=$(printf '%s\n' "$REMOTE" | awk '{print $1}')
git -C "$REPO" fetch origin \
  "refs/heads/$INPUT_BRANCH:refs/remotes/origin/$INPUT_BRANCH"
git -C "$REPO" merge-base --is-ancestor "$INPUT_COMMIT" "$INPUT_TIP" \
  || evsp_die "reviewed input commit is not an ancestor of input tip"

if squeue --me -h -o '%j' | grep -qE '^ca03_'; then
  evsp_die "CG acceleration campaign already active"
fi
[[ ! -e "$ROOT" ]] || evsp_die "campaign root already exists: $ROOT"

SOLVER_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
INPUT_REPO=$(evsp_execution_checkout "$REPO" "$INPUT_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
HELP_TEXT=$("$PYTHON_BIN" "$SOLVER_REPO/src/exact_pricer_expanded.py" --help)
grep -q -- '--event-network-cache' <<< "$HELP_TEXT" \
  || evsp_die "solver lacks event-network caching"
grep -q -- '--column-selection' <<< "$HELP_TEXT" \
  || evsp_die "solver lacks complementary column selection"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_cg_acceleration.py" \
  --input-repo "$INPUT_REPO" --solver-repo "$SOLVER_REPO" \
  --root "$ROOT" --input-commit "$INPUT_COMMIT" \
  --solver-commit "$SOLVER_COMMIT" --wrapper-commit "$WRAPPER_COMMIT"

COMMON="EVSP_EXECUTION_REPO=$SOLVER_REPO,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_PYTHON=$PYTHON_BIN"
RECORD="$ROOT/jobs.tsv"
printf 'stage\tarm\tarray_job_id\tindices\tpartition\tmem\ttimelimit\tdependency\tcolumns_per_iter\tselection\tdiversity_weight\twrapper_commit\tsolver_commit\tinput_commit\n' > "$RECORD"

CACHE_JOB=$(evsp_submit_and_resolve ca03_net \
  --array='0-11%12' -p default_partition -c 1 --mem=96G \
  -t 12:15:00 --requeue --open-mode=append --signal=B:TERM@180 \
  --export="ALL,$COMMON" \
  -o "$ROOT/logs/cache/%x_%A_%a.out" \
  -e "$ROOT/logs/cache/%x_%A_%a.err" \
  "$SCRIPT_DIR/event_network_cache.sub")
printf 'cache\tNA\t%s\t0-11\tdefault_partition\t96G\t12:15:00\tNA\tNA\tNA\tNA\t%s\t%s\t%s\n' \
  "$CACHE_JOB" "$WRAPPER_COMMIT" "$SOLVER_COMMIT" "$INPUT_COMMIT" \
  >> "$RECORD"

submit_arm() {
  local name="$1" arm="$2" columns="$3" selection="$4" weight="$5"
  local exports job
  exports="$COMMON,EVSP_ARM=$arm,EVSP_COLUMNS_PER_ITER=$columns,EVSP_COLUMN_SELECTION=$selection,EVSP_COLUMN_DIVERSITY_WEIGHT=$weight"
  job=$(evsp_submit_and_resolve "$name" \
    --array='0-11%12' -p default_partition -c 1 --mem=96G \
    -t 12:15:00 --requeue --open-mode=append --signal=B:TERM@180 \
    --dependency="aftercorr:$CACHE_JOB" --export="ALL,$exports" \
    -o "$ROOT/logs/cg/${name}_%A_%a.out" \
    -e "$ROOT/logs/cg/${name}_%A_%a.err" \
    "$SCRIPT_DIR/cg_acceleration.sub")
  printf 'cg\t%s\t%s\t0-11\tdefault_partition\t96G\t12:15:00\taftercorr:%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$job" "$CACHE_JOB" "$columns" "$selection" "$weight" \
    "$WRAPPER_COMMIT" "$SOLVER_COMMIT" "$INPUT_COMMIT" >> "$RECORD"
  echo "$name: $job"
}

submit_arm ca03_30r b030_reduced 30 reduced_cost 0.0
submit_arm ca03_60r b060_reduced 60 reduced_cost 0.0
submit_arm ca03_12r b120_reduced 120 reduced_cost 0.0
submit_arm ca03_20r b200_reduced 200 reduced_cost 0.0
submit_arm ca03_12d b120_complementary 120 complementary 0.5
submit_arm ca03_20d b200_complementary 200 complementary 0.5

sha256sum "$ROOT/matrix.tsv" "$ROOT/execution_plan.json" "$RECORD" \
  > "$ROOT/SUBMISSION_SHA256SUMS"
echo "CG acceleration campaign: $ROOT"
echo "Cache array: $CACHE_JOB"
echo "Machine-readable jobs: $RECORD"
