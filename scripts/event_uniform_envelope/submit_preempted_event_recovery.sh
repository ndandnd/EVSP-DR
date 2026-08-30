#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 2 ]] || evsp_die "usage: $0 [MEDIUM_ROOT] [EXTENSION_ROOT]"

MEDIUM_ROOT="${1:-$HOME/ladder-lite/medium_event_legacy_20260830_44b6d5}"
EXTENSION_ROOT="${2:-$HOME/ladder-lite/event_extension_20260830_44b6d5}"
MEDIUM_ROOT=$(cd "$MEDIUM_ROOT" && pwd)
EXTENSION_ROOT=$(cd "$EXTENSION_ROOT" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
SOLVER_COMMIT="44b6d5030a78ddca9c74f582d70ad87572e61794"
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
WALL_LIMIT=43200

MEDIUM_RESUME="$MEDIUM_ROOT/preempt_recovery_i12_20260830"
EXTENSION_RESUME="$EXTENSION_ROOT/preempt_recovery_i12_20260830"

prepare_one() {
  local source_root="$1" out_root="$2" source_index="$3"
  if [[ ! -e "$out_root" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_preempted_event_resume.py" \
      --source-root "$source_root" --out-root "$out_root" \
      --source-index "$source_index" --solver-commit "$SOLVER_COMMIT" \
      --wall-limit-s "$WALL_LIMIT"
  fi
  "$PYTHON_BIN" "$SCRIPT_DIR/repair_cg_resume_telemetry.py" \
    --resume-root "$out_root"
}

prepare_one "$MEDIUM_ROOT" "$MEDIUM_RESUME" 12
prepare_one "$EXTENSION_ROOT" "$EXTENSION_RESUME" 12

submit_one() {
  local root="$1" name="$2" memory="$3" stage="$4"
  local index_file active array exports job
  index_file=$(mktemp)
  "$PYTHON_BIN" "$SCRIPT_DIR/select_cg_resume_indices.py" \
    --resume-root "$root" --expected-commit "$SOLVER_COMMIT" \
    --expected-wall-limit-s "$WALL_LIMIT" > "$index_file"
  if [[ ! -s "$index_file" ]]; then
    echo "$stage: no pending cell"
    rm -f "$index_file"
    return
  fi
  active=$(
    squeue --me -h -o '%A|%j' |
      awk -F'|' -v name="$name" '$2 == name {print $1}' |
      sort -u
  )
  [[ "$(printf '%s\n' "$active" | awk 'NF {n++} END {print n+0}')" -le 1 ]] \
    || evsp_die "multiple active $name jobs"
  if [[ -n "$active" ]]; then
    echo "$stage already active: $active"
    rm -f "$index_file"
    return
  fi
  array=$(paste -sd, "$index_file")
  rm -f "$index_file"
  exports="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$root,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CUMULATIVE_WALL_LIMIT_S=$WALL_LIMIT,EVSP_MAX_ITERS=10000,EVSP_PYTHON=$PYTHON_BIN"
  job=$(evsp_submit_and_resolve "$name" \
    --array="$array%1" -p default_partition -c 1 --mem="$memory" \
    -t 10:00:00 --no-requeue --signal=B:TERM@180 \
    --export="$exports" \
    -o "$root/logs/resume_%A_%a.out" \
    -e "$root/logs/resume_%A_%a.err" \
    "$SCRIPT_DIR/cg_resume_extended.sub")
  {
    echo -e "stage\tarray_job_id\ttasks\tindices\tpartition\tmem\ttimelimit\twrapper_commit\tsolver_commit\tcumulative_wall_limit_s"
    echo -e "$stage\t$job\t1\t$array\tdefault_partition\t$memory\t10:00:00\t$WRAPPER_COMMIT\t$SOLVER_COMMIT\t$WALL_LIMIT"
  } > "$root/jobs_${job}.tsv"
  sha256sum "$root/execution_plan.json" "$root/matrix.tsv" \
    "$root/jobs_${job}.tsv" > "$root/SUBMISSION_INPUT_SHA256SUMS"
  echo "$stage: $job (1 task)"
}

submit_one "$MEDIUM_RESUME" med30_r20 96G medium_k20_preempt_recovery
submit_one "$EXTENSION_RESUME" ext30_r40 192G extension_k40_preempt_recovery

echo "Original running cells were not duplicated."
echo "Medium recovery root: $MEDIUM_RESUME"
echo "Extension recovery root: $EXTENSION_RESUME"
