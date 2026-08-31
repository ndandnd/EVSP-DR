#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
if [[ "${EVSP_DEFERRED_CONTROLLER:-0}" == "1" ]]; then
  [[ "$(id -un)" == "nc437" ]] || evsp_die "expected user nc437"
  [[ -n "${SLURM_JOB_ID:-}" ]] || evsp_die "controller mode requires Slurm"
else
  evsp_require_unicorn
fi
[[ $# -le 2 ]] || evsp_die "usage: $0 [MEDIUM_ROOT] [EXTENSION_ROOT]"

MEDIUM_ROOT="${1:-$HOME/ladder-lite/medium_event_corrected_20260831_44b6d5}"
EXTENSION_ROOT="${2:-$HOME/ladder-lite/event_extension_corrected_20260831_44b6d5}"
MEDIUM_ROOT=$(cd "$MEDIUM_ROOT" && pwd)
EXTENSION_ROOT=$(cd "$EXTENSION_ROOT" && pwd)
if [[ "${EVSP_DEFERRED_CONTROLLER:-0}" == "1" ]]; then
  REPO="${EVSP_WRAPPER_REPO:?}"
  WRAPPER_COMMIT="${EVSP_WRAPPER_COMMIT:?}"
  [[ "$(git -C "$REPO" rev-parse HEAD)" == "$WRAPPER_COMMIT" ]] \
    || evsp_die "controller wrapper commit mismatch"
  [[ -z "$(git -C "$REPO" status --porcelain)" ]] \
    || evsp_die "controller wrapper checkout is dirty"
else
  REPO=$(evsp_repo_root)
  BRANCH=$(git -C "$REPO" branch --show-current)
  WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
fi
SOLVER_COMMIT="44b6d5030a78ddca9c74f582d70ad87572e61794"
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
PARENT_CAP=43200
CHILD_CAP=86400

MEDIUM_RESUME="$MEDIUM_ROOT/cg_resume24h_20260831"
EXTENSION_RESUME="$EXTENSION_ROOT/cg_resume24h_20260831"

prepare_root() {
  local source_root="$1" resume_root="$2"
  if [[ ! -e "$resume_root" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/prepare_wall_capped_event_resume.py" \
      --source-root "$source_root" --out-root "$resume_root" \
      --solver-commit "$SOLVER_COMMIT" \
      --parent-wall-limit-s "$PARENT_CAP" --wall-limit-s "$CHILD_CAP"
  fi
  if [[ "$(awk 'END {print NR-1}' "$resume_root/matrix.tsv")" -gt 0 ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/repair_cg_resume_telemetry.py" \
      --resume-root "$resume_root"
  else
    echo "No qualified wall-capped cells under $source_root"
  fi
}

prepare_root "$MEDIUM_ROOT" "$MEDIUM_RESUME"
prepare_root "$EXTENSION_ROOT" "$EXTENSION_RESUME"

memory_for_scale() {
  case "$1" in
    2|3|5) echo 16G ;;
    8) echo 32G ;;
    13) echo 64G ;;
    20) echo 96G ;;
    30) echo 128G ;;
    40) echo 192G ;;
    *) evsp_die "no memory policy for scale $1" ;;
  esac
}

submit_root() {
  local root="$1" prefix="$2" category="$3"
  local pending_file scale memory name active array exports job count
  local -a indices
  local submitted=0
  pending_file=$(mktemp)
  "$PYTHON_BIN" "$SCRIPT_DIR/select_cg_resume_indices.py" \
    --resume-root "$root" --expected-commit "$SOLVER_COMMIT" \
    --expected-wall-limit-s "$CHILD_CAP" > "$pending_file"
  for scale in 2 3 5 8 13 20 30 40; do
    mapfile -t indices < <(
      awk -F'\t' -v scale="$scale" \
        'NR==FNR {pending[$1]=1; next} FNR>1 && pending[$1] && $4==scale {print $1}' \
        "$pending_file" "$root/matrix.tsv"
    )
    [[ ${#indices[@]} -gt 0 ]] || continue
    memory=$(memory_for_scale "$scale")
    name="${prefix}r${scale}"
    active=$(
      squeue --me -h -o '%A|%j' |
        awk -F'|' -v name="$name" '$2 == name {print $1}' |
        sort -u
    )
    [[ "$(printf '%s\n' "$active" | awk 'NF {n++} END {print n+0}')" -le 1 ]] \
      || evsp_die "multiple active $name jobs"
    if [[ -n "$active" ]]; then
      echo "$category scale $scale already active: $active"
      continue
    fi
    array=$(IFS=,; echo "${indices[*]}")
    count=${#indices[@]}
    exports="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$root,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_CUMULATIVE_WALL_LIMIT_S=$CHILD_CAP,EVSP_MAX_ITERS=20000,EVSP_TIME_MODEL=event,EVSP_EVENT_ARC_MODE=lazy,EVSP_PYTHON=$PYTHON_BIN"
    job=$(evsp_submit_and_resolve "$name" \
      --array="$array%$count" -p default_partition -c 1 --mem="$memory" \
      -t 12:30:00 --no-requeue --signal=B:TERM@180 \
      --export="$exports" \
      -o "$root/logs/resume_%A_%a.out" \
      -e "$root/logs/resume_%A_%a.err" \
      "$SCRIPT_DIR/cg_resume_extended.sub")
    {
      echo -e "stage\tarray_job_id\ttasks\tindices\tscale\tpartition\tmem\ttimelimit\twrapper_commit\tsolver_commit\tcumulative_wall_limit_s"
      echo -e "$category\t$job\t$count\t$array\t$scale\tdefault_partition\t$memory\t12:30:00\t$WRAPPER_COMMIT\t$SOLVER_COMMIT\t$CHILD_CAP"
    } > "$root/jobs_${job}.tsv"
    submitted=$((submitted + count))
    echo "$category scale $scale: $job ($count tasks)"
  done
  rm -f "$pending_file"
  if compgen -G "$root/jobs_*.tsv" > /dev/null; then
    sha256sum "$root/execution_plan.json" "$root/matrix.tsv" \
      "$root"/jobs_*.tsv > "$root/SUBMISSION_INPUT_SHA256SUMS"
  fi
  echo "$category submitted tasks: $submitted"
}

submit_root "$MEDIUM_RESUME" m31 medium_event_resume24h
submit_root "$EXTENSION_RESUME" x31 extension_event_resume24h

echo "Medium 24h resume root: $MEDIUM_RESUME"
echo "Extension 24h resume root: $EXTENSION_RESUME"
