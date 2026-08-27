#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT PANEL_B_ROOT"
A_ROOT=$(cd "$1" && pwd)
B_ROOT=$(cd "$2" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
SOLVER_COMMIT="44b6d5030a78ddca9c74f582d70ad87572e61794"
AGENT_BRANCH="cursor/event-based-pricer-2969"
REMOTE=$(git -C "$REPO" ls-remote --heads origin "${AGENT_BRANCH}*")
printf '%s\n' "$REMOTE" >&2
[[ "$(printf '%s\n' "$REMOTE" | awk 'NF {n++} END {print n+0}')" == 1 ]] \
  || evsp_die "expected exactly one Agent E branch"
[[ "$(printf '%s\n' "$REMOTE" | awk '{print $2}')" == "refs/heads/$AGENT_BRANCH" ]] \
  || evsp_die "unexpected Agent E ref"
AGENT_SHA=$(printf '%s\n' "$REMOTE" | awk '{print $1}')
git -C "$REPO" fetch origin \
  "refs/heads/$AGENT_BRANCH:refs/remotes/origin/$AGENT_BRANCH"
git -C "$REPO" merge-base --is-ancestor "$SOLVER_COMMIT" "$AGENT_SHA" \
  || evsp_die "reviewed solver commit is not an ancestor of Agent E tip"

PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
mapfile -t A_SELECTED < <(
  "$PYTHON_BIN" "$SCRIPT_DIR/select_highs_unresolved_retry_indices.py" \
    --root "$A_ROOT" --panel A
)
mapfile -t B_SELECTED < <(
  "$PYTHON_BIN" "$SCRIPT_DIR/select_highs_unresolved_retry_indices.py" \
    --root "$B_ROOT" --panel B
)
TOTAL=$(( ${#A_SELECTED[@]} + ${#B_SELECTED[@]} ))
[[ $TOTAL -gt 0 ]] || { echo "all two-hour disagreements resolved; nothing submitted"; exit 0; }
[[ ! -e "$A_ROOT/highs_unresolved_retry28800_jobs.tsv" ]] \
  || evsp_die "Panel A eight-hour job record already exists"
[[ ! -e "$B_ROOT/highs_unresolved_retry28800_jobs.tsv" ]] \
  || evsp_die "Panel B eight-hour job record already exists"
if squeue --me -h -o '%j' | grep -qE '^(eua27_h8|eub27_h8)$'; then
  evsp_die "eight-hour HiGHS retry already active"
fi

EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
PYTHON_TAG=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
HIGHS_TARGET="$HOME/ladder-lite/vendor/highspy-1.15.1-py${PYTHON_TAG}"
PYTHONPATH="$HIGHS_TARGET" "$PYTHON_BIN" -c \
  'import highspy; assert highspy.Highs().version() == "1.15.1"'
A_OUT="$A_ROOT/mip_highs_native_retry28800"
B_OUT="$B_ROOT/mip_highs_native_retry28800"
mkdir -p "$A_OUT" "$B_OUT"

validate_absent() {
  local panel="$1" manifest="$2" out="$3"
  shift 3
  local index line cell rep
  for index in "$@"; do
    line=$(sed -n "$((index + 2))p" "$manifest")
    IFS=$'\t' read -r _ cell _ rep _ _ _ _ <<< "$line"
    [[ ! -e "$out/${panel}__${cell}__${rep}.json" ]] \
      || evsp_die "eight-hour artifact already exists for $panel index $index"
  done
}
validate_absent A "$A_ROOT/panel_a_highs_inputs.tsv" "$A_OUT" "${A_SELECTED[@]}"
validate_absent B "$B_ROOT/panel_b_highs_inputs.tsv" "$B_OUT" "${B_SELECTED[@]}"

join_indices() { local IFS=,; echo "$*"; }
A_INDICES=$(join_indices "${A_SELECTED[@]}")
B_INDICES=$(join_indices "${B_SELECTED[@]}")
COMMON="EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_HIGHS_PYTHONPATH=$HIGHS_TARGET,EVSP_MIP_TIMELIMIT_S=28800,EVSP_MIP_THREADS=8,EVSP_PYTHON=$PYTHON_BIN"
A_JOB=""
B_JOB=""
RECORD="$A_ROOT/highs_unresolved_retry28800_jobs.tsv"
echo -e "panel\tarray_job_id\tindices\twrapper_commit\tsolver_commit\tbackend\ttimelimit_s\tthreads\tpartition" > "$RECORD"
cp "$RECORD" "$B_ROOT/highs_unresolved_retry28800_jobs.tsv"
if [[ ${#A_SELECTED[@]} -gt 0 ]]; then
  A_EXPORTS="ALL,$COMMON,EVSP_PANEL=A,EVSP_INTEGER_MANIFEST=$A_ROOT/panel_a_highs_inputs.tsv,EVSP_HIGHS_OUTPUT_DIR=$A_OUT"
  A_JOB=$(evsp_submit_and_resolve eua27_h8 \
    --array="$A_INDICES%${#A_SELECTED[@]}" -p scaglione -c 8 --mem=24G -t 08:30:00 \
    --no-requeue --export="$A_EXPORTS" \
    -o "$A_ROOT/logs/highs8_%A_%a.out" -e "$A_ROOT/logs/highs8_%A_%a.err" \
    "$SCRIPT_DIR/pool_mip_highs_native.sub")
  echo -e "A\t$A_JOB\t$A_INDICES\t$WRAPPER_COMMIT\t$SOLVER_COMMIT\thighspy_native\t28800\t8\tscaglione" >> "$RECORD"
  cp "$RECORD" "$B_ROOT/highs_unresolved_retry28800_jobs.tsv"
fi
if [[ ${#B_SELECTED[@]} -gt 0 ]]; then
  B_EXPORTS="ALL,$COMMON,EVSP_PANEL=B,EVSP_INTEGER_MANIFEST=$B_ROOT/panel_b_highs_inputs.tsv,EVSP_HIGHS_OUTPUT_DIR=$B_OUT"
  B_JOB=$(evsp_submit_and_resolve eub27_h8 \
    --array="$B_INDICES%${#B_SELECTED[@]}" -p scaglione -c 8 --mem=24G -t 08:30:00 \
    --no-requeue --export="$B_EXPORTS" \
    -o "$B_ROOT/logs/highs8_%A_%a.out" -e "$B_ROOT/logs/highs8_%A_%a.err" \
    "$SCRIPT_DIR/pool_mip_highs_native.sub")
  echo -e "B\t$B_JOB\t$B_INDICES\t$WRAPPER_COMMIT\t$SOLVER_COMMIT\thighspy_native\t28800\t8\tscaglione" >> "$RECORD"
  cp "$RECORD" "$B_ROOT/highs_unresolved_retry28800_jobs.tsv"
fi
echo "Panel A eight-hour HiGHS retry: ${A_JOB:-skipped} (${#A_SELECTED[@]} tasks)"
echo "Panel B eight-hour HiGHS retry: ${B_JOB:-skipped} (${#B_SELECTED[@]} tasks)"
