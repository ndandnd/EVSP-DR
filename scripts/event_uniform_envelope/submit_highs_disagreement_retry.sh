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
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
PYTHON_TAG=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
HIGHS_TARGET="$HOME/ladder-lite/vendor/highspy-1.15.1-py${PYTHON_TAG}"
PYTHONPATH="$HIGHS_TARGET" "$PYTHON_BIN" -c \
  'import highspy; assert highspy.Highs().version() == "1.15.1"'
[[ -f "$A_ROOT/panel_a_highs_inputs.tsv" ]] || evsp_die "missing Panel A manifest"
[[ -f "$B_ROOT/panel_b_highs_inputs.tsv" ]] || evsp_die "missing Panel B manifest"
if squeue --me -h -o '%j' | grep -qE '^(eua27_h2|eub27_h2)$'; then
  evsp_die "two-hour HiGHS retry already active"
fi

A_OUT="$A_ROOT/mip_highs_native_retry7200"
B_OUT="$B_ROOT/mip_highs_native_retry7200"
mkdir -p "$A_OUT" "$B_OUT"
A_INDICES="38,44,45,47,48,50,51,52,53"
B_INDICES="31,33,36,37,41,42,43"
validate_absent() {
  local panel="$1" manifest="$2" out="$3" indices="$4"
  local index line cell rep
  for index in ${indices//,/ }; do
    line=$(sed -n "$((index + 2))p" "$manifest")
    IFS=$'\t' read -r _ cell _ rep _ _ _ _ <<< "$line"
    [[ ! -e "$out/${panel}__${cell}__${rep}.json" ]] \
      || evsp_die "retry artifact already exists for $panel index $index"
  done
}
validate_absent A "$A_ROOT/panel_a_highs_inputs.tsv" "$A_OUT" "$A_INDICES"
validate_absent B "$B_ROOT/panel_b_highs_inputs.tsv" "$B_OUT" "$B_INDICES"

COMMON="EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_HIGHS_PYTHONPATH=$HIGHS_TARGET,EVSP_MIP_TIMELIMIT_S=7200,EVSP_MIP_THREADS=8,EVSP_PYTHON=$PYTHON_BIN"
A_EXPORTS="ALL,$COMMON,EVSP_PANEL=A,EVSP_INTEGER_MANIFEST=$A_ROOT/panel_a_highs_inputs.tsv,EVSP_HIGHS_OUTPUT_DIR=$A_OUT"
B_EXPORTS="ALL,$COMMON,EVSP_PANEL=B,EVSP_INTEGER_MANIFEST=$B_ROOT/panel_b_highs_inputs.tsv,EVSP_HIGHS_OUTPUT_DIR=$B_OUT"
A_JOB=$(evsp_submit_and_resolve eua27_h2 \
  --array="$A_INDICES%9" -p scaglione -c 8 --mem=24G -t 02:30:00 \
  --no-requeue --export="$A_EXPORTS" \
  -o "$A_ROOT/logs/highs2_%A_%a.out" -e "$A_ROOT/logs/highs2_%A_%a.err" \
  "$SCRIPT_DIR/pool_mip_highs_native.sub")
B_JOB=$(evsp_submit_and_resolve eub27_h2 \
  --array="$B_INDICES%7" -p scaglione -c 8 --mem=24G -t 02:30:00 \
  --no-requeue --export="$B_EXPORTS" \
  -o "$B_ROOT/logs/highs2_%A_%a.out" -e "$B_ROOT/logs/highs2_%A_%a.err" \
  "$SCRIPT_DIR/pool_mip_highs_native.sub")
{
  echo -e "panel\tarray_job_id\tindices\tsolver_commit\tbackend\ttimelimit_s\tthreads\tpartition"
  echo -e "A\t$A_JOB\t$A_INDICES\t$SOLVER_COMMIT\thighspy_native\t7200\t8\tscaglione"
  echo -e "B\t$B_JOB\t$B_INDICES\t$SOLVER_COMMIT\thighspy_native\t7200\t8\tscaglione"
} | tee "$A_ROOT/highs_disagreement_retry_jobs.tsv"
cp "$A_ROOT/highs_disagreement_retry_jobs.tsv" \
  "$B_ROOT/highs_disagreement_retry_jobs.tsv"
echo "Panel A two-hour HiGHS retry: $A_JOB (9 tasks)"
echo "Panel B two-hour HiGHS retry: $B_JOB (7 tasks)"
