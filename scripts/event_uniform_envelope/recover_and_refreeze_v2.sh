#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT PANEL_B_ROOT"

A_ROOT=$(cd "$1" && pwd)
B_ROOT=$(cd "$2" && pwd)
REPO=$(evsp_repo_root)
MANAGER_BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$MANAGER_BRANCH" ]] || evsp_die "manager checkout must be on a named branch"
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$MANAGER_BRANCH" | tail -1)

# Reviewed Agent E boundary/replay correction.  A later remote tip is allowed
# only when it retains this exact commit as an ancestor.
AGENT_BRANCH="cursor/event-based-pricer-2969"
SOLVER_COMMIT="44b6d5030a78ddca9c74f582d70ad87572e61794"
REMOTE=$(
  git -C "$REPO" ls-remote --heads origin "${AGENT_BRANCH}*"
) || evsp_die "could not resolve Agent E branch"
printf '%s\n' "$REMOTE" >&2
[[ "$(printf '%s\n' "$REMOTE" | awk 'NF {n++} END {print n+0}')" == 1 ]] \
  || evsp_die "expected exactly one Agent E branch"
[[ "$(printf '%s\n' "$REMOTE" | awk '{print $2}')" == "refs/heads/$AGENT_BRANCH" ]] \
  || evsp_die "unexpected Agent E ref"
REMOTE_SHA=$(printf '%s\n' "$REMOTE" | awk '{print $1}')
git -C "$REPO" fetch origin \
  "refs/heads/$AGENT_BRANCH:refs/remotes/origin/$AGENT_BRANCH"
git -C "$REPO" merge-base --is-ancestor "$SOLVER_COMMIT" "$REMOTE_SHA" \
  || evsp_die "reviewed Agent E commit is not an ancestor of the remote tip"
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"

if squeue --me -h -o '%j' | grep -qE '^eu[ab]24_(mip2|tf2|fz2)$'; then
  evsp_die "v2 recovery/refreeze jobs already exist"
fi

MANIFEST="$A_ROOT/panel_a_integer_inputs_v2.tsv"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_integer_manifest.py" \
  --root "$A_ROOT" --panel A --source-dir cg --out "$MANIFEST" \
  --provenance "$A_ROOT/integer_recovery_v2_provenance.json" \
  --wrapper-commit "$WRAPPER_COMMIT" --solver-commit "$SOLVER_COMMIT"
sha256sum "$MANIFEST" > "$A_ROOT/panel_a_integer_inputs_v2.sha256"

mapfile -t MIP_INDICES < <(
  tail -n +2 "$MANIFEST" |
    while IFS=$'\t' read -r index cell target rep source rest; do
      [[ -s "$A_ROOT/mip/A__${cell}__${rep}.json" ]] || echo "$index"
    done
)
mapfile -t TARGET_INDICES < <(
  tail -n +2 "$MANIFEST" |
    while IFS=$'\t' read -r index cell target rep source rest; do
      [[ -s "$A_ROOT/target/A__${cell}__${rep}.json" ]] || echo "$index"
    done
)
[[ ${#MIP_INDICES[@]} -gt 0 ]] || evsp_die "no missing Panel A MIP outputs"
[[ ${#TARGET_INDICES[@]} -gt 0 ]] || evsp_die "no missing Panel A target outputs"
MIP_ARRAY=$(IFS=,; echo "${MIP_INDICES[*]}")
TARGET_ARRAY=$(IFS=,; echo "${TARGET_INDICES[*]}")

EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$A_ROOT,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_PANEL=A,EVSP_INTEGER_MANIFEST=$MANIFEST,EVSP_MIP_OUTPUT_DIR=$A_ROOT/mip,EVSP_TARGET_OUTPUT_DIR=$A_ROOT/target,EVSP_PYTHON=$PYTHON_BIN"
MIP_JOB=$(evsp_submit_and_resolve eua24_mip2 \
  --array="${MIP_ARRAY}%54" -p default_partition -c 8 --mem=24G -t 00:45:00 \
  --no-requeue --export="$EXPORTS" \
  -o "$A_ROOT/logs/mip2_%A_%a.out" -e "$A_ROOT/logs/mip2_%A_%a.err" \
  "$SCRIPT_DIR/pool_mip.sub")
sleep 1
TF_JOB=$(evsp_submit_and_resolve eua24_tf2 \
  --array="$TARGET_ARRAY" -p default_partition -c 8 --mem=24G -t 00:45:00 \
  --no-requeue --export="$EXPORTS" \
  -o "$A_ROOT/logs/tf2_%A_%a.out" -e "$A_ROOT/logs/tf2_%A_%a.err" \
  "$SCRIPT_DIR/target_feasibility.sub")

FROZEN_V2="$B_ROOT/frozen_v2"
[[ ! -e "$FROZEN_V2" ]] || evsp_die "v2 frozen directory already exists: $FROZEN_V2"
mkdir -p "$FROZEN_V2"
FREEZE_EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$B_ROOT,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_FROZEN_OUTPUT_DIR=$FROZEN_V2,EVSP_PYTHON=$PYTHON_BIN"
FREEZE_JOB=$(evsp_submit_and_resolve eub24_fz2 \
  --array=0-44%45 -p default_partition -c 1 --mem=4G -t 00:15:00 \
  --no-requeue --export="$FREEZE_EXPORTS" \
  -o "$B_ROOT/logs/fz2_%A_%a.out" -e "$B_ROOT/logs/fz2_%A_%a.err" \
  "$SCRIPT_DIR/panel_b_freeze.sub")

{
  echo -e "stage\tarray_job_id\ttasks\tsolver_commit"
  echo -e "mip_recovery\t$MIP_JOB\t${#MIP_INDICES[@]}\t$SOLVER_COMMIT"
  echo -e "target_recovery\t$TF_JOB\t${#TARGET_INDICES[@]}\t$SOLVER_COMMIT"
} | tee "$A_ROOT/integer_recovery_v2_jobs.tsv"
{
  echo -e "stage\tarray_job_id\ttasks\tsolver_commit"
  echo -e "freeze_v2\t$FREEZE_JOB\t45\t$SOLVER_COMMIT"
} | tee "$B_ROOT/refreeze_v2_jobs.tsv"
sha256sum \
  "$A_ROOT/panel_a_integer_inputs_v2.tsv" \
  "$A_ROOT/integer_recovery_v2_provenance.json" \
  "$A_ROOT/integer_recovery_v2_jobs.tsv" \
  "$B_ROOT/refreeze_v2_jobs.tsv" \
  > "$B_ROOT/RECOVERY_V2_SHA256SUMS"

echo "Panel A MIP recovery: $MIP_JOB (${#MIP_INDICES[@]} tasks)"
echo "Panel A target retry: $TF_JOB (${#TARGET_INDICES[@]} tasks)"
echo "Panel B v2 refreeze: $FREEZE_JOB (45 tasks)"
