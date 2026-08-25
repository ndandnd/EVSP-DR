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

# Commit 3dd2274 printed one blank stdout line for an empty missing-index set.
# Bash treated that line as one empty task specification and submitted these
# two wrapper failures.  Bind them into accounting without treating them as
# scientific executions.
EMPTY_ARRAY_JOBS="$A_ROOT/integer_recovery_v2_empty_array_jobs.tsv"
if [[ ! -e "$EMPTY_ARRAY_JOBS" ]]; then
  EMPTY_ARRAY_WRAPPER_COMMIT="3dd2274d49e93b09fc0294f6448fde79ff1bc5da"
  {
    echo -e "stage\tarray_job_id\ttasks\tsolver_commit\treason"
    echo -e "mip_empty_array_wrapper_failure\t587819\t0\t$EMPTY_ARRAY_WRAPPER_COMMIT\tempty_missing_index_line"
    echo -e "target_empty_array_wrapper_failure\t587820\t0\t$EMPTY_ARRAY_WRAPPER_COMMIT\tempty_missing_index_line"
  } > "$EMPTY_ARRAY_JOBS"
fi

MANIFEST="$A_ROOT/panel_a_integer_inputs_v2.tsv"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_integer_manifest.py" \
  --root "$A_ROOT" --panel A --source-dir cg --out "$MANIFEST" \
  --provenance "$A_ROOT/integer_recovery_v2_provenance.json" \
  --wrapper-commit "$WRAPPER_COMMIT" --solver-commit "$SOLVER_COMMIT"
sha256sum "$MANIFEST" > "$A_ROOT/panel_a_integer_inputs_v2.sha256"

MIP_INDEX_FILE=$(mktemp)
"$PYTHON_BIN" "$SCRIPT_DIR/select_missing_integer_indices.py" \
  --manifest "$MANIFEST" --root "$A_ROOT" --panel A --stage mip \
  > "$MIP_INDEX_FILE"
mapfile -t MIP_INDICES < "$MIP_INDEX_FILE"
rm -f "$MIP_INDEX_FILE"
TARGET_INDEX_FILE=$(mktemp)
"$PYTHON_BIN" "$SCRIPT_DIR/select_missing_integer_indices.py" \
  --manifest "$MANIFEST" --root "$A_ROOT" --panel A --stage target \
  > "$TARGET_INDEX_FILE"
mapfile -t TARGET_INDICES < "$TARGET_INDEX_FILE"
rm -f "$TARGET_INDEX_FILE"

EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$A_ROOT,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_PANEL=A,EVSP_INTEGER_MANIFEST=$MANIFEST,EVSP_MIP_OUTPUT_DIR=$A_ROOT/mip,EVSP_TARGET_OUTPUT_DIR=$A_ROOT/target,EVSP_PYTHON=$PYTHON_BIN"
MIP_JOB=""
if [[ ${#MIP_INDICES[@]} -gt 0 ]]; then
  MIP_ARRAY=$(IFS=,; echo "${MIP_INDICES[*]}")
  MIP_JOB=$(evsp_submit_and_resolve eua24_mip2 \
    --array="${MIP_ARRAY}%54" -p default_partition -c 8 --mem=24G -t 00:45:00 \
    --no-requeue --export="$EXPORTS" \
    -o "$A_ROOT/logs/mip2_%A_%a.out" -e "$A_ROOT/logs/mip2_%A_%a.err" \
    "$SCRIPT_DIR/pool_mip.sub")
  sleep 1
fi
TF_JOB=""
if [[ ${#TARGET_INDICES[@]} -gt 0 ]]; then
  TARGET_ARRAY=$(IFS=,; echo "${TARGET_INDICES[*]}")
  TF_JOB=$(evsp_submit_and_resolve eua24_tf2 \
    --array="$TARGET_ARRAY" -p default_partition -c 8 --mem=24G -t 00:45:00 \
    --no-requeue --export="$EXPORTS" \
    -o "$A_ROOT/logs/tf2_%A_%a.out" -e "$A_ROOT/logs/tf2_%A_%a.err" \
    "$SCRIPT_DIR/target_feasibility.sub")
fi

FROZEN_V2="$B_ROOT/frozen_v2"
mkdir -p "$FROZEN_V2"
FREEZE_INDEX_FILE=$(mktemp)
"$PYTHON_BIN" "$SCRIPT_DIR/select_missing_frozen_v2_indices.py" \
  --root "$B_ROOT" --output-dir "$FROZEN_V2" > "$FREEZE_INDEX_FILE"
mapfile -t FREEZE_INDICES < "$FREEZE_INDEX_FILE"
rm -f "$FREEZE_INDEX_FILE"
FREEZE_EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$B_ROOT,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_FROZEN_OUTPUT_DIR=$FROZEN_V2,EVSP_PYTHON=$PYTHON_BIN"
FREEZE_JOB=""
if [[ ${#FREEZE_INDICES[@]} -gt 0 ]]; then
  FREEZE_ARRAY=$(IFS=,; echo "${FREEZE_INDICES[*]}")
  FREEZE_JOB=$(evsp_submit_and_resolve eub24_fz2 \
    --array="${FREEZE_ARRAY}%45" -p default_partition -c 1 --mem=4G -t 00:15:00 \
    --no-requeue --export="$FREEZE_EXPORTS" \
    -o "$B_ROOT/logs/fz2_%A_%a.out" -e "$B_ROOT/logs/fz2_%A_%a.err" \
    "$SCRIPT_DIR/panel_b_freeze.sub")
fi

A_JOBS="$A_ROOT/integer_recovery_v2_jobs.tsv"
if [[ -n "$MIP_JOB$TF_JOB" && -e "$A_JOBS" ]]; then
  A_JOBS="$A_ROOT/integer_recovery_v2_${MIP_JOB:-none}_${TF_JOB:-none}_jobs.tsv"
fi
if [[ -n "$MIP_JOB$TF_JOB" || ! -e "$A_JOBS" ]]; then
  {
    echo -e "stage\tarray_job_id\ttasks\tsolver_commit"
    [[ -z "$MIP_JOB" ]] \
      || echo -e "mip_recovery\t$MIP_JOB\t${#MIP_INDICES[@]}\t$SOLVER_COMMIT"
    [[ -z "$TF_JOB" ]] \
      || echo -e "target_recovery\t$TF_JOB\t${#TARGET_INDICES[@]}\t$SOLVER_COMMIT"
  } | tee "$A_JOBS"
fi
B_JOBS="$B_ROOT/refreeze_v2_jobs.tsv"
if [[ -n "$FREEZE_JOB" && -e "$B_JOBS" ]]; then
  B_JOBS="$B_ROOT/refreeze_v2_${FREEZE_JOB}_jobs.tsv"
fi
if [[ -n "$FREEZE_JOB" || ! -e "$B_JOBS" ]]; then
  {
    echo -e "stage\tarray_job_id\ttasks\tsolver_commit"
    [[ -z "$FREEZE_JOB" ]] \
      || echo -e "freeze_v2\t$FREEZE_JOB\t${#FREEZE_INDICES[@]}\t$SOLVER_COMMIT"
  } | tee "$B_JOBS"
fi
sha256sum \
  "$A_ROOT/panel_a_integer_inputs_v2.tsv" \
  "$A_ROOT/integer_recovery_v2_provenance.json" \
  "$A_JOBS" \
  "$B_JOBS" \
  > "$B_ROOT/RECOVERY_V2_SHA256SUMS"

if [[ -n "$MIP_JOB" ]]; then
  echo "Panel A MIP recovery: $MIP_JOB (${#MIP_INDICES[@]} tasks)"
else
  echo "Panel A MIP recovery: skipped (54 valid artifacts already exist)"
fi
if [[ -n "$TF_JOB" ]]; then
  echo "Panel A target retry: $TF_JOB (${#TARGET_INDICES[@]} tasks)"
else
  echo "Panel A target retry: skipped (54 valid artifacts already exist)"
fi
if [[ -n "$FREEZE_JOB" ]]; then
  echo "Panel B v2 refreeze: $FREEZE_JOB (${#FREEZE_INDICES[@]} tasks)"
else
  echo "Panel B v2 refreeze: skipped (45 valid artifacts already exist)"
fi
