#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# == 1 ]] || evsp_die "usage: $0 PANEL_B_ROOT"

ROOT=$(cd "$1" && pwd)
REPO=$(evsp_repo_root)
MANAGER_BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$MANAGER_BRANCH" ]] || evsp_die "manager checkout must be on a named branch"
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$MANAGER_BRANCH" | tail -1)

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
  || evsp_die "reviewed solver commit is not an ancestor of Agent E tip"
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$SOLVER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"

if squeue --me -h -o '%j' | grep -qE '^eub24_(mip2|tf2)$'; then
  evsp_die "Panel B integer jobs already exist"
fi

FROZEN_INDEX_FILE=$(mktemp)
"$PYTHON_BIN" "$SCRIPT_DIR/select_missing_frozen_v2_indices.py" \
  --root "$ROOT" --output-dir "$ROOT/frozen_v2" > "$FROZEN_INDEX_FILE"
if [[ -s "$FROZEN_INDEX_FILE" ]]; then
  cat "$FROZEN_INDEX_FILE" >&2
  evsp_die "Panel B frozen_v2 is incomplete; run recover_and_refreeze_v2.sh"
fi
rm -f "$FROZEN_INDEX_FILE"

MANIFEST="$ROOT/panel_b_integer_inputs_v2.tsv"
"$PYTHON_BIN" "$SCRIPT_DIR/prepare_integer_manifest.py" \
  --root "$ROOT" --panel B --source-dir frozen_v2 --out "$MANIFEST" \
  --provenance "$ROOT/integer_v2_provenance.json" \
  --wrapper-commit "$WRAPPER_COMMIT" --solver-commit "$SOLVER_COMMIT"
sha256sum "$MANIFEST" > "$ROOT/panel_b_integer_inputs_v2.sha256"
mkdir -p "$ROOT/mip" "$ROOT/target"

MIP_INDEX_FILE=$(mktemp)
"$PYTHON_BIN" "$SCRIPT_DIR/select_missing_integer_indices.py" \
  --manifest "$MANIFEST" --root "$ROOT" --panel B --stage mip \
  > "$MIP_INDEX_FILE"
mapfile -t MIP_INDICES < "$MIP_INDEX_FILE"
rm -f "$MIP_INDEX_FILE"
TARGET_INDEX_FILE=$(mktemp)
"$PYTHON_BIN" "$SCRIPT_DIR/select_missing_integer_indices.py" \
  --manifest "$MANIFEST" --root "$ROOT" --panel B --stage target \
  > "$TARGET_INDEX_FILE"
mapfile -t TARGET_INDICES < "$TARGET_INDEX_FILE"
rm -f "$TARGET_INDEX_FILE"

EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_EXPECTED_COMMIT=$SOLVER_COMMIT,EVSP_PANEL=B,EVSP_INTEGER_MANIFEST=$MANIFEST,EVSP_MIP_OUTPUT_DIR=$ROOT/mip,EVSP_TARGET_OUTPUT_DIR=$ROOT/target,EVSP_PYTHON=$PYTHON_BIN"
MIP_JOB=""
if [[ ${#MIP_INDICES[@]} -gt 0 ]]; then
  MIP_ARRAY=$(IFS=,; echo "${MIP_INDICES[*]}")
  MIP_JOB=$(evsp_submit_and_resolve eub24_mip2 \
    --array="${MIP_ARRAY}%45" -p default_partition -c 8 --mem=24G -t 01:15:00 \
    --no-requeue --export="$EXPORTS" \
    -o "$ROOT/logs/mip2_%A_%a.out" -e "$ROOT/logs/mip2_%A_%a.err" \
    "$SCRIPT_DIR/pool_mip.sub")
  sleep 1
fi
TF_JOB=""
if [[ ${#TARGET_INDICES[@]} -gt 0 ]]; then
  TARGET_ARRAY=$(IFS=,; echo "${TARGET_INDICES[*]}")
  TF_JOB=$(evsp_submit_and_resolve eub24_tf2 \
    --array="${TARGET_ARRAY}%45" -p default_partition -c 8 --mem=24G -t 01:15:00 \
    --no-requeue --export="$EXPORTS" \
    -o "$ROOT/logs/tf2_%A_%a.out" -e "$ROOT/logs/tf2_%A_%a.err" \
    "$SCRIPT_DIR/target_feasibility.sub")
fi

JOBS="$ROOT/integer_v2_jobs.tsv"
if [[ -n "$MIP_JOB$TF_JOB" && -e "$JOBS" ]]; then
  JOBS="$ROOT/integer_v2_${MIP_JOB:-none}_${TF_JOB:-none}_jobs.tsv"
fi
if [[ -n "$MIP_JOB$TF_JOB" || ! -e "$JOBS" ]]; then
  {
    echo -e "stage\tarray_job_id\ttasks\tsolver_commit\tbackend\ttimelimit_s\tthreads"
    [[ -z "$MIP_JOB" ]] \
      || echo -e "mip_v2\t$MIP_JOB\t${#MIP_INDICES[@]}\t$SOLVER_COMMIT\tgurobi_two_stage\t1800\t8"
    [[ -z "$TF_JOB" ]] \
      || echo -e "target_v2\t$TF_JOB\t${#TARGET_INDICES[@]}\t$SOLVER_COMMIT\tgurobi\t1800\t8"
  } | tee "$JOBS"
fi
sha256sum \
  "$ROOT/panel_b_integer_inputs_v2.tsv" \
  "$ROOT/integer_v2_provenance.json" \
  "$JOBS" \
  > "$ROOT/INTEGER_V2_SUBMISSION_SHA256SUMS"

if [[ -n "$MIP_JOB" ]]; then
  echo "Panel B v2 MIP: $MIP_JOB (${#MIP_INDICES[@]} tasks)"
else
  echo "Panel B v2 MIP: skipped (45 valid artifacts already exist)"
fi
if [[ -n "$TF_JOB" ]]; then
  echo "Panel B v2 target: $TF_JOB (${#TARGET_INDICES[@]} tasks)"
else
  echo "Panel B v2 target: skipped (45 valid artifacts already exist)"
fi
