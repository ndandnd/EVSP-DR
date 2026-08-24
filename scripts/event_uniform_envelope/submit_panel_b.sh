#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -ge 1 && $# -le 2 ]] || evsp_die "usage: $0 PANEL_A_ROOT [PANEL_B_ROOT]"

A_ROOT=$(cd "$1" && pwd)
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "submission checkout must be on a named branch"
COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
B_ROOT="${2:-$HOME/ladder-lite/event_uniform_B_$(date -u +%Y%m%d)_${COMMIT:0:7}}"

if squeue --me -h -o '%j' | grep -qE '^eub24_(cg|frz)$'; then
  evsp_die "Panel B jobs already exist"
fi
[[ ! -e "$B_ROOT" ]] || evsp_die "Panel B root already exists: $B_ROOT"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_panel_b.py" \
  --panel-a "$A_ROOT" --panel-b "$B_ROOT" \
  --execution-repo "$EXECUTION_REPO" --commit "$COMMIT"

EXPORTS="ALL,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_CAMPAIGN_ROOT=$B_ROOT,EVSP_EXPECTED_COMMIT=$COMMIT,EVSP_PYTHON=$PYTHON_BIN"
CG_JOB=$(evsp_submit_and_resolve eub24_cg \
  --array=0-44%45 -p default_partition -c 2 --mem=24G -t 06:30:00 \
  --no-requeue --export="$EXPORTS" \
  -o "$B_ROOT/logs/cg_%A_%a.out" -e "$B_ROOT/logs/cg_%A_%a.err" \
  "$SCRIPT_DIR/panel_b_cg.sub")
sleep 1
FREEZE_JOB=$(evsp_submit_and_resolve eub24_frz \
  --array=0-44%45 --dependency="afterany:$CG_JOB" --kill-on-invalid-dep=yes \
  -p default_partition -c 1 --mem=4G -t 00:15:00 --no-requeue \
  --export="$EXPORTS" \
  -o "$B_ROOT/logs/frz_%A_%a.out" -e "$B_ROOT/logs/frz_%A_%a.err" \
  "$SCRIPT_DIR/panel_b_freeze.sub")

{
  echo -e "stage\tarray_job_id\ttasks"
  echo -e "cg\t$CG_JOB\t45"
  echo -e "freeze\t$FREEZE_JOB\t45"
} | tee "$B_ROOT/jobs.tsv"
sha256sum "$B_ROOT/execution_plan.json" "$B_ROOT/matrix.tsv" "$B_ROOT/jobs.tsv" \
  > "$B_ROOT/SUBMISSION_SHA256SUMS"
echo "Panel B root: $B_ROOT"
