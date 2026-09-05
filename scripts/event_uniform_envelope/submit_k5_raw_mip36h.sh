#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [SMALL_THRESHOLD_ROOT]"
SOURCE_ROOT="${1:-$HOME/ladder-lite/small_threshold_event_20260903_44b6d5}"
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd)
RESUME_ROOT="$SOURCE_ROOT/cg_resume48h_20260904"
[[ -d "$RESUME_ROOT" ]] || evsp_die "missing cumulative-48h root: $RESUME_ROOT"

REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "manager checkout must be on a named branch"
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$WRAPPER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
[[ -x "$PYTHON_BIN" ]] || evsp_die "missing Python interpreter: $PYTHON_BIN"
[[ -r /share/apps/software/gurobi/gurobi.lic ]] \
  || evsp_die "shared Gurobi license is unreadable"

MIP_ROOT="$SOURCE_ROOT/k5_raw_mip36h_20260905_v3"
mkdir -p "$MIP_ROOT/logs"
MANIFEST="$MIP_ROOT/snapshot_manifest.tsv"
if compgen -G "$MIP_ROOT/jobs_*.tsv" >/dev/null; then
  existing=$(awk -F'\t' 'FNR > 1 {print $2}' "$MIP_ROOT"/jobs_*.tsv | sort -u)
  echo "k=5 RAW 36h MIP submission already recorded: $existing"
  exit 0
fi
if squeue --me -h -o '%j' | grep -qE '^(k5f36|MPBk5R36T30)$'; then
  evsp_die "k=5 36h freeze/MIP pipeline is already active"
fi

FREEZE_WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/k5_raw_freeze36h.sub"
FREEZER="$EXECUTION_REPO/src/freeze_exact_cg_prefix.py"
WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/k5_raw_mip36h.sub"
RUNNER="$EXECUTION_REPO/src/run_exact_pool_mip.py"
FREEZE_WORKER_SHA=$(sha256sum "$FREEZE_WORKER" | awk '{print $1}')
FREEZER_SHA=$(sha256sum "$FREEZER" | awk '{print $1}')
WORKER_SHA=$(sha256sum "$WORKER" | awk '{print $1}')
RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
EXPORTS="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$WRAPPER_COMMIT,EVSP_MIP_ROOT=$MIP_ROOT,EVSP_MIP_MANIFEST=$MANIFEST,EVSP_MIP_EXPECTED_WORKER_SHA256=$WORKER_SHA,EVSP_MIP_EXPECTED_RUNNER_SHA256=$RUNNER_SHA,EVSP_PYTHON=$PYTHON_BIN"
FREEZE_EXPORTS="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$WRAPPER_COMMIT,EVSP_RESUME_ROOT=$RESUME_ROOT,EVSP_MIP_ROOT=$MIP_ROOT,EVSP_FREEZE_EXPECTED_WORKER_SHA256=$FREEZE_WORKER_SHA,EVSP_FREEZE_EXPECTED_RUNNER_SHA256=$FREEZER_SHA,EVSP_PYTHON=$PYTHON_BIN"

FREEZE_JOB=$(evsp_submit_and_resolve k5f36 \
  -p scaglione -c 1 --mem=32G -t 02:15:00 --no-requeue \
  --open-mode=append --export="$FREEZE_EXPORTS" \
  -o "$MIP_ROOT/logs/%x_%j.out" -e "$MIP_ROOT/logs/%x_%j.err" \
  "$FREEZE_WORKER")
JOB=$(evsp_submit_and_resolve MPBk5R36T30 \
  --array=0-3%4 -p scaglione -c 8 --mem=32G -t 02:15:00 \
  --no-requeue --open-mode=append \
  --dependency="afterok:$FREEZE_JOB" --kill-on-invalid-dep=yes \
  --export="$EXPORTS" \
  -o "$MIP_ROOT/logs/%x_%A_%a.out" \
  -e "$MIP_ROOT/logs/%x_%A_%a.err" "$WORKER")

JOBS="$MIP_ROOT/jobs_${JOB}.tsv"
{
  printf 'stage\tarray_job_id\ttasks\tindices\tpartition\trequeue\tthreads\tmem\tslurm_timelimit\tmip_timelimit_s\tmip_gap\ttwo_stage\tpool_treatment\tsnapshot_budget_s\twrapper_commit\trunner_sha256\tworker_sha256\n'
  printf 'k5_raw_mip36h\t%s\t4\t0,1,2,3\tscaglione\tfalse\t8\t32G\t02:15:00\t1800\t0.0001\ttrue\tRAW\t129600\t%s\t%s\t%s\n' \
    "$JOB" "$WRAPPER_COMMIT" "$RUNNER_SHA" "$WORKER_SHA"
} > "$JOBS"
FREEZE_JOBS="$MIP_ROOT/freeze_job_${FREEZE_JOB}.tsv"
{
  printf 'stage\tjob_id\tpartition\trequeue\tcpus\tmem\ttimelimit\twrapper_commit\tfreezer_sha256\tworker_sha256\n'
  printf 'k5_raw_freeze36h\t%s\tscaglione\tfalse\t1\t32G\t02:15:00\t%s\t%s\t%s\n' \
    "$FREEZE_JOB" "$WRAPPER_COMMIT" "$FREEZER_SHA" "$FREEZE_WORKER_SHA"
} > "$FREEZE_JOBS"
sha256sum "$RESUME_ROOT/execution_plan.json" "$RESUME_ROOT/matrix.tsv" \
  "$JOBS" "$FREEZE_JOBS" > "$MIP_ROOT/SUBMISSION_INPUT_SHA256SUMS"

echo "k=5 RAW 36h snapshot stage: $FREEZE_JOB"
echo "k=5 RAW 36h two-stage Gurobi MIP: $JOB (4 dependent tasks)"
echo "Scientific solver limit: 1800 seconds per task"
echo "CSV after snapshot stage: $MIP_ROOT/snapshot_manifest.csv"
echo "After completion: bash scripts/event_uniform_envelope/audit_k5_raw_mip36h.sh '$SOURCE_ROOT'"
