#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [SMALL_THRESHOLD_ROOT]"
SOURCE_ROOT="${1:-$HOME/ladder-lite/small_threshold_event_20260903_44b6d5}"
SOURCE_ROOT=$(cd "$SOURCE_ROOT" && pwd)
FAILED_ROOT="$SOURCE_ROOT/k5_raw_mip36h_20260905_v3"
RECOVERY_ROOT="$SOURCE_ROOT/k5_raw_mip36h_20260905_v4"

REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "manager checkout must be on a named branch"
WRAPPER_COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$WRAPPER_COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
[[ -x "$PYTHON_BIN" ]] || evsp_die "missing Python interpreter: $PYTHON_BIN"
[[ -r /share/apps/software/gurobi/gurobi.lic ]] \
  || evsp_die "shared Gurobi license is unreadable"
if squeue --me -h -o '%j' | grep -q '^MPBk5R36R1$'; then
  evsp_die "k5 path-policy recovery is already active"
fi

if [[ ! -e "$RECOVERY_ROOT" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_k5_raw_mip36h_recovery.py" \
    --failed-root "$FAILED_ROOT" --output-root "$RECOVERY_ROOT"
fi
MANIFEST="$RECOVERY_ROOT/snapshot_manifest.tsv"
[[ -s "$MANIFEST" ]] || evsp_die "missing recovery manifest"
if compgen -G "$RECOVERY_ROOT/jobs_*.tsv" >/dev/null; then
  existing=$(awk -F'\t' 'FNR > 1 {print $2}' "$RECOVERY_ROOT"/jobs_*.tsv | sort -u)
  echo "k5 path-policy recovery already recorded: $existing"
  exit 0
fi

WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/k5_raw_mip36h.sub"
RUNNER="$EXECUTION_REPO/src/run_exact_pool_mip.py"
WORKER_SHA=$(sha256sum "$WORKER" | awk '{print $1}')
RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
EXPORTS="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$WRAPPER_COMMIT,EVSP_MIP_ROOT=$RECOVERY_ROOT,EVSP_MIP_MANIFEST=$MANIFEST,EVSP_MIP_EXPECTED_WORKER_SHA256=$WORKER_SHA,EVSP_MIP_EXPECTED_RUNNER_SHA256=$RUNNER_SHA,EVSP_PYTHON=$PYTHON_BIN"

JOB=$(evsp_submit_and_resolve MPBk5R36R1 \
  --array=0-3%4 -p scaglione -c 8 --mem=32G -t 02:15:00 \
  --no-requeue --open-mode=append --export="$EXPORTS" \
  -o "$RECOVERY_ROOT/logs/%x_%A_%a.out" \
  -e "$RECOVERY_ROOT/logs/%x_%A_%a.err" "$WORKER")

JOBS="$RECOVERY_ROOT/jobs_${JOB}.tsv"
{
  printf 'stage\tarray_job_id\ttasks\tindices\tpartition\trequeue\tthreads\tmem\tslurm_timelimit\tmip_timelimit_s\tmip_gap\ttwo_stage\tpool_treatment\tsnapshot_budget_s\twrapper_commit\trunner_sha256\tworker_sha256\n'
  printf 'k5_raw_mip36h_path_recovery\t%s\t4\t0,1,2,3\tscaglione\tfalse\t8\t32G\t02:15:00\t1800\t0.0001\ttrue\tRAW\t129600\t%s\t%s\t%s\n' \
    "$JOB" "$WRAPPER_COMMIT" "$RUNNER_SHA" "$WORKER_SHA"
} > "$JOBS"
sha256sum "$MANIFEST" "$RECOVERY_ROOT/snapshot_manifest.csv" \
  "$RECOVERY_ROOT/recovery_provenance.json" "$JOBS" \
  > "$RECOVERY_ROOT/SUBMISSION_INPUT_SHA256SUMS"

echo "k5 RAW 36h final-replay recovery: $JOB (4 tasks)"
echo "Reused the four hash-identical v3 snapshots; no CG freezing repeated"
echo "After completion: bash scripts/event_uniform_envelope/audit_k5_raw_mip36h.sh '$SOURCE_ROOT' k5_raw_mip36h_20260905_v4"
