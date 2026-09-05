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

MIP_ROOT="$SOURCE_ROOT/k5_raw_mip36h_20260905"
if [[ ! -e "$MIP_ROOT" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_k5_raw_mip36h.py" \
    --resume-root "$RESUME_ROOT" --output-root "$MIP_ROOT" \
    --freezer "$EXECUTION_REPO/src/freeze_exact_cg_at_wall.py" \
    --python "$PYTHON_BIN"
fi
MANIFEST="$MIP_ROOT/snapshot_manifest.tsv"
[[ -s "$MANIFEST" ]] || evsp_die "missing snapshot manifest: $MANIFEST"
[[ "$(awk 'END {print NR}' "$MANIFEST")" == "5" ]] \
  || evsp_die "snapshot manifest must contain four data rows"
(cd "$MIP_ROOT" && sha256sum -c snapshot_manifest.sha256) \
  || evsp_die "snapshot manifest checksum validation failed"
while IFS=$'\t' read -r LOCAL_INDEX SOURCE_INDEX CELL TARGET REP SOURCE \
  SNAPSHOT SNAPSHOT_SHA JOURNAL JOURNAL_SHA REST; do
  [[ "$(sha256sum "$SNAPSHOT" | awk '{print $1}')" == "$SNAPSHOT_SHA" ]] \
    || evsp_die "snapshot hash mismatch before submission: $CELL"
  [[ "$(sha256sum "$JOURNAL" | awk '{print $1}')" == "$JOURNAL_SHA" ]] \
    || evsp_die "journal hash mismatch before submission: $CELL"
done < <(tail -n +2 "$MANIFEST")

if compgen -G "$MIP_ROOT/jobs_*.tsv" >/dev/null; then
  existing=$(awk -F'\t' 'FNR > 1 {print $2}' "$MIP_ROOT"/jobs_*.tsv | sort -u)
  echo "k=5 RAW 36h MIP submission already recorded: $existing"
  exit 0
fi

WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/k5_raw_mip36h.sub"
RUNNER="$EXECUTION_REPO/src/run_exact_pool_mip.py"
WORKER_SHA=$(sha256sum "$WORKER" | awk '{print $1}')
RUNNER_SHA=$(sha256sum "$RUNNER" | awk '{print $1}')
EXPORTS="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$WRAPPER_COMMIT,EVSP_MIP_ROOT=$MIP_ROOT,EVSP_MIP_MANIFEST=$MANIFEST,EVSP_MIP_EXPECTED_WORKER_SHA256=$WORKER_SHA,EVSP_MIP_EXPECTED_RUNNER_SHA256=$RUNNER_SHA,EVSP_PYTHON=$PYTHON_BIN"

JOB=$(evsp_submit_and_resolve MPBk5R36T30 \
  --array=0-3%4 -p scaglione -c 8 --mem=32G -t 02:15:00 \
  --no-requeue --open-mode=append --export="$EXPORTS" \
  -o "$MIP_ROOT/logs/%x_%A_%a.out" \
  -e "$MIP_ROOT/logs/%x_%A_%a.err" "$WORKER")

JOBS="$MIP_ROOT/jobs_${JOB}.tsv"
{
  printf 'stage\tarray_job_id\ttasks\tindices\tpartition\trequeue\tthreads\tmem\tslurm_timelimit\tmip_timelimit_s\tmip_gap\ttwo_stage\tpool_treatment\tsnapshot_budget_s\twrapper_commit\trunner_sha256\tworker_sha256\n'
  printf 'k5_raw_mip36h\t%s\t4\t0,1,2,3\tscaglione\tfalse\t8\t32G\t02:15:00\t1800\t0.0001\ttrue\tRAW\t129600\t%s\t%s\t%s\n' \
    "$JOB" "$WRAPPER_COMMIT" "$RUNNER_SHA" "$WORKER_SHA"
} > "$JOBS"
sha256sum "$MANIFEST" "$MIP_ROOT/snapshot_manifest.csv" "$JOBS" \
  > "$MIP_ROOT/SUBMISSION_INPUT_SHA256SUMS"

echo "k=5 RAW 36h two-stage Gurobi MIP: $JOB (4 tasks)"
echo "Scientific solver limit: 1800 seconds per task"
echo "CSV: $MIP_ROOT/snapshot_manifest.csv"
echo "After completion: bash scripts/event_uniform_envelope/audit_k5_raw_mip36h.sh '$SOURCE_ROOT'"
