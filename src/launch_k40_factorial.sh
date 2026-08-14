#!/bin/bash
# Submit the four controlled k40 diagnostic arms from a reviewed detached
# checkout. This script is safe to run from an SSH shell: its strict mode is
# confined to the child bash process and cannot log the caller out.

set -euo pipefail

fatal() {
  echo "[K40-LAUNCH] FATAL: $*" >&2
  exit 2
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

command -v sbatch >/dev/null || fatal "sbatch is unavailable"
if git symbolic-ref -q HEAD >/dev/null; then
  fatal "use a detached reviewed worktree, not a branch checkout"
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  fatal "tracked checkout is dirty"
fi

EXPECTED_COMMIT=$(git rev-parse HEAD)
SHORT_COMMIT=${EXPECTED_COMMIT:0:8}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CAMPAIGN=${K40_FACTORIAL_CAMPAIGN:-k40fx_${STAMP}_${SHORT_COMMIT}}
RESULT_DIR="$ROOT/src/results/k40_factorial/$CAMPAIGN"
LOG_DIR="$ROOT/src/cluster_logs/k40_factorial/$CAMPAIGN"
MANIFEST="$RESULT_DIR/launch.tsv"

[ ! -e "$MANIFEST" ] || fatal "campaign already exists: $CAMPAIGN"
mkdir -p "$RESULT_DIR" "$LOG_DIR"
printf 'role\tjob_id\tjob_name\tmaster_sense\tinitial_pool\n' > "$MANIFEST"

COMMON_EXPORT="HOME=$HOME,PATH=$PATH,EVSP_DR_ROOT=$ROOT,EVSP_EXPECTED_COMMIT=$EXPECTED_COMMIT,K40_FACTORIAL_CAMPAIGN=$CAMPAIGN"
PREP_JOB=$(sbatch --parsable \
  --job-name=K40-PREP \
  --output="$LOG_DIR/K40-PREP_%j.out" \
  --error="$LOG_DIR/K40-PREP_%j.err" \
  --export="$COMMON_EXPORT" \
  "$ROOT/src/submit_k40_factorial_prep.sub")
PREP_JOB=${PREP_JOB%%;*}
printf 'prep\t%s\tK40-PREP\t-\t-\n' "$PREP_JOB" >> "$MANIFEST"

JOB_IDS=()
submit_arm() {
  local job_name=$1
  local master_sense=$2
  local initial_pool=$3
  local job_id
  job_id=$(sbatch --parsable \
    --dependency="afterok:$PREP_JOB" \
    --job-name="$job_name" \
    --output="$LOG_DIR/${job_name}_%j.out" \
    --error="$LOG_DIR/${job_name}_%j.err" \
    --export="$COMMON_EXPORT,K40_MASTER_SENSE=$master_sense,K40_INITIAL_POOL=$initial_pool" \
    "$ROOT/src/submit_k40_factorial.sub")
  job_id=${job_id%%;*}
  JOB_IDS+=("$job_id")
  printf 'arm\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$job_name" "$master_sense" "$initial_pool" >> "$MANIFEST"
}

submit_arm K40-CA24 cover artificial
submit_arm K40-CS24 cover singletons
submit_arm K40-PA24 partition artificial
submit_arm K40-PS24 partition singletons

ALL_IDS=$PREP_JOB
for job_id in "${JOB_IDS[@]}"; do
  ALL_IDS="$ALL_IDS,$job_id"
done

echo "[K40-LAUNCH] campaign=$CAMPAIGN"
echo "[K40-LAUNCH] commit=$EXPECTED_COMMIT"
echo "[K40-LAUNCH] prep=$PREP_JOB arms=${JOB_IDS[*]}"
echo "[K40-LAUNCH] manifest=$MANIFEST"
echo "[K40-LAUNCH] monitor: bash src/monitor_k40_factorial.sh"
squeue -j "$ALL_IDS" -o '%.14i %.15j %.2t %.10M %R' || true
