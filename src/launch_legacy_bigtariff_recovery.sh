#!/bin/bash
# Lightweight Unicorn launcher for the three preemption-damaged big-tariff
# tasks from array 867334.  Default is a dry run; add --submit to queue jobs.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_ROOT=$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel)
ROOT="${EVSP_DR_ROOT:-$SCRIPT_ROOT}"
SOURCE_ROOT="$HOME/EVSP-DR"
ARRAY_JOB=867334
TASKS="22,24,32"
LEGACY_REF=f4e31c3
SUBMIT=0

usage() {
  cat <<'EOF'
Usage:
  bash src/launch_legacy_bigtariff_recovery.sh [options]

Options:
  --continuation-root PATH
                       Current checkout containing this launcher (default:
                       checkout from which this script is invoked)
  --source-root PATH   Legacy pinned checkout (default: $HOME/EVSP-DR)
  --array-job ID       Original Slurm array id (default: 867334)
  --tasks CSV          Subset of 22,24,32 (default: 22,24,32)
  --legacy-ref REF     Legacy generation commit (default: f4e31c3)
  --submit             Actually submit; without this, print the plan only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --continuation-root) ROOT=$2; shift 2 ;;
    --source-root) SOURCE_ROOT=$2; shift 2 ;;
    --array-job) ARRAY_JOB=$2; shift 2 ;;
    --tasks) TASKS=$2; shift 2 ;;
    --legacy-ref) LEGACY_REF=$2; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -z "${SLURM_JOB_ID:-}" ] \
  || { echo "run this lightweight wrapper on a Unicorn login node" >&2; exit 2; }
command -v sbatch >/dev/null 2>&1 \
  || { echo "sbatch is unavailable; run this on Unicorn" >&2; exit 2; }
[ -d "$ROOT/src" ] || { echo "continuation root has no src/: $ROOT" >&2; exit 2; }
[ -d "$SOURCE_ROOT/src" ] \
  || { echo "legacy source root has no src/: $SOURCE_ROOT" >&2; exit 2; }
git -C "$ROOT" diff --quiet \
  || { echo "continuation checkout has tracked changes" >&2; exit 2; }
git -C "$ROOT" diff --cached --quiet \
  || { echo "continuation checkout has staged changes" >&2; exit 2; }

COMMIT=$(git -C "$ROOT" rev-parse HEAD)
LEGACY_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse "$LEGACY_REF^{commit}")
SOURCE_HEAD=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
[ "$SOURCE_HEAD" = "$LEGACY_COMMIT" ] || {
  echo "legacy source checkout must remain pinned at $LEGACY_COMMIT; found $SOURCE_HEAD" >&2
  exit 2
}
LOGDIR="$ROOT/src/cluster_logs/legacy_recovery/job${ARRAY_JOB}"
mkdir -p "$LOGDIR"

IFS=',' read -r -a TASK_LIST <<< "$TASKS"
submitted=0
for TASK in "${TASK_LIST[@]}"; do
  case "$TASK" in
    22) SHORT="30r2-p18";
        CSV_REL="duty_unions_big/Practice_Custom_DutyUnion_k30_r2.csv";
        PRICE_REL="hourly_prices_single_peak_18.csv";
        SOURCE_CELL="Practice_Custom_DutyUnion_k30_r2_peak18" ;;
    24) SHORT="30r4-p18";
        CSV_REL="duty_unions_big/Practice_Custom_DutyUnion_k30_r4.csv";
        PRICE_REL="hourly_prices_single_peak_18.csv";
        SOURCE_CELL="Practice_Custom_DutyUnion_k30_r4_peak18" ;;
    32) SHORT="30r2-sek";
        CSV_REL="duty_unions_big/Practice_Custom_DutyUnion_k30_r2.csv";
        PRICE_REL="hourly_prices_transdev_sek.csv";
        SOURCE_CELL="Practice_Custom_DutyUnion_k30_r2_sek" ;;
    *) echo "unsupported task: $TASK (allowed: 22,24,32)" >&2; exit 2 ;;
  esac
  missing=0
  for relative in "$CSV_REL" "$PRICE_REL"; do
    if [ ! -f "$ROOT/data/$relative" ]; then
      echo "missing continuation input: $ROOT/data/$relative" >&2
      echo "copy it without overwriting existing data:" >&2
      echo "  rsync -a --relative '$SOURCE_ROOT/data/./$relative' '$ROOT/data/'" >&2
      missing=1
    fi
  done
  for suffix in "" ".columns.jsonl" ".iters.csv"; do
    source_artifact="$SOURCE_ROOT/src/results/tariff_big/${SOURCE_CELL}.json${suffix}"
    if [ ! -f "$source_artifact" ]; then
      echo "missing legacy artifact: $source_artifact" >&2
      missing=1
    fi
  done
  [ "$missing" -eq 0 ] || exit 2
  JOB="R${TASK}-${SHORT}-c${COMMIT:0:6}"
  active=$(squeue -h -u "$USER" -n "$JOB" -o '%A' 2>/dev/null || true)
  if [ -n "$active" ]; then
    echo "[RECOVERY] SKIP_ACTIVE $JOB job=${active%%$'\n'*}"
    continue
  fi
  echo "[RECOVERY] task=$TASK job=$JOB source=$SOURCE_ROOT@$LEGACY_COMMIT continuation=$ROOT@$COMMIT"
  if [ "$SUBMIT" -eq 1 ]; then
    job_id=$(sbatch --parsable \
      --job-name "$JOB" \
      --output "$LOGDIR/${JOB}_%j.out" \
      --error "$LOGDIR/${JOB}_%j.err" \
      --export="ALL,EVSP_DR_ROOT=$ROOT,EVSP_LEGACY_ROOT=$SOURCE_ROOT,EVSP_EXPECTED_COMMIT=$COMMIT" \
      "$ROOT/src/submit_legacy_bigtariff_recovery.sub" \
      "$TASK" "$ARRAY_JOB" "$LEGACY_COMMIT" "$COMMIT")
    job_id=${job_id%%;*}
    echo "[RECOVERY] SUBMITTED $JOB job=$job_id"
    submitted=$((submitted + 1))
  fi
done

if [ "$SUBMIT" -eq 0 ]; then
  echo "[RECOVERY] dry run only; add --submit after checking the three lines above"
else
  echo "[RECOVERY] submitted=$submitted"
  squeue --me -o '%.14i %.32j %.2t %.10M %R' | sed -n '1,20p'
fi
