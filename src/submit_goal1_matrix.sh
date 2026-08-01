#!/usr/bin/env bash
# Submit the reproducible flat-price Goal-1 benchmark matrix on Unicorn.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/submit_goal1_matrix.sh smoke [RUN_TAG]
  src/submit_goal1_matrix.sh 3h [RUN_TAG]
  src/submit_goal1_matrix.sh 6h [RUN_TAG]
  src/submit_goal1_matrix.sh unlimited SLURM_WALLTIME [RUN_TAG]

Defaults for 3h/6h/unlimited:
  instances: Practice_10bus.csv,Practice_15bus.csv
  modes:     NO_CHEAT,GREEDY

Override with EVSP_INSTANCES, EVSP_MODES, EVSP_PRICE_CSV, EVSP_PARTITION,
EVSP_MEMORY, or the pricing variables documented in UNICORN_RUNBOOK.md.
EOF
}

if (( $# < 1 )); then
    usage
    exit 2
fi

PROFILE=$1
shift

case "$PROFILE" in
    smoke)
        ACTIVE_HOURS=0.05
        MILESTONES=0.05
        WALLTIME=00:30:00
        RUN_TAG=${1:-}
        INSTANCES_RAW=${EVSP_INSTANCES:-Practice_10bus.csv}
        MODES_RAW=${EVSP_MODES:-NO_CHEAT}
        export EVSP_MAX_LABELS=${EVSP_MAX_LABELS:-5000}
        export EVSP_PRICING_TIERS=${EVSP_PRICING_TIERS:-5000:30}
        export EVSP_PRICING_WALL_PER_ITER=${EVSP_PRICING_WALL_PER_ITER:-60}
        ;;
    3h)
        ACTIVE_HOURS=3
        MILESTONES=3
        WALLTIME=05:00:00
        RUN_TAG=${1:-}
        INSTANCES_RAW=${EVSP_INSTANCES:-Practice_10bus.csv,Practice_15bus.csv}
        MODES_RAW=${EVSP_MODES:-NO_CHEAT,GREEDY}
        ;;
    6h)
        ACTIVE_HOURS=6
        MILESTONES=3,6
        WALLTIME=08:00:00
        RUN_TAG=${1:-}
        INSTANCES_RAW=${EVSP_INSTANCES:-Practice_10bus.csv,Practice_15bus.csv}
        MODES_RAW=${EVSP_MODES:-NO_CHEAT,GREEDY}
        ;;
    unlimited)
        if (( $# < 1 )); then
            echo "unlimited requires an explicit Slurm walltime, e.g. 2-00:00:00" >&2
            exit 2
        fi
        ACTIVE_HOURS=0
        MILESTONES=${EVSP_MILESTONES:-3,6,12,24}
        WALLTIME=$1
        RUN_TAG=${2:-}
        INSTANCES_RAW=${EVSP_INSTANCES:-Practice_10bus.csv,Practice_15bus.csv}
        MODES_RAW=${EVSP_MODES:-NO_CHEAT,GREEDY}
        ;;
    *)
        usage
        exit 2
        ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
mkdir -p src/logs src/results

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Checkout is dirty; refusing an unreproducible benchmark submission." >&2
    git status --short >&2
    exit 2
fi

if ! command -v sbatch >/dev/null 2>&1; then
    if [[ -x /usr/local/slurm/current/bin/sbatch ]]; then
        export PATH=/usr/local/slurm/current/bin:$PATH
    else
        echo "sbatch is unavailable. Run this launcher on Unicorn." >&2
        exit 127
    fi
fi

SHORT_SHA=$(git rev-parse --short HEAD)
if [[ -z "$RUN_TAG" ]]; then
    RUN_TAG="goal1_${PROFILE}_${SHORT_SHA}_$(date +%Y%m%dT%H%M%S)"
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "RUN_TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
fi
PRICE_CSV=${EVSP_PRICE_CSV:-hourly_prices_flat.csv}
MEMORY=${EVSP_MEMORY:-32G}

IFS=',' read -r -a INSTANCES <<< "$INSTANCES_RAW"
IFS=',' read -r -a MODES <<< "$MODES_RAW"

sbatch_args=(--time="$WALLTIME" --mem="$MEMORY" --cpus-per-task=4)
if [[ -n "${EVSP_PARTITION:-}" ]]; then
    sbatch_args+=(--partition="$EVSP_PARTITION")
fi

echo "Submitting Goal-1 profile=$PROFILE tag=$RUN_TAG commit=$SHORT_SHA"
echo "  instances : ${INSTANCES[*]}"
echo "  modes     : ${MODES[*]}"
echo "  active    : ${ACTIVE_HOURS}h; milestones=$MILESTONES; wall=$WALLTIME"
echo "  prices    : $PRICE_CSV"

for instance in "${INSTANCES[@]}"; do
    for mode in "${MODES[@]}"; do
        job_name="G1_${PROFILE}_${instance%.csv}_${mode}"
        job_id=$(sbatch --parsable "${sbatch_args[@]}" --job-name="$job_name" \
            src/submit_goal1_colgen.sub \
            "$instance" "$mode" "$ACTIVE_HOURS" "$RUN_TAG" "$MILESTONES" "$PRICE_CSV")
        echo "  submitted $job_id  $instance  $mode"
    done
done

echo "Run tag: $RUN_TAG"
echo "Monitor: squeue --me"
echo "Accounting: sacct -S today --name='G1_*' --format=JobID,JobName,State,Elapsed,Timelimit,MaxRSS,ExitCode"
