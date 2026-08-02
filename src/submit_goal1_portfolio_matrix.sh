#!/usr/bin/env bash
# Submit complementary Goal-1 pricing policies as separate reproducible jobs.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/submit_goal1_portfolio_matrix.sh smoke [BASE_RUN_TAG]
  src/submit_goal1_portfolio_matrix.sh 5m [BASE_RUN_TAG]
  src/submit_goal1_portfolio_matrix.sh 30m [BASE_RUN_TAG]
  src/submit_goal1_portfolio_matrix.sh 3h [BASE_RUN_TAG]

The default instance set is all five deterministic synthetic 10-bus samples
and all five deterministic synthetic 15-bus samples under
data/random_goal1_instances/seed_20260802/. Generate them first with:

  python -u src/generate_random_goal1_instances.py

Override the comma-separated list with EVSP_INSTANCES. Every job uses GREEDY
so this remains a pricing-discovery test. Three components are submitted:

  bound_resource   reduced_cost_bound + diversified + resource
  fair_resource    start_fair_bound + diversified + resource
  fair_incidence   start_fair_bound + diversified + incidence_diverse

The components intentionally use distinct run tags and result pools. Once all
jobs finish, merge their final-pool JSON files with
src/audit_goal1_column_pools.py. Separate-job completion does not itself mean
that the production column-generation loop is a portfolio.
EOF
}

if (( $# < 1 || $# > 2 )); then
    usage
    exit 2
fi

PROFILE=$1
BASE_RUN_TAG=${2:-}
case "$PROFILE" in
    smoke|5m|30m|3h) ;;
    *)
        usage
        exit 2
        ;;
esac

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Checkout is dirty; refusing an unreproducible portfolio submission." >&2
    git status --short >&2
    exit 2
fi

DEFAULT_INSTANCES="random_goal1_instances/seed_20260802/Practice_SyntheticRandom_10bus_s20260802_r01.csv"
for size in 10 15; do
    for replicate in 01 02 03 04 05; do
        candidate="random_goal1_instances/seed_20260802/Practice_SyntheticRandom_${size}bus_s20260802_r${replicate}.csv"
        if [[ "$candidate" != "$DEFAULT_INSTANCES" ]]; then
            DEFAULT_INSTANCES+=",$candidate"
        fi
    done
done
INSTANCES_RAW=${EVSP_INSTANCES:-$DEFAULT_INSTANCES}

IFS=',' read -r -a INSTANCES <<< "$INSTANCES_RAW"
for instance in "${INSTANCES[@]}"; do
    if [[ ! -f "data/$instance" ]]; then
        echo "Missing data/$instance" >&2
        echo "Run: python -u src/generate_random_goal1_instances.py" >&2
        exit 2
    fi
done

SHORT_SHA=$(git rev-parse --short HEAD)
if [[ -z "$BASE_RUN_TAG" ]]; then
    BASE_RUN_TAG="goal1_portfolio_${PROFILE}_${SHORT_SHA}_$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ ! "$BASE_RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "BASE_RUN_TAG may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 2
fi

MATRIX=src/submit_goal1_matrix.sh

submit_component() {
    local component=$1
    local queue_order=$2
    local dominance_mode=$3
    local component_tag="${BASE_RUN_TAG}_${component}"

    echo
    echo "Submitting portfolio component: $component"
    EVSP_INSTANCES="$INSTANCES_RAW" \
    EVSP_MODES=GREEDY \
    EVSP_QUEUE_ORDER="$queue_order" \
    EVSP_PRICING_OUTPUT_SELECTION=diversified \
    EVSP_DOMINANCE_MODE="$dominance_mode" \
        bash "$MATRIX" "$PROFILE" "$component_tag"
}

echo "Synthetic/unverified-day Goal-1 portfolio"
echo "  profile  : $PROFILE"
echo "  base tag : $BASE_RUN_TAG"
echo "  instances: ${#INSTANCES[@]}"

submit_component bound_resource reduced_cost_bound resource
submit_component fair_resource start_fair_bound resource
submit_component fair_incidence start_fair_bound incidence_diverse

echo
echo "Submitted all three components. Base tag: $BASE_RUN_TAG"
echo "Monitor: squeue --me"
echo "After completion, list pools with:"
echo "  find src/results -type f -name 'routes_colgen_final_*.json' -path '*${BASE_RUN_TAG}*' -print"
echo "Then pass those paths to: python -u src/audit_goal1_column_pools.py"
