#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE="$REPO_ROOT/environment-py312.yml"
ENV_PREFIX=${EVSP_PY312_ENV_PREFIX:-"$REPO_ROOT/../.evspdr-envs/py312"}
MPL_CACHE=${EVSP_MPLCONFIGDIR:-"$REPO_ROOT/../.evspdr-cache/matplotlib"}
MODE=${1:-create-or-update}

case "$MODE" in
    create-or-update|check) ;;
    *)
        echo "Usage: bash src/bootstrap_python312.sh [create-or-update|check]" >&2
        exit 2
        ;;
esac

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to build the pinned Python 3.12 environment" >&2
    exit 127
fi

cd "$REPO_ROOT"
mkdir -p "$MPL_CACHE"
export MPLCONFIGDIR="$MPL_CACHE"

if [[ "$MODE" == "create-or-update" ]]; then
    mkdir -p "$(dirname "$ENV_PREFIX")"
    if [[ -x "$ENV_PREFIX/bin/python" ]]; then
        conda env update --prefix "$ENV_PREFIX" --file "$ENV_FILE"
    else
        conda env create --prefix "$ENV_PREFIX" --file "$ENV_FILE"
    fi
elif [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Python environment not found: $ENV_PREFIX" >&2
    echo "Run: bash src/bootstrap_python312.sh create-or-update" >&2
    exit 2
fi

conda run --prefix "$ENV_PREFIX" python --version
conda run --prefix "$ENV_PREFIX" python -m compileall -q src tests
conda run --prefix "$ENV_PREFIX" python -m unittest discover -s tests
conda run --prefix "$ENV_PREFIX" python -u src/unicorn_preflight.py \
    --csv Practice_1bus.csv \
    --prices_csv hourly_prices_flat.csv \
    --mode NO_CHEAT \
    --skip_gurobi \
    --allow_dirty

echo "Python 3.12 environment validated: $ENV_PREFIX"
echo "For Unicorn jobs: export EVSP_CONDA_ENV=$ENV_PREFIX"
