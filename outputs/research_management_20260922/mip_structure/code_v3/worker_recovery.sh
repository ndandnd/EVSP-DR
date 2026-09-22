#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
exec /home/nc437/evsp_env/bin/python "$1/code_v2/runner.py" "$@" --manifest "$1/manifest_v2.json"
