#!/bin/bash
set -euo pipefail
root=${1:?campaign root required}
caseid=${2:?case required}
mode=${3:?mode required}
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
exec /home/nc437/evsp_env/bin/python "$root/worker.py" --root "$root" --code "$root/code" --case "$caseid" --mode "$mode"
