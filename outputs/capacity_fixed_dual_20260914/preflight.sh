#!/bin/bash
set -euo pipefail
root=/home/nc437/ladder-lite/capacity_fixed_dual_20260914
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
cd "$root"
exec /home/nc437/evsp_env/bin/python -u preflight.py "$root"
