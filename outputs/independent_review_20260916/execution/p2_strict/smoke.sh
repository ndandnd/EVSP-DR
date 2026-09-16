#!/bin/bash
set -euo pipefail
root=${1:?root}
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/nc437/evsp_env/bin/python "$root/native_smoke.py"
