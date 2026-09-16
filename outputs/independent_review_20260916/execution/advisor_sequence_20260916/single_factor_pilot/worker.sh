#!/bin/bash
set -euo pipefail
campaign_root=${1:?campaign root required}
pilot_arm=${2:?pilot arm required}
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
exec /home/nc437/evsp_env/bin/python "$campaign_root/worker.py" --root "$campaign_root" --arm "$pilot_arm"
