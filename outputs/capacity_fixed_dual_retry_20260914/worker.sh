#!/bin/bash
set -euo pipefail

root=/home/nc437/ladder-lite/capacity_fixed_dual_retry_20260914
case_id=${1:?case id required}
attempt=${SLURM_JOB_ID:?SLURM_JOB_ID required}_r${SLURM_RESTART_COUNT:-0}

unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic

exec /home/nc437/evsp_env/bin/python -u "$root/campaign.py" \
  --root "$root" --case "$case_id" --attempt "$attempt"
