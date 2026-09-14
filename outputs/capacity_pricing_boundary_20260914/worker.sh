#!/bin/bash
set -euo pipefail

root=/home/nc437/ladder-lite/capacity_pricing_boundary_20260914
index=${1:?case index required}
attempt_id=${SLURM_JOB_ID:?SLURM_JOB_ID required}_r${SLURM_RESTART_COUNT:-0}

unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic

exec /home/nc437/evsp_env/bin/python "$root/campaign.py" worker \
  --manifest "$root/manifest.json" \
  --campaign-root "$root" \
  --code-root "$root/code" \
  --index "$index" \
  --attempt-id "$attempt_id" \
  --python /home/nc437/evsp_env/bin/python
