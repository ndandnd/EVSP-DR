#!/bin/bash
#SBATCH --partition=default_partition
#SBATCH --nice=10000
#SBATCH --exclude=scaglione-compute-01
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --requeue
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
: "${CAMPAIGN_ROOT:?}" "${CAPACITY_SOLVER:?}" "${CAPACITY_DATA:?}" "${SOLVER_PYTHON:?}"
exec "$SOLVER_PYTHON" "$CAMPAIGN_ROOT/paired_code/campaign.py" --k "$1" --index "${SLURM_ARRAY_TASK_ID:-0}" "${@:2}"
