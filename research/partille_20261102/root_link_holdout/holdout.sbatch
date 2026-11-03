#!/bin/bash
#SBATCH --job-name=drLH1103
#SBATCH --partition=default_partition
#SBATCH --nice=0
#SBATCH --exclude=scaglione-compute-01
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
: "${HOLDOUT_ROOT:?}"
exec /home/nc437/evsp_env/bin/python -u "$HOLDOUT_ROOT/code/array_worker.py" --root "$HOLDOUT_ROOT" --index "$SLURM_ARRAY_TASK_ID"
