#!/bin/bash
#SBATCH --partition=default_partition
#SBATCH --exclude=scaglione-compute-01
#SBATCH --nice=0
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --requeue
# Operational amendment: 2026-11-02 Pacific / 2026-11-03 UTC; as-submitted template used nice=10000.
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
exec "$SOLVER_PYTHON" "$PREFIX_ROOT/code/sparse_worker.py" --k "$1" --chain "$((SLURM_ARRAY_TASK_ID+1))"
