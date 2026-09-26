#!/bin/bash
# Case-pinned recovery only; submission/dependencies are recorded separately.
#SBATCH --partition=default_partition
#SBATCH --exclude=scaglione-compute-01
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --job-name=sx_rc_mix1_two_price_split
#SBATCH --chdir=/home/nc437/ladder-lite/spatial_tariff_expansion_20260925
#SBATCH --output=/home/nc437/ladder-lite/spatial_tariff_expansion_20260925/logs/%x_%j.out
#SBATCH --error=/home/nc437/ladder-lite/spatial_tariff_expansion_20260925/logs/%x_%j.err
set -euo pipefail
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUTF8=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
RECOVERY_DIR=/home/nc437/ladder-lite/spatial_tariff_expansion_20260925/recovery_20260926_case11
exec /home/nc437/evsp_env/bin/python -u "$RECOVERY_DIR/recovery_worker.py"
