#!/bin/bash
#SBATCH --partition=default_partition
#SBATCH --exclude=scaglione-compute-01
#SBATCH --nice=10000
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
set -euo pipefail
exec "$SOLVER_PYTHON" "$CAMPAIGN_ROOT/paired_code/gate.py" "$1"
