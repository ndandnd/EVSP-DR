#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
exec /home/nc437/evsp_env/bin/python -u /home/nc437/ladder-lite/zero_charge_start_fee_20260913/validate_mip_retry.py --code /home/nc437/ladder-lite/zero_charge_start_fee_20260913/code_06b5cb8 --commit 06b5cb86d6c24df0ec0a5ca7189fa9552f527dd0
