#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
export PYTHON_BIN=/home/nc437/evsp_env/bin/python
exec "$PYTHON_BIN" -u /home/nc437/ladder-lite/giro_zero_start_fee_20260913/scripts/event_uniform_envelope/giro_zero_fee_campaign.py worker \
  --root /home/nc437/ladder-lite/giro_zero_start_fee_20260913 \
  --pair-id "$1" --fee "${1##*_fee}" --stage "$2"
