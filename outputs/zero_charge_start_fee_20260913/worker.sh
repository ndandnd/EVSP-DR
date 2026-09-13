#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONHASHSEED=0
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
export PYTHON_BIN=/home/nc437/evsp_env/bin/python
"$PYTHON_BIN" - <<'PY'
import gurobipy as gp
m = gp.Model('fee_comparison_license_check')
m.Params.OutputFlag = 0
m.Params.Threads = 1
x = m.addVars(3001, lb=0, ub=1, obj=1)
m.addConstr(x.sum() >= 1)
m.optimize()
assert m.Status == gp.GRB.OPTIMAL
print('Gurobi preflight passed with 3001 variables', flush=True)
m.dispose()
PY
exec "$PYTHON_BIN" -u /home/nc437/ladder-lite/zero_charge_start_fee_20260913/tooling/campaign.py worker --root /home/nc437/ladder-lite/zero_charge_start_fee_20260913 --index "$1"
