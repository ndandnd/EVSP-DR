#!/bin/bash

# Every Gurobi worker overrides inherited login-node license variables and
# validates the token before opening a large pool or starting optimization.
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
unset LM_LICENSE_FILE || true

[[ -r "$GRB_LICENSE_FILE" ]] || {
  echo "Gurobi license is missing or unreadable: $GRB_LICENSE_FILE" >&2
  exit 2
}

: "${PYTHON_BIN:?PYTHON_BIN must be set before Gurobi preflight}"
"$PYTHON_BIN" - <<'PY'
import gurobipy as gp

model = gp.Model("evsp_worker_preflight")
model.Params.OutputFlag = 0
model.Params.Threads = 1
x = model.addVar(lb=0.0, ub=1.0, obj=1.0)
model.addConstr(x >= 0.5)
model.optimize()
if model.Status != gp.GRB.OPTIMAL:
    raise SystemExit(f"Gurobi preflight status={model.Status}")
print("[GUROBI] preflight OK", ".".join(map(str, gp.gurobi.version())))
model.dispose()
PY
