#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "$SCRIPT_DIR/common.sh"
evsp_require_unicorn
[[ $# -le 1 ]] || evsp_die "usage: $0 [CAMPAIGN_ROOT]"
REPO=$(evsp_repo_root)
BRANCH=$(git -C "$REPO" branch --show-current)
[[ -n "$BRANCH" ]] || evsp_die "named branch required"
COMMIT=$(evsp_verify_remote_head "$REPO" "$BRANCH" | tail -1)
EXECUTION_REPO=$(evsp_execution_checkout "$REPO" "$COMMIT")
PYTHON_BIN="${EVSP_PYTHON:-$HOME/evsp_env/bin/python}"
ROOT="${1:-$HOME/ladder-lite/terminal_energy_fair_20260910_${COMMIT:0:7}}"
mkdir -p "$(dirname "$ROOT")"
exec 9>"$ROOT.launch.lock"
flock -n 9 || evsp_die "another launcher owns $ROOT"

# A model above the size-limited license threshold makes this a full-license
# preflight, rather than merely testing that gurobipy imports.
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
unset LM_LICENSE_FILE || true
[[ -r "$GRB_LICENSE_FILE" ]] || evsp_die "shared Gurobi license is unreadable"
"$PYTHON_BIN" - <<'PY'
import gurobipy as gp
m=gp.Model('full_license_preflight');m.Params.OutputFlag=0
x=m.addVars(2101,lb=0,ub=1);m.addConstr(gp.quicksum(x.values())>=1)
m.setObjective(gp.quicksum(x.values()),gp.GRB.MINIMIZE);m.optimize()
assert m.Status==gp.GRB.OPTIMAL and abs(m.ObjVal-1)<1e-8
PY

if [[ ! -e "$ROOT" ]]; then
  "$PYTHON_BIN" "$EXECUTION_REPO/scripts/event_uniform_envelope/terminal_energy_fair_pilot.py" prepare \
    --root "$ROOT" --commit "$COMMIT" \
    --source peak08="$HOME/ladder-lite/matched_tariff8_20260908_a9a9720/original_eligible_k05_g240_p350_peak08" \
    --source peak12="$HOME/ladder-lite/matched_tariff_peak12_peak18_a6e5059/original_eligible_k05_g240_p350_peak12" \
    --source peak18="$HOME/ladder-lite/matched_tariff_peak12_peak18_a6e5059/original_eligible_k05_g240_p350_peak18"
fi
PLAN_SHA=$(sha256sum "$ROOT/plan.json" | awk '{print $1}')
"$PYTHON_BIN" - "$ROOT/plan.json" "$COMMIT" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));assert p['commit']==sys.argv[2]
assert p['schema']=='evsp-dr-terminal-energy-fair-pilot-v1'
assert len(p['cells'])==3 and p['master_sense']=='cover'
assert p['target_physical_terminal_energy_kwh']==280.7833253
assert p['per_bus_terminal_floor_kwh']==0.0 and p['fleet_cap']==5
PY
WORKER="$EXECUTION_REPO/scripts/event_uniform_envelope/terminal_energy_fair_worker.sub"
WORKER_SHA=$(sha256sum "$WORKER" | awk '{print $1}')
COMMON="HOME,PATH,USER,EVSP_EXECUTION_REPO=$EXECUTION_REPO,EVSP_EXPECTED_COMMIT=$COMMIT,EVSP_CAMPAIGN_ROOT=$ROOT,EVSP_PLAN_SHA256=$PLAN_SHA,EVSP_PYTHON=$PYTHON_BIN,EVSP_WORKER_SHA256=$WORKER_SHA"
tag=$(printf '%s' "$ROOT" | shasum | cut -c1-6)

record() {
  "$PYTHON_BIN" - "$ROOT/submission.$1.json" "$@" "$COMMIT" "$PLAN_SHA" <<'PY'
import json,os,sys
path,stage,job,dep,partition,cpus,memory,allocation,commit,digest=sys.argv[1:]
with open(path,'x') as f:
 json.dump(dict(stage=stage,job_id=job,dependency=dep,partition=partition,
  cpus=int(cpus),memory=memory,allocation=allocation,commit=commit,
  plan_sha256=digest,array=('0-2%2' if stage=='mip' else '0-2%3'),requeue=False,
  excluded_nodes=['scaglione-compute-01']+(['scaglione-cpu-04'] if stage=='mip' else [])),f,indent=2)
 f.write('\n');f.flush();os.fsync(f.fileno())
PY
}
[[ ! -e "$ROOT/submission.frontier.json" ]] || evsp_die "frontier already submitted"
[[ ! -e "$ROOT/submission.mip.json" ]] || evsp_die "MIP already submitted"

FNAME="tef${COMMIT:0:4}$tag"
FRONTIER=$(evsp_submit_and_resolve "$FNAME" --array=0-2%3 -p default_partition \
  --exclude=scaglione-compute-01 -c 1 --mem=24G -t 02:00:00 --no-requeue \
  --open-mode=append --export="$COMMON,EVSP_STAGE=frontier" \
  -o "$ROOT/logs/frontier_%A_%a.out" -e "$ROOT/logs/frontier_%A_%a.err" "$WORKER")
record frontier "$FRONTIER" none default_partition 1 24G 02:00:00

MNAME="tem${COMMIT:0:4}$tag"
MIP=$(evsp_submit_and_resolve "$MNAME" --array=0-2%2 -p scaglione \
  --exclude=scaglione-compute-01,scaglione-cpu-04 -c 8 --mem=48G -t 02:00:00 \
  --no-requeue --open-mode=append --dependency="aftercorr:$FRONTIER" \
  --kill-on-invalid-dep=yes --export="$COMMON,EVSP_STAGE=mip" \
  -o "$ROOT/logs/mip_%A_%a.out" -e "$ROOT/logs/mip_%A_%a.err" "$WORKER")
record mip "$MIP" "aftercorr:$FRONTIER" scaglione 8 48G 02:00:00
echo "Terminal-energy pilot: frontier $FRONTIER -> MIP $MIP at $ROOT"
