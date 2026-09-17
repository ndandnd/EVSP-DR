"""Restricted-pool LP test: does removing mixed-vehicle-group columns raise the LP fleet bound?
Reads a CG's saved column journal, solves the covering LP over (a) the full pool, (b) unmixed columns only,
and reports total route weight, per-group weight, and ceil(LP_A)+ceil(LP_B). No pricing, no new columns."""
import sys, json, csv, math, time, os
import gurobipy as gp
from gurobipy import GRB

cg_json, master_csv, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
t0 = time.time()
cg = json.load(open(cg_json))
inst = os.path.join(os.path.dirname(cg_json).split('/cases/')[0], 'code', 'data', cg['csv'])
# trip (count_trip_id) -> group via Ordered_Trip_ID -> VehicleTask prefix
master = list(csv.DictReader(open(master_csv)))
reg = [r for r in master if r['Identifier'] == 'Regular']
ord2task = {int(r['Ordered_Trip_ID']): r['VehicleTask'] for r in reg}
group = {}
for r in csv.DictReader(open(inst)):
    task = ord2task[int(r['Ordered_Trip_ID'])]
    group[int(r['count_trip_id'])] = '18E1' if task.startswith('134') else '18E2' if task.startswith('133') else 'UNK'
trips = sorted(group)
assert set(trips) == set(cg['trip_ids']) or len(trips) == len(cg['trip_ids']), 'trip id mismatch'
n_trip = len(trips)
gA = sum(1 for t in trips if group[t] == '18E1'); gB = n_trip - gA

cols = []  # (cost, trips, mixed)
mixed_ct = 0
with open(cg['columns_journal']) as f:
    for line in f:
        d = json.loads(line)
        ts = d['trips']; gs = {group[t] for t in ts}
        mixed = len(gs) == 2
        mixed_ct += mixed
        cols.append((float(d['cost']), ts, mixed))
t_load = time.time() - t0

def solve(subset, label):
    m = gp.Model(label); m.Params.OutputFlag = 0; m.Params.Threads = int(os.environ.get('SLURM_CPUS_PER_TASK', 4)); m.Params.Method = 2
    x = m.addVars(len(subset), lb=0.0, obj=[c for c, _, _ in subset], name='x')
    cover = {t: [] for t in trips}
    for j, (_, ts, _) in enumerate(subset):
        for t in ts: cover[t].append(j)
    for t in trips:
        m.addConstr(gp.quicksum(x[j] for j in cover[t]) >= 1, name=f'c{t}')
    m.optimize()
    if m.Status != GRB.OPTIMAL: return dict(status=int(m.Status))
    lam = [x[j].X for j in range(len(subset))]
    tot = math.fsum(lam)
    # per-group weight: attribute a route's weight to its group if pure; mixed routes tracked separately
    pureA = math.fsum(l for l, (_, ts, mx) in zip(lam, subset) if not mx and group[ts[0]] == '18E1')
    pureB = math.fsum(l for l, (_, ts, mx) in zip(lam, subset) if not mx and group[ts[0]] == '18E2')
    mixw = math.fsum(l for l, (_, _, mx) in zip(lam, subset) if mx)
    return dict(status='OPTIMAL', objective=m.ObjVal, total_route_weight=tot, pure_18E1_weight=pureA, pure_18E2_weight=pureB,
                mixed_weight=mixw, positive_routes=sum(1 for l in lam if l > 1e-9), columns=len(subset), runtime_s=m.Runtime)

full = solve(cols, 'full')
unm = solve([c for c in cols if not c[2]], 'unmixed')
seg_bound = None
if unm.get('status') == 'OPTIMAL':
    seg_bound = math.ceil(unm['pure_18E1_weight'] - 1e-6) + math.ceil(unm['pure_18E2_weight'] - 1e-6)
out = dict(case=os.path.basename(os.path.dirname(cg_json)), cg_json=cg_json, instance_csv=inst, trips=n_trip, trips_18E1=gA, trips_18E2=gB,
           columns_total=len(cols), columns_mixed=mixed_ct, recorded_final_lp=cg.get('final_lp', {}).get('objective'),
           recorded_route_weight=cg.get('final_lp', {}).get('route_weight') or cg.get('final', {}).get('route_weight'),
           certified=cg.get('certified_rc_optimal'), full_pool_lp=full, unmixed_pool_lp=unm,
           segregated_integer_fleet_lower_bound_in_pool=seg_bound, load_s=t_load, total_s=time.time() - t0,
           note='Restricted-pool LPs: upper bounds on the true LP of each model (no pricing). Unmixed LP >= full LP always. '
                'seg bound = ceil(LP_A)+ceil(LP_B) over pure columns; valid lower bound for a segregated integer fleet *within this pool* only.')
json.dump(out, open(out_path, 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('full_pool_lp', 'unmixed_pool_lp')}, indent=None))
print('FULL   ', {k: round(v, 4) if isinstance(v, float) else v for k, v in full.items()})
print('UNMIXED', {k: round(v, 4) if isinstance(v, float) else v for k, v in unm.items()})
