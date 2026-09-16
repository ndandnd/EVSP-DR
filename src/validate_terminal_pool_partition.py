"""Separate exact-once MIP on a frozen terminal-CG pool; no new CG claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from run_terminal_energy_cg import save, sha, audit_energy
from config import BUS_COST_KX, CHARGING_STATIONS
from audit_giro_known_columns import build_problem, HORIZON_MIN
from expanded_path_realization import realize_expanded_path, realized_costs
from utils_v2 import load_station_hourly_prices
from types import SimpleNamespace


def solve(routes, trips, *, target, cap, seconds, log_dir, threads=1):
    import gurobipy as gp
    m = gp.Model('terminal_pool_exact_once')
    m.Params.Threads = threads
    m.Params.MIPGap = 0
    v = m.addVars(len(routes), vtype=gp.GRB.BINARY, obj=1)
    incidence = {t: [] for t in trips}
    for i, r in enumerate(routes):
        assert len(set(r['trips'])) == len(r['trips'])
        for t in r['trips']:
            incidence[t].append(i)
    for t, indices in incidence.items():
        m.addConstr(gp.quicksum(v[i] for i in indices) == 1, name=f'trip_{t}')
    m.addConstr(v.sum() <= cap, name='fleet_cap')
    m.addConstr(gp.quicksum(r['terminal_energy_kwh']*v[i] for i,r in enumerate(routes)) >= target,
                name='aggregate_return_energy')
    start = time.monotonic()
    m.Params.TimeLimit = seconds/2
    m.Params.LogFile = str(Path(log_dir)/'fleet.log')
    m.optimize()
    result = dict(fleet_status=m.Status, fleet=round(m.ObjVal) if m.SolCount else None,
                  fleet_bound=m.ObjBound if abs(m.ObjBound)<1e90 else None,
                  proof_scope='finite_frozen_pool_with_exact_once_trip_rows; no new CG certificate')
    selected = []
    if m.SolCount:
        starts = [v[i].X for i in range(len(routes))]
        m.addConstr(v.sum() <= round(m.ObjVal), name='incumbent_fleet_cap')
        for i, r in enumerate(routes):
            v[i].Obj = r['cost']-BUS_COST_KX
            v[i].Start = starts[i]
        m.Params.TimeLimit = max(1,seconds-(time.monotonic()-start))
        m.Params.LogFile = str(Path(log_dir)/'charging.log')
        m.optimize()
        selected = [r for i,r in enumerate(routes) if m.SolCount and v[i].X>.5]
        result.update(charging_status=m.Status, charging_cost=m.ObjVal if m.SolCount else None,
                      charging_bound=m.ObjBound if abs(m.ObjBound)<1e90 else None)
    result['seconds'] = time.monotonic()-start
    if selected:
        counts = Counter(t for r in selected for t in r['trips'])
        assert counts == Counter({t:1 for t in trips})
        assert sum(r['terminal_energy_kwh'] for r in selected) >= target-1e-6
        result['exact_once_verified'] = True
    m.dispose()
    return result, selected


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True)
    p.add_argument('--source-sha256',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=float,default=3600)
    p.add_argument('--threads',type=int,default=8)
    args=p.parse_args()
    source=Path(args.source)
    assert sha(source)==args.source_sha256
    summary=json.loads(source.read_text())
    pool=source.parent/'columns.jsonl'
    assert sha(pool)==summary['columns_sha256']
    for path,h in summary['input_hashes'].items():
        assert sha(path)==h
    paths=[Path(path) for path in summary['input_hashes']]
    instance=next(path for path in paths if 'Practice_Custom' in path.name)
    tariff=next(path for path in paths if path.name.startswith('peak'))
    data=next(path.parent for path in paths if path.name=='Ref_dict.csv')
    problem=build_problem(data,str(instance),max_station_to_trip_wait_min=HORIZON_MIN)
    prices=load_station_hourly_prices(tariff,CHARGING_STATIONS)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    routes=[json.loads(line) for line in pool.read_text().splitlines() if line.strip()]
    assert len(routes)==summary['columns']
    result,selected=solve(routes,problem.trips,target=summary['terminal_target_kwh'],
                          cap=summary['fleet_cap'],seconds=args.seconds,log_dir=out,threads=args.threads)
    net=SimpleNamespace(problem=problem,g=240,charge_kw=350,reserve=0,soc_step=2.5,block_min=5)
    for r in selected:
        audit_energy(net,r)
        realized,detail=realize_expanded_path(problem,r,g_kwh=240,charge_kw=350,reserve_kwh=0,
                                             soc_step=2.5,block_min=5,time_model='event')
        costs=realized_costs(realized,detail['mapping'],station_prices=prices)
        assert abs(costs['recomputed_expanded_grid_cost']-r['cost'])<1e-5
        assert abs(costs['continuous_realized_cost']-r['continuous_realized_cost'])<1e-5
    result.update(source=str(source),source_sha256=sha(source),pool_sha256=sha(pool),
                  individual_replay_verified=bool(selected),shared_capacity_validated=False,
                  continuous_charging_cost=sum(r['continuous_realized_cost']-BUS_COST_KX for r in selected) if selected else None,
                  terminal_kwh=sum(r['terminal_energy_kwh'] for r in selected) if selected else None)
    save(out/'selected_routes.json',selected)
    save(out/'summary.json',result)


if __name__=='__main__':
    main()
