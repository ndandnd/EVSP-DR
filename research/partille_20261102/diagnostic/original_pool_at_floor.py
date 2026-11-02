"""Does the immutable C2 k8 post-dive pool contain an eight-bus selection?"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
from capacity_event_pricer import record_occupancy

FULL_HASH = '5744645fa8fc62e3c79e3880a12b1388dda8e0342fe46bb6c700d61e98c86d0d'
INSTANCE_HASH = 'c1dbe1dd15d20979799c68d01430a5fd0eddc2ca3ced3f46bd8522123e6c480d'


def minutes(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m


def simultaneous(instance):
    raw = instance.read_bytes()
    if hashlib.sha256(raw).hexdigest() != INSTANCE_HASH:
        raise ValueError('unrecognized instance')
    spans = [(minutes(r['Start1']), minutes(r['End1'])) for r in csv.DictReader(raw.decode().splitlines())]
    if any(b <= a for a,b in spans):
        raise ValueError('nonpositive service interval')
    events = sorted({a for a,b in spans} | {b for a,b in spans})
    active = {frozenset(i for i, (a,b) in enumerate(spans) if a <= t < b) for t in events}
    floor = max(map(len,active))
    # Intervals are half open; scanning every distinct start/end visits every active set on the day.
    return [g for g in active if len(g)==floor], floor, len(spans)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pool','instance','summary','output'):
        p.add_argument('--'+key, required=True, type=Path)
    p.add_argument('--seconds', required=True, type=float)
    p.add_argument('--threads', required=True, type=int)
    p.add_argument('--arm', choices=['partition','cover'], required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    s=json.loads(a.summary.read_text())
    assert s['physics']['shared_capacity'] and s['physics']['station_kw']=={'PARX':60}
    assert s['terminal_target_kwh']==0 and s['fleet_cap']==12
    groups, floor, ntrips=simultaneous(a.instance)
    saturated = set().union(*groups)
    root, full = hashlib.sha256(), hashlib.sha256()
    eligible_ids, cover, occupancy, terminal = [], defaultdict(list), defaultdict(list), []
    count=0
    with a.pool.open('rb') as f:
        for count, line in enumerate(f, start=1):
            full.update(line)
            if count<=s['columns']:
                root.update(line)
            r=json.loads(line)
            trips=set(r['trips'])
            assert len(trips)==len(r['trips']) and trips<=set(range(ntrips))
            hits=[len(trips & g) for g in groups]
            if max(hits)>1:
                raise ValueError('a source route serves overlapping service trips; filter invalid')
            if min(hits)==0:
                continue
            i=len(eligible_ids)
            eligible_ids.append(count-1)
            for t in trips:
                cover[t].append(i)
            terminal.append(r['terminal_energy_kwh'])
            for (station,m),n in record_occupancy(r).items():
                if s['plugs'][station] is not None:
                    occupancy[station,m].append((i,n))
    if root.hexdigest()!=s['columns_sha256'] or full.hexdigest()!=FULL_HASH or count!=s['dive']['columns_after_dive']:
        raise ValueError('wrong or truncated original pool')
    with gp.Model('original_pool_at_time_floor') as m:
        m.Params.LogFile=str(a.output/'solver.log')
        m.Params.Threads=a.threads
        m.Params.TimeLimit=a.seconds
        m.Params.DualReductions=0
        m.Params.MIPGap=0
        x=m.addVars(len(eligible_ids),vtype=GRB.BINARY)
        m.addConstr(x.sum()==floor, name='fleet_at_proved_floor')
        for t in range(ntrips):
            expr=gp.quicksum(x[i] for i in cover[t])
            m.addConstr(expr==1 if a.arm=='partition' or t in saturated else expr>=1)
        for (station,minute), incidences in occupancy.items():
            if sum(n for _,n in incidences)>s['plugs'][station]:
                m.addConstr(gp.LinExpr([n for _,n in incidences], [x[i] for i,_ in incidences])<=s['plugs'][station])
        m.addConstr(gp.LinExpr(terminal,list(x.values()))>=s['terminal_target_kwh'])
        m.setObjective(gp.LinExpr(),GRB.MINIMIZE)
        m.optimize()
        selected={eligible_ids[i] for i,var in x.items() if m.SolCount and var.X>.5}
        result=dict(status=m.Status, solutions=m.SolCount, solver_seconds=m.Runtime, explored_nodes=m.NodeCount,
                    model_rows=m.NumConstrs,model_columns=m.NumVars,solver_version=gp.gurobi.version(),
                    finite_pool_infeasible=(m.Status==GRB.INFEASIBLE),fleet=len(selected) if selected else None)
    result.update(instance_sha256=INSTANCE_HASH,source_pool_sha256=FULL_HASH,
                  root_prefix_sha256=root.hexdigest(),pool_size=count,eligible_count=len(eligible_ids),
                  filtered_count=count-len(eligible_ids),peak_simultaneous_trips=floor,
                  distinct_peak_sets=len(groups),distinct_trips_in_peak_sets=len(saturated),
                  trip_rows=a.arm,threads=a.threads,time_limit=a.seconds,
                  objective='feasibility at the independently derived fleet floor',
                  filter='Every selected route must hit every simultaneous peak set once; any failure was excluded.')
    if selected:
        with a.pool.open('rb') as f:
            chosen=[json.loads(line) for i,line in enumerate(f) if i in selected]
        path=a.output/'selected_routes.json'
        path.write_text(json.dumps(chosen,indent=2)+'\n')
        result['selected_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
