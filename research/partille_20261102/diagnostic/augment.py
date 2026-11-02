"""Membership and finite-pool fleet experiment on the frozen v15 pool plus replayed witnesses."""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
from run_capacity_terminal_cg import DEFAULT_PLUGS
from capacity_event_pricer import record_occupancy
from run_terminal_energy_cg import sha


def column_key(route):
    return (frozenset(route['trips']), route['terminal_energy_kwh'], route['cost'],
            tuple(sorted((s,m,n) for (s,m),n in record_occupancy(route).items() if DEFAULT_PLUGS[s] is not None)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','witness','verified','out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--start-fee', type=float, required=True, choices=[5])
    a=p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    verified=json.loads(a.verified.read_text())
    assert verified['validated'] and verified['arm']=='fixed_giro_diagnostic' and verified['exact_once']
    assert sha(a.witness)==verified['selected_sha256']
    witness=json.loads(a.witness.read_text())
    summary=json.loads((a.baseline/'summary.json').read_text())
    assert summary['plugs']==DEFAULT_PLUGS and verified['physics']==summary['physics']
    assert verified['inputs_sha256']==summary['input_hashes']
    pool=a.baseline/'columns.jsonl'
    root_hash=hashlib.sha256()
    file_hash=hashlib.sha256()
    rows={t: [] for route in witness for t in route['trips']}
    plugs=defaultdict(list)
    energies=[]
    duty_rows={tuple(route['trips']): [] for route in witness}
    candidates={column_key(route): None for route in witness}
    def add(i,r):
        for t in r['trips']:
            rows[t].append(i)
        for (s,m),n in record_occupancy(r).items():
            if DEFAULT_PLUGS[s] is not None:
                plugs[s,m].append((i,n))
        energies.append(r['terminal_energy_kwh'])
    with pool.open('rb') as f:
        for i, line in enumerate(f):
            file_hash.update(line)
            if i < summary['columns']:
                root_hash.update(line)
            r=json.loads(line)
            add(i,r)
            seq=tuple(r['trips'])
            if seq in duty_rows:
                duty_rows[seq].append(i)
                key=column_key(r)
                if key in candidates:
                    candidates[key]=i
    assert root_hash.hexdigest()==summary['columns_sha256'], 'source root LP columns hash differs'
    n_before=len(energies)
    assert n_before==summary['dive']['columns_after_dive']
    witness_ids=[]
    for r in witness:
        old=candidates[column_key(r)]
        if old is None:
            old=len(energies)
            add(old,r)
        witness_ids.append(old)
    with gp.Model('augmented_fleet_diagnostic') as mip:
        mip.Params.LogFile=str(a.out/'fleet.gurobi.log')
        mip.Params.Threads=int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
        mip.Params.TimeLimit=1800
        mip.Params.MIPGap=0
        # Time-only floor derived from the source instance; this experiment tests feasibility.
        mip.Params.BestObjStop=verified['peak_simultaneous_trips']+1e-7
        x=mip.addVars(len(energies),vtype=GRB.BINARY,obj=1)
        for t,ids in rows.items():
            mip.addConstr(gp.quicksum(x[i] for i in ids)>=1)
        for (s,m),pairs in plugs.items():
            if sum(n for _,n in pairs)>DEFAULT_PLUGS[s]:
                mip.addConstr(gp.LinExpr([n for _,n in pairs], [x[i] for i,_ in pairs])<=DEFAULT_PLUGS[s])
        mip.addConstr(x.sum()<=summary['fleet_cap'])
        mip.addConstr(gp.LinExpr(energies,list(x.values()))>=summary['terminal_target_kwh'])
        for i,var in x.items():
            var.Start=float(i in witness_ids)
        mip.optimize()
        ids=[i for i,v in x.items() if mip.SolCount and v.X>0.5]
        result=dict(status=mip.Status, nsolutions=mip.SolCount, bound=mip.ObjBound, solver_seconds=mip.Runtime,
                    fleet=len(ids) if ids else None)
    result.update(baseline_fleet=summary['fleet_stage']['buses'], baseline_bound=summary['fleet_stage']['bound'],
                  historical_root_sha256=root_hash.hexdigest(), full_pool_sha256=file_hash.hexdigest(),
                  source_summary_sha256=sha(a.baseline/'summary.json'), source_columns=n_before,
                  old_giro_sequence_count=sum(bool(v) for v in duty_rows.values()),
                  new_witness_columns=len(energies)-n_before, witness_sha256=sha(a.witness),
                  subset='v15 entire post-dive pool plus distinct verified GIRO sequence realizations',
                  physics=summary['physics'],inputs_sha256=summary['input_hashes'],
                  selection_stop='fleet feasibility down to time-only lower bound; no charging optimization')
    chosen={}
    if ids:
        with pool.open('rb') as f:
            for i,line in enumerate(f):
                if i in ids: chosen[i]=json.loads(line)
        for i,r in zip(witness_ids,witness):
            if i>=n_before and i in ids: chosen[i]=r
        selected=[chosen[i] for i in ids]
        (a.out/'selected_routes.json').write_text(json.dumps(selected,indent=2)+'\n')
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    main()
