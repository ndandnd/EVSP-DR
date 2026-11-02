"""Continuous feasibility of the authenticated eight-bus restricted partition master.

Every eligible archived record remains a separate column. For eight buses,
all four simultaneous peaks require one trip per route even fractionally:
every route contributes at most one to a peak and total weight equals eight.
"""
import argparse
from array import array
import collections
import gzip
import json
import math
from pathlib import Path

import gurobipy as gp
import numpy as np
from scipy.sparse import coo_matrix
from verify import read_inputs,require,sha

p=argparse.ArgumentParser(description=__doc__)
for k in ('instance','journal','summary','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args(); rep,_,anchors,n,fleet=read_inputs(a.instance,a.journal,a.summary)
meta=json.loads(a.summary.read_bytes());plugs=meta['plugs'];terminal=meta['terminal_target_kwh'];tol=1e-9

def live_routes():
    with gzip.open(a.journal,'rb') as f:
        for raw in f:
            r=json.loads(raw)
            bits=sum(1<<i for i in r['trips'])
            if all(bits & peak for peak in anchors): yield r

def finite_occupancy(r):
    stops=r['expanded_grid_charging_stops']
    occ=collections.Counter()
    for station,start,end in zip(stops['stations'],stops['cst'],stops['cet'],strict=True):
        head,sep,tail=station.rpartition('_'); base=head if sep and tail.isdigit() else station
        if plugs[base] is not None:
            for m in range(math.floor(start+tol),math.ceil(end-tol)): occ[base,m]+=1
    return occ

row_key={}; ii=array('i'); jj=array('i'); vv=array('d');col_count=0
for j,r in enumerate(live_routes()):
    col_count=j+1
    rows=list(r['trips'])+[n,n+1]
    vals=[1.]*(len(rows)-1)+[r['terminal_energy_kwh']]
    occ=finite_occupancy(r)
    stored={(s,m):v for s,m,v in r['capacity_occupancy'] if plugs[s] is not None}
    require(stored==occ,'minute coefficients disagree with authenticated journal')
    for key,v in occ.items():
        idx=row_key.setdefault(key,len(row_key)+n+2)
        rows.append(idx);vals.append(v)
    ii.extend(rows);jj.extend([j]*len(rows));vv.extend(vals)
require(col_count==rep['records_touching_all_maximum_sets'],'wrong eligible columns')
rhs=np.array([1.]*n+[float(fleet),terminal]+[float(plugs[k[0]]) for k in row_key]); sense=np.array(['=']*(n+1)+['>']+['<']*len(row_key))
A=coo_matrix((vv,(ii,jj)),shape=(len(rhs),col_count)).tocsr()
with gp.Env(empty=True) as env:
    env.setParam('OutputFlag',0);env.start()
    with gp.Model(env=env) as model:
        model.Params.Threads=2; model.Params.TimeLimit=60.;model.Params.FeasibilityTol=1e-9
        x=model.addMVar(col_count,lb=0.,ub=1.); model.addMConstr(A,x,sense,rhs); model.optimize()
        rep.update(gurobi_version=gp.gurobi.version(),solver_status=model.Status,sol_count=model.SolCount,solver_seconds=model.Runtime,
                   solver_limit_seconds=model.Params.TimeLimit,threads=model.Params.Threads)
        weights=x.X.copy() if model.SolCount else None
rep.update(constraints='trip equality, fleet equality, original terminal lower bound, all finite station-minute upper bounds, 0<=x<=1',
           eligible_columns=col_count, finite_minute_rows=len(row_key),checked_all_eligible_occupancy_against_source_windows=True,
           matrix_nonzeros=A.nnz.item() if hasattr(A.nnz,'item') else int(A.nnz),
           verification_source_sha256=sha(Path(__file__).read_bytes()),input_adapter_sha256=sha(Path(__file__).with_name('verify.py').read_bytes()))
if weights is not None:
    coverage=np.zeros(n);total=term=0.; occ=collections.Counter()
    for j,r in enumerate(live_routes()):
        x=weights[j]
        if x==0.: continue
        total+=x;term+=x*r['terminal_energy_kwh']
        for t in r['trips']:coverage[t]+=x
        for key,count in finite_occupancy(r).items():occ[key]+=x*count
    errors={'max_abs_trip_residual':float(max(abs(coverage-1))),'abs_fleet_residual':float(abs(total-fleet)),
            'max_plug_violation':float(max([0.]+[occ[key]-plugs[key[0]] for key in row_key])),
            'terminal_shortfall':float(max(0.,terminal-term)),
            'max_bound_violation':float(max(0.,-min(weights),max(weights)-1.))}
    rep.update(residuals=errors,feasible_to_1e_7=max(errors.values())<=1e-7,interior_columns=int(np.count_nonzero((weights>1e-7)&(weights<1.-1e-7))))
a.output.write_text(json.dumps(rep,indent=2)+'\n'); print(json.dumps(rep))
