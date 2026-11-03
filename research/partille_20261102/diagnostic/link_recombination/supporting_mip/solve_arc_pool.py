"""Authenticate fresh trip paths, independently replay, and optimize an exact finite-pool selection.

Run with PYTHONPATH pointing at the pinned v15 solver src and --start-fee 5.
Inputs and all route-level outputs belong in a private directory. No historical duty labels are read.
"""
import argparse, collections, hashlib, json, os, subprocess, sys, time
from pathlib import Path
T0=time.perf_counter()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
p=argparse.ArgumentParser(description=__doc__)
for name in ['owner-summary','pool','routes','paths','retime-receipt','link-receipt','data','instance','validator','out']:
    p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--start-fee',type=float,required=True,choices=[5])
a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
import gurobipy as gp
from gurobipy import GRB
from capacity_event_pricer import record_occupancy
from run_capacity_terminal_cg import DEFAULT_PLUGS
summary=read(a.owner_summary);retime=read(a.retime_receipt);link=read(a.link_receipt)
routes=read(a.routes);paths=read(a.paths)
assert len(routes)==len(paths)==8 and [r['trips'] for r in routes]==paths
assert sha(a.paths)==retime['paths_sha256']==link['paths_sha256']
assert retime['validated'] and retime['exact_once'] and not retime['witness_data_read'] and not link['witness_data_read']
assert link['paths_new_to_original_pool']==8
assert sorted(t for r in routes for t in r['trips'])==list(range(173))
inputs={'data_dir':str(a.data),'instance':str(a.instance),'tariff':str(a.data/'hourly_prices_flat.csv')}
input_paths=[a.instance,a.data/'hourly_prices_flat.csv',a.data/'Ref_dict.csv',a.data/'par_ref_dhd.csv']
input_hashes={str(x):sha(x) for x in input_paths}
assert sorted(input_hashes.values())==sorted(summary['input_hashes'].values())==sorted(retime['input_sha256'].values())
for source,digest in retime['source_code_sha256'].items():
    # Independently replay against the imported solver's source tree.
    target=Path(sys.modules['run_capacity_terminal_cg'].__file__).parent/Path(source).name
    assert sha(target)==digest, f'solver source differs: {target.name}'
replay_summary={**summary,'inputs':inputs,'input_hashes':input_hashes,'fleet_cap':8}
write(a.out/'replay_summary.json',replay_summary)
def replay(selected,name):
    start=time.perf_counter()
    with (a.out/(name+'.log')).open('w') as f:
        r=subprocess.run([sys.executable,str(a.validator),'--summary',str(a.out/'replay_summary.json'),
                          '--selected',str(selected),'--out',str(a.out/(name+'.json')),
                          '--arm','augmented','--start-fee','5'],stdout=f,stderr=subprocess.STDOUT)
    v=read(a.out/(name+'.json'))
    assert r.returncode==0 and v['validated'] and v['exact_once'] and v['grid_plug_overload_minutes']==0,v.get('reason')
    return v,time.perf_counter()-start
initial,initial_seconds=replay(a.routes,'initial_replay')
mark=time.perf_counter();records=[];full=hashlib.sha256();prefix=hashlib.sha256()
for i,line in enumerate(a.pool.open('rb')):
    full.update(line)
    if i<summary['columns']:prefix.update(line)
    records.append(json.loads(line))
assert len(records)==39390==summary['dive']['columns_after_dive']
assert full.hexdigest()==link['source_pool_uncompressed_sha256']=='5744645fa8fc62e3c79e3880a12b1388dda8e0342fe46bb6c700d61e98c86d0d'
assert prefix.hexdigest()==summary['columns_sha256']
oldseq={tuple(r['trips']) for r in records}
assert all(tuple(r['trips']) not in oldseq for r in routes)
records.extend(routes)
tripids=sorted({t for r in routes for t in r['trips']});tripcols={t:[] for t in tripids};plugcols=collections.defaultdict(list)
for i,r in enumerate(records):
    assert len(r['trips'])==len(set(r['trips']))
    for t in r['trips']:tripcols[t].append(i)
    for (s,m),n in record_occupancy(r).items():
        if DEFAULT_PLUGS[s] is not None:plugcols[s,m].append((i,n))
load_seconds=time.perf_counter()-mark;mark=time.perf_counter()
m=gp.Model('arc_assembled_exact_partition_fixed8')
m.Params.LogFile=str(a.out/'gurobi.log');m.Params.Threads=2;m.Params.TimeLimit=120;m.Params.MIPGap=0
x=m.addVars(len(records),vtype=GRB.BINARY,obj=[r['cost']-100000 for r in records],name='route')
for t,ids in tripcols.items():m.addConstr(gp.quicksum(x[i] for i in ids)==1,name=f'trip_{t}')
for (s,minute),pairs in plugcols.items():
    m.addConstr(gp.LinExpr([n for _,n in pairs],[x[i] for i,_ in pairs])<=DEFAULT_PLUGS[s],name=f'plug_{s}_{minute}')
m.addConstr(x.sum()==8,name='fixed_fleet')
m.addConstr(gp.LinExpr([r['terminal_energy_kwh'] for r in records],list(x.values()))>=summary['terminal_target_kwh'],name='terminal')
for i,v in x.items():v.Start=float(i>=39390)
m.update();build_seconds=time.perf_counter()-mark;mark=time.perf_counter();m.optimize();optimize_wall=time.perf_counter()-mark
assert m.SolCount
ids=[i for i,v in x.items() if v.X>0.5];selected=[records[i] for i in ids]
write(a.out/'selected_routes.json',selected)
final,final_seconds=replay(a.out/'selected_routes.json','final_replay')
assert abs(final['objective_nonfleet']-m.ObjVal)<1e-6
aggregate={'schema':'archive-adjacency-new-paths-exact-pool-v1','source_hashes':{
    'owner_summary':sha(a.owner_summary),'original_pool':full.hexdigest(),'original_root_prefix':prefix.hexdigest(),
    'paths':sha(a.paths),'retimer_routes':sha(a.routes),'retimer_receipt':sha(a.retime_receipt),
    'link_receipt':sha(a.link_receipt),'validator_code':sha(a.validator),'driver_code':sha(__file__),
    'final_selection':sha(a.out/'selected_routes.json')},
    'input_hashes':{q.name:input_hashes[str(q)] for q in input_paths},
    'solver_source_hashes':{Path(q).name:h for q,h in retime['source_code_sha256'].items()},
    'solver':{'version':'.'.join(map(str,gp.gurobi.version())),'threads':2,'limit_seconds':120,'objective':'charging cost at fixed fleet eight',
        'status':m.Status,'status_name':{GRB.OPTIMAL:'OPTIMAL',GRB.TIME_LIMIT:'TIME_LIMIT',GRB.SUBOPTIMAL:'SUBOPTIMAL'}.get(m.Status,str(m.Status)),
        'runtime_seconds':m.Runtime,'optimize_wall_seconds':optimize_wall,'solutions':m.SolCount,
        'objective_nonfleet':m.ObjVal,'bound_nonfleet_finite_pool':m.ObjBound,'mip_gap':m.MIPGap,
        'weighted_objective':800000+m.ObjVal,'weighted_bound_finite_pool':800000+m.ObjBound,
        'nodes':m.NodeCount,'columns':m.NumVars,'rows':m.NumConstrs,'plug_rows':len(plugcols),'trip_equalities':len(tripcols),
        'new_columns_selected':sum(i>=39390 for i in ids)},
    'original_columns':39390,'new_columns':8,'physics':summary['physics'],'actual_master_trip_rows':'partition (=1)',
    'fixed_fleet':8,'terminal_target_kwh':0,'plugs':DEFAULT_PLUGS,
    'initial_replay':{k:initial[k] for k in ['validated','fleet','trip_count','exact_once','duplicate_service','grid_plug_overload_minutes','objective_nonfleet','continuous_nonfleet','grid_terminal_kwh','peak_simultaneous_trips']},
    'final_replay':{k:final[k] for k in ['validated','fleet','trip_count','exact_once','duplicate_service','grid_plug_overload_minutes','objective_nonfleet','continuous_nonfleet','grid_terminal_kwh','peak_simultaneous_trips']},
    'timing_seconds':{'initial_independent_replay':initial_seconds,'pool_load_and_incidence':load_seconds,'model_build':build_seconds,'optimize_wall':optimize_wall,'final_independent_replay':final_seconds,'driver_total':time.perf_counter()-T0},
    'upstream_incremental_seconds':{'archive_edge_matching':link['compute_seconds'],'fixed_sequence_retimer':retime['elapsed_seconds']},
    'historical_ancestor_compute':{'graph_seconds':summary['graph_seconds'],'root_cg_seconds':summary['cg_seconds'],
        'root_lp_seconds_component':summary['lp_seconds'],'root_pricing_seconds_component':summary['pricing_seconds'],
        'dive_requested_seconds':summary['dive_settings']['seconds'],'dive_actual_seconds':None,
        'dive_state':summary['dive']['state'],'fleet_mip_seconds':summary['fleet_stage']['seconds'],
        'charging_mip_actual_seconds':None,'historical_selected_fleet':summary['fleet_stage']['buses'],
        'historical_selected_duplicate_service':44,'warning':'Missing native dive and charging elapsed times; do not report an equal-end-to-end speed ratio.'},
    'scope':'One deterministic archive-edge recombination pilot. Existing 39,390-route archive and its ancestor graph/CG/dive cost are reused. Fleet minimum eight follows from independent simultaneous-trip floor plus replayed selection. Charging bound is finite augmented-pool only. Not blind from-scratch or a new pricing certificate.'}
aggregate['timing_seconds']['incremental_workflow_total_excluding_driver_startup_gap']=aggregate['timing_seconds']['driver_total']+link['compute_seconds']+retime['elapsed_seconds']
write(a.out/'aggregate.json',aggregate)
print(json.dumps({'status':aggregate['solver']['status_name'],'runtime':m.Runtime,'fleet':len(ids),'cost':m.ObjVal,'bound':m.ObjBound,'validated':final['validated'],'driver_seconds':aggregate['timing_seconds']['driver_total']}))
