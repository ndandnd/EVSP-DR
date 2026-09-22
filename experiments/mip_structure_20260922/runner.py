"""Frozen same-pool diagnostics and fleet-only parameter trials; no CG."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time,math,re,gzip,socket,subprocess
from core import digest,structure,dual_certificate,validate_start


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,indent=2);f.write('\n')


def native(manifest):
    p=Path(manifest['pinned_code']);assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip()==manifest['native_execution_commit'];assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=p,text=True).strip();sys.path.insert(0,str(p/'src'))
    import run_exact_pool_mip as mod
    assert sha(p/'src/run_exact_pool_mip.py')==manifest['native_mip_source_sha256']
    return mod


def route_hash(r):
    return digest({k:r.get(k) for k in ['trips','route_nodes','charging_stops','cost']})


def model(columns,m,costs,cap,log,threads=8):
    import gurobipy as gp
    from gurobipy import GRB
    mdl=gp.Model('frozen_trip_route_incidence');mdl.Params.LogFile=str(log);mdl.Params.Threads=threads;mdl.Params.Seed=0;mdl.Params.MIPGap=0.0001
    x=mdl.addVars(len(columns),lb=0,ub=1,vtype=GRB.CONTINUOUS,name='a');rows=[[] for _ in range(m)]
    for j,col in enumerate(columns):
        for t in col:rows[t].append(j)
    cover=[mdl.addConstr(gp.quicksum(x[j] for j in row)>=1,name=f'cov_{i}') for i,row in enumerate(rows)]
    fleet=mdl.addConstr(gp.quicksum(x.values())<=cap,name='fleet_cap') if cap is not None else None
    mdl.setObjective(gp.quicksum(costs[j]*x[j] for j in range(len(columns))),GRB.MINIMIZE);mdl.update()
    return mdl,x,cover,fleet


def native_smoke(out):
    import gurobipy as gp
    from gurobipy import GRB
    assert gp.gurobi.version()==(12,0,3)
    cols=[[0],[1],[0,1]];records=[]
    for label,costs,cap in [('fleet',[1.,1.,1.],None),('charging',[3.,4.,2.],1)]:
        mdl,x,rows,fleet=model(cols,2,costs,cap,out/f'smoke_{label}.gurobi.log');mdl.Params.TimeLimit=10;mdl.optimize()
        assert mdl.Status==GRB.OPTIMAL
        certificate=dual_certificate(cols,costs,[r.Pi for r in rows],fleet.Pi if fleet is not None else 0,cap,[2])
        assert certificate['lower_bound_integer']<=certificate['incumbent_upper_bound_integer']
        assert abs(mdl.ObjVal-(1 if label=='fleet' else 2))<1e-7
        records.append({'objective':label,'lp':mdl.ObjVal,'lower':certificate['lower_bound'],'upper':certificate['incumbent_upper_bound']});mdl.dispose()
    write(out/'native_smoke.json',{'passed':True,'gurobi_version':'.'.join(map(str,gp.gurobi.version())),'checks':records})


def prepare(root,manifest,case,out):
    import numpy as np
    import gurobipy as gp
    started=time.monotonic();native_smoke(out);mod=native(manifest);pins=case['pins']
    for p,h in pins.items():assert sha(p)==h,p
    endpoint=json.loads(Path(case['endpoint']).read_text());status,routes,trips=mod.load_pool(Path(endpoint['source_result']),deduplicate=True)
    assert len(routes)==case['columns'];assert len(trips)==case['rows']
    routes,audit=mod.prepare_strict_partition_pool(status,routes,data_dir=manifest['data_dir'],reference_data_dir=manifest['data_dir']);routes=mod.deduplicate_pool(routes)
    ordered=mod.ordered_pool_sha256(routes)
    assert ordered==endpoint['physical_pool_audit']['mip_ordered_pool_sha256']
    assert len(routes)==case['columns'] and audit['rejected_columns']==audit['deterministically_repaired']==0
    mapping={t:i for i,t in enumerate(trips)};columns=[[mapping[t] for t in r['trips']] for r in routes];costs=[float(r['cost'])-100000.0 for r in routes];hashes=[route_hash(r) for r in routes]
    assert len(set(hashes))==len(hashes);assert sum(map(len,columns))==case['nonzeros'];assert all(math.isfinite(c) for c in costs)
    singleton=mod.singleton_partition_indices(routes,trips);greedy=mod.greedy_partition_start_indices(routes,trips,singleton)
    assert len(greedy)==endpoint['mip_start']['validated_bus_count'];validate_start(columns,len(trips),greedy)
    strong=endpoint['two_stage']['stage1_selected_route_indices'];strong_reason=None
    try:
        validate_start(columns,len(trips),strong)
        assert len(strong)==endpoint['two_stage']['stage1_buses'];assert abs(sum(routes[j]['cost'] for j in strong)-endpoint['two_stage']['stage1_total_objective'])<1e-5
    except (AssertionError,ValueError,IndexError) as exc:
        strong_reason='Saved stage-one incumbent failed exact frozen-pool identity/coverage/objective check: '+str(exc);strong=None
    by_hash={h:j for j,h in enumerate(hashes)};charging=[by_hash[h] for h in endpoint['selected_route_hashes']];validate_start(columns,len(trips),charging,endpoint['two_stage']['stage1_buses'])
    assert abs(sum(costs[j] for j in charging)-endpoint['two_stage']['stage2_variable_obj'])<1e-5
    identity={'trips':trips,'columns':columns,'costs':costs,'route_hashes':hashes};matrix_hash=digest(identity)
    offsets=[0]
    for col in columns:offsets.append(offsets[-1]+len(col))
    np.savez_compressed(out/'matrix.npz',indptr=np.asarray(offsets,dtype=np.int64),indices=np.asarray([t for col in columns for t in col],dtype=np.int32),costs=np.asarray(costs,dtype=np.float64))
    meta={'case':case['id'],'trips':trips,'route_hashes':hashes,'ordered_native_pool_sha256':ordered,'matrix_identity_sha256':matrix_hash,'matrix_file_sha256':sha(out/'matrix.npz'),'rows':len(trips),'columns':len(columns),'nonzeros':sum(map(len,columns)),'greedy_start':greedy,'strong_start':strong,'strong_start_available':strong is not None,'strong_start_blocked_reason':strong_reason,'charging_incumbent':charging,'strong_start_scope':'Offline saved stage-one incumbent from this exact ordered pool; original acquisition time excluded from new trial, not a timed algorithm','strong_start_source':case['endpoint'],'strong_start_source_sha256':pins[case['endpoint']],'physical_pool_audit':audit,'preparation_s':time.monotonic()-started,'host':socket.gethostname(),'pins':pins,'signed_costs':{'negative_count':sum(c<0 for c in costs),'min':min(costs),'max':max(costs)},'gurobi_version':'.'.join(map(str,gp.gurobi.version()))}
    write(out/'matrix.json',meta)
    # No matrix change from the diagnostic findings is applied to the 25 trials.
    diagnostic=structure(columns,costs,len(trips),max_seconds=600);write(out/'structure.json',diagnostic)
    screens={}
    for mode,objective,cap,witness in [('fleet',[1.0]*len(columns),None,strong if strong is not None else charging),('charging',costs,endpoint['two_stage']['stage1_buses'],charging)]:
        mdl,x,cover,fleet=model(columns,len(trips),objective,cap,out/f'{mode}_lp.gurobi.log');mdl.Params.TimeLimit=300;t0=time.monotonic();mdl.optimize();entry={'status':int(mdl.Status),'wall_s':time.monotonic()-t0,'objective':'unit fleet' if mode=='fleet' else 'expanded-grid charging-related cost','fleet_cap':cap,'matrix_identity_sha256':matrix_hash,'solver_objective':float(mdl.ObjVal) if mdl.SolCount else None}
        try:
            cert=dual_certificate(columns,objective,[r.Pi for r in cover],fleet.Pi if fleet is not None else 0,cap,witness)
            with gzip.open(out/f'{mode}_dual_certificate.json.gz','wt') as f:json.dump(cert,f,separators=(',',':'))
            entry.update({k:cert[k] for k in ['lower_bound','incumbent_upper_bound','removable_fraction']});entry.update(fix_zero_count=len(cert['safe_fix_zero_indices']),fix_one_count=len(cert['safe_fix_one_indices']),certificate_sha256=sha(out/f'{mode}_dual_certificate.json.gz'))
        except (AttributeError,gp.GurobiError) as exc:entry['screening_unavailable']=str(exc)
        screens[mode]=entry;mdl.dispose()
    write(out/'screening.json',screens)
    for p,h in pins.items():assert sha(p)==h,p
    write(out/'COMPLETE.json',{'case':case['id'],'matrix_file':str(out/'matrix.npz'),'matrix_metadata':str(out/'matrix.json'),'matrix_metadata_sha256':sha(out/'matrix.json'),'matrix_file_sha256':sha(out/'matrix.npz'),'wall_s':time.monotonic()-started,'scientific_model_unchanged':True})
    write(root/'prepared'/f"{case['id']}.json",{'complete':str(out/'COMPLETE.json'),'sha256':sha(out/'COMPLETE.json')})


def trial(root,manifest,case,arm,out):
    import numpy as np
    from gurobipy import GRB
    import gurobipy as gp
    loaded=time.monotonic();assert '.'.join(map(str,gp.gurobi.version()))==manifest['gurobi_version']
    gate=json.loads((root/'prepared'/f"{case['id']}.json").read_text());assert sha(gate['complete'])==gate['sha256'];complete=json.loads(Path(gate['complete']).read_text());assert sha(complete['matrix_file'])==complete['matrix_file_sha256'];assert sha(complete['matrix_metadata'])==complete['matrix_metadata_sha256'];meta=json.loads(Path(complete['matrix_metadata']).read_text());z=np.load(complete['matrix_file'],allow_pickle=False);columns=[z['indices'][int(a):int(b)].tolist() for a,b in zip(z['indptr'][:-1],z['indptr'][1:])];costs=z['costs'].tolist();assert digest({'trips':meta['trips'],'columns':columns,'costs':costs,'route_hashes':meta['route_hashes']})==meta['matrix_identity_sha256']
    if arm=='strong_start' and not meta['strong_start_available']:
        write(out/'result.json',{'case':case['id'],'arm':arm,'status':'BLOCKED','reason':meta['strong_start_blocked_reason'],'optimizer_run':False});return
    start=meta['strong_start'] if arm=='strong_start' else meta['greedy_start'];validate_start(columns,meta['rows'],start)
    artifact_loading_s=time.monotonic()-loaded;build_started=time.monotonic()
    mdl,x,cover,cap=model(columns,meta['rows'],[1.0]*len(columns),None,out/'gurobi.log')
    start_set=set(start)
    for j in range(len(columns)):x[j].VType=GRB.BINARY;x[j].Start=1 if j in start_set else 0
    if arm=='focus1':mdl.Params.MIPFocus=1
    elif arm=='focus2':mdl.Params.MIPFocus=2
    elif arm=='presparsify1':mdl.Params.PreSparsify=1
    elif arm not in ['default','strong_start']:raise ValueError(arm)
    mdl.Params.TimeLimit=1800;mdl.update();assert (mdl.NumConstrs,mdl.NumVars,mdl.NumNZs)==(case['rows'],case['columns'],case['nonzeros'])
    build_wall_s=time.monotonic()-build_started;dimensions={'rows':mdl.NumConstrs,'variables':mdl.NumVars,'nonzeros':mdl.NumNZs};parameters={k:getattr(mdl.Params,k) for k in ['Threads','Seed','TimeLimit','MIPGap','MIPFocus','PreSparsify','Presolve','Method']}
    events=[]
    def callback(model,where):
        if where==GRB.Callback.MIPSOL:
            events.append({'runtime_s':model.cbGet(GRB.Callback.RUNTIME),'fleet':model.cbGet(GRB.Callback.MIPSOL_OBJ),'node_count':model.cbGet(GRB.Callback.MIPSOL_NODCNT)})
    t0=time.monotonic();mdl.optimize(callback);wall=time.monotonic()-t0;selected=[j for j in range(len(columns)) if mdl.SolCount and x[j].X>.5]
    if selected:validate_start(columns,meta['rows'],selected)
    bound=float(mdl.ObjBound);fleet=len(selected) if selected else None;proved=fleet is not None and math.isfinite(bound) and math.ceil(bound-1e-7)>=fleet
    result={'case':case['id'],'arm':arm,'model':'same-pool binary covering fleet-only','gurobi_version':'.'.join(map(str,gp.gurobi.version())),'model_dimensions':dimensions,'effective_parameters':parameters,'artifact_loading_s':artifact_loading_s,'model_build_wall_s':build_wall_s,'matrix_identity_sha256':meta['matrix_identity_sha256'],'prepared_metadata_sha256':complete['matrix_metadata_sha256'],'original_endpoint':case['endpoint'],'threads':8,'seed':0,'time_limit_s':1800,'actual_optimize_wall_s':wall,'gurobi_runtime_s':mdl.Runtime,'gurobi_status':int(mdl.Status),'fleet':fleet,'bound':bound,'finite_pool_fleet_proven':proved,'target':case['target'],'target_attained':fleet==case['target'],'selected_indices':selected,'selected_route_hashes':[meta['route_hashes'][j] for j in selected],'row_coverage_validated':bool(selected),'shared_capacity_enforced':False,'start_indices':start,'start_buses':len(start),'start_scope':meta['strong_start_scope'] if arm=='strong_start' else 'Same deterministic greedy pool partition as original saved run','events':events,'first_target_time_s':next((e['runtime_s'] for e in events if e['fleet']<=case['target']+1e-6),None),'host':socket.gethostname(),'log_sha256':None}
    mdl.dispose();log=(out/'gurobi.log').read_text();result['log_sha256']=sha(out/'gurobi.log');result['start_acceptance_messages']=[s for s in log.splitlines() if 'MIP start' in s];write(out/'result.json',result);write(out/'COMPLETE.json',{'result_sha256':sha(out/'result.json'),'status':'finished'})


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('mode',choices=['prepare','trial']);p.add_argument('case');p.add_argument('--arm',default='default');p.add_argument('--manifest',type=Path);args=p.parse_args();manifest_path=args.manifest or args.root/'manifest.json';manifest=json.loads(manifest_path.read_text());assert sha(__file__)==manifest['runner_sha256'];assert sha(Path(__file__).with_name('core.py'))==manifest['core_sha256'];case=next(c for c in manifest['cases'] if c['id']==args.case);attempt=os.environ.get('SLURM_JOB_ID','local')+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');out=args.root/'results'/args.case/(args.mode if args.mode=='prepare' else args.arm)/attempt;out.mkdir(parents=True,exist_ok=False)
    write(out/'execution.json',{'manifest_path':str(manifest_path),'manifest_sha256':sha(manifest_path),'script_sha256':sha(__file__),'core_sha256':sha(Path(__file__).with_name('core.py')),'code_commit':manifest['code_commit'],'argv':sys.argv,'job_id':os.environ.get('SLURM_JOB_ID'),'restart':os.environ.get('SLURM_RESTART_COUNT','0'),'host':socket.gethostname(),'started_unix':time.time()})
    if args.mode=='prepare':prepare(args.root,manifest,case,out)
    else:trial(args.root,manifest,case,args.arm,out)
if __name__=='__main__':main()
