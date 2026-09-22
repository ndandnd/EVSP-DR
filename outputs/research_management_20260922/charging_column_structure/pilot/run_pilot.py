#!/usr/bin/env python3
"""Standalone saved-pool formulation pilot. --validate-only never imports a solver."""
from pathlib import Path
import argparse,collections,csv,hashlib,json,math,os,platform,re,resource,time
EXPECTED='714cec263633e3fa8ac178052b3be595097c1990769503afcac86e3d5d11a845'
COUNTS={'2190L':1,'4808':1,'3127L':2,'7880C':1,'JON_A':1}
VARIANTS=('original_minute','identical_rows','endpoint_difference')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,payload):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,allow_nan=False));tmp.replace(path)
def number(x):
 try:
  v=float(x);return v if math.isfinite(v) else None
 except:return None
def load(pool):
 assert sha(pool)==EXPECTED,'Source pool hash mismatch'
 rr=[json.loads(s) for s in Path(pool).read_text().splitlines()];trips=sorted({t for r in rr for t in r['trips']});grid=collections.defaultdict(set)
 for j,r in enumerate(rr):
  assert len(r['trips'])==len(set(r['trips']))
  for a in r['actions']:
   if a.get('kind')!='charge' or a['station'] not in COUNTS:continue
   start,end=float(a['setup_start_min']),float(a['connection_end_min'])
   assert end>=start
   for k in range(math.floor(start),math.ceil(end)):
    if k+1e-9<end and k+1>start+1e-9:grid[a['station'],k].add(j)
 rows=[(s,k,tuple(sorted(v))) for (s,k),v in sorted(grid.items())];seen=set();unique=[]
 for s,k,support in rows:
  key=s,COUNTS[s],support
  if key not in seen:unique.append((s,k,support));seen.add(key)
 diff=[]
 for s in sorted({s for s,k in grid}):
  first=min(k for st,k in grid if st==s);last=max(k for st,k in grid if st==s);previous=set()
  for k in range(first,last+2):
   now=grid.get((s,k),set())
   if now!=previous:diff.append((s,k,tuple(sorted(now-previous)),tuple(sorted(previous-now))))
   previous=now
 # Strong exact column/coefficient reconstruction across EVERY original minute row.
 for s in COUNTS:
  cur=set();events=[d for d in diff if d[0]==s];ei=0
  for k in range(min((q[1] for q in events),default=0),max((q[1] for q in events),default=-1)+1):
   while ei<len(events) and events[ei][1]==k:
    _,_,plus,minus=events[ei];assert not(cur&set(plus)) and set(minus)<=cur;cur.difference_update(minus);cur.update(plus);ei+=1
   assert cur==grid.get((s,k),set())
  assert not cur
 lookup={(s,COUNTS[s],v) for s,k,v in unique};assert all((s,COUNTS[s],v) in lookup for s,k,v in rows)
 A={t:tuple(j for j,r in enumerate(rr) if t in r['trips']) for t in trips}
 data=dict(routes=rr,trips=trips,A=A,grid=grid,rows=rows,unique=unique,diff=diff)
 annz=sum(map(len,A.values()));station_events=collections.Counter(s for s,k,p,m in diff)
 dims=dict(original_minute=dict(rows=len(trips)+len(rows),columns=len(rr),binary=len(rr),nonzeros=annz+sum(len(v) for s,k,v in rows)),identical_rows=dict(rows=len(trips)+len(unique),columns=len(rr),binary=len(rr),nonzeros=annz+sum(len(v) for s,k,v in unique)),endpoint_difference=dict(rows=len(trips)+len(diff),columns=len(rr)+len(diff),binary=len(rr),nonzeros=annz+sum(len(p)+len(m) for s,k,p,m in diff)+sum(2*n-1 for n in station_events.values())))
 assert dims['original_minute']==dict(rows=1789,columns=321,binary=321,nonzeros=44844)
 assert dims['identical_rows']==dict(rows=303,columns=321,binary=321,nonzeros=10232)
 assert dims['endpoint_difference']==dict(rows=307,columns=593,binary=321,nonzeros=7714)
 return data,dims

def replay_selection(data,values,tol=1e-6):
 coverage={str(t):sum(values[j] for j in support) for t,support in data['A'].items()};occupancy=[(s,k,sum(values[j] for j in support)) for s,k,support in data['rows']]
 violations=[dict(kind='trip',trip=t,value=v) for t,v in coverage.items() if v<1-tol]+[dict(kind='capacity',station=s,minute=k,value=v,limit=COUNTS[s]) for s,k,v in occupancy if v>COUNTS[s]+tol]
 return dict(valid=not violations,violations=violations,max_occupancy={s:max((v for st,k,v in occupancy if st==s),default=0) for s in COUNTS},covered_trip_count=sum(v>=1-tol for v in coverage.values()),scope='Original saved trip-incidence and one-minute charger capacity rows; no new route-physics certification')

def extract_log(path):
 text=Path(path).read_text() if Path(path).exists() else '';lines=text.splitlines();out={}
 for key,pattern in [('presolve_time_s',r'Presolve time: ([\d.]+)s'),('root_relaxation',r'Root relaxation: objective ([\deE+.-]+), (\d+) iterations, ([\d.]+) seconds'),('presolved_dimensions',r'Presolved: (\d+) rows, (\d+) columns, (\d+) nonzeros')]:
  matches=[dict(line=i+1,groups=list(m.groups()),text=l) for i,l in enumerate(lines) if (m:=re.search(pattern,l))];out[key]=matches
 return out

def build(gp,variant,data,log,args):
 begin=time.perf_counter();m=gp.Model('charging_interval_'+variant);m.Params.OutputFlag=1;m.Params.LogFile=str(log);m.Params.Threads=args.threads;m.Params.Seed=args.seed;m.Params.TimeLimit=args.mip_seconds;m.Params.MIPGap=0.0001
 x=m.addVars(len(data['routes']),vtype=gp.GRB.BINARY,lb=0,ub=1,name='x')
 for t,support in data['A'].items():m.addConstr(gp.quicksum(x[j] for j in support)>=1,name=f'trip_{t}')
 if variant!='endpoint_difference':
  for s,k,support in data['rows' if variant=='original_minute' else 'unique']:m.addConstr(gp.quicksum(x[j] for j in support)<=COUNTS[s],name=f'capacity_{s}_{k}')
 else:
  previous={}
  for s,k,plus,minus in data['diff']:
   u=m.addVar(lb=0,ub=COUNTS[s],vtype=gp.GRB.CONTINUOUS,name=f'occupancy_{s}_{k}')
   delta=gp.quicksum(x[j] for j in plus)-gp.quicksum(x[j] for j in minus)
   m.addConstr(u-previous.get(s,0)==delta,name=f'balance_{s}_{k}');previous[s]=u
 m.setObjective(gp.quicksum(x.values()),gp.GRB.MINIMIZE);m.update()
 return m,x,time.perf_counter()-begin

def stats(model):
 def safe_attr(name):
  try:return getattr(model,name)
  except Exception:return None
 d={name:number(safe_attr(name)) for name in ['Status','SolCount','Runtime','Work','NodeCount','IterCount','BarIterCount','ObjBound','MIPGap','MemUsed','MaxMemUsed'] if name not in ['ObjBound','MIPGap'] or model.IsMIP}
 d['objective']=number(model.ObjVal) if model.SolCount else None
 return d

def main():
 a=argparse.ArgumentParser();a.add_argument('--pool',type=Path,default=Path(__file__).parent/'inputs/pool.jsonl');a.add_argument('--out',type=Path);a.add_argument('--validate-only',action='store_true');a.add_argument('--execution-commit');a.add_argument('--seed',type=int,default=0);a.add_argument('--threads',type=int,default=4);a.add_argument('--mip-seconds',type=float,default=300);a.add_argument('--lp-seconds',type=float,default=30);args=a.parse_args()
 start=time.perf_counter();data,dims=load(args.pool);preparation=time.perf_counter()-start
 if args.validate_only:
  # Non-solving rational occupancy comparisons, including the all-zero/all-one vectors.
  import random;random.seed(22)
  for _ in range(1000):
   values=[random.randrange(5) for r in data['routes']]
   original=max(sum(values[j] for j in v)-4*COUNTS[s] for s,k,v in data['rows']);compact=max(sum(values[j] for j in v)-4*COUNTS[s] for s,k,v in data['unique']);assert original==compact
  print(json.dumps(dict(valid=True,dimensions=dims,coefficient_projection='exact',rational_vectors_checked=1000,solver_imported=False,preparation_seconds=preparation),indent=2));return
 if args.out is None:a.error('--out is required for optimizer execution')
 if args.execution_commit is None or not re.fullmatch('[0-9a-f]{40}',args.execution_commit):a.error('--execution-commit must pin the deployed runner commit')
 args.out.mkdir(parents=True,exist_ok=False)
 # Solver import happens only after validated, unique attempt directory is reserved.
 import gurobipy as gp
 meta=dict(pool_path=str(args.pool.resolve()),pool_sha256=sha(args.pool),script_sha256=sha(__file__),execution_commit=args.execution_commit,inspected_reference_commit='35770aae2c08e7d5a356cc3b673e67608e5b1036',original_pool_execution_commit=None,worker_sha256=sha(Path(__file__).parent/'worker.sub'),native_model='run_giro_small_cg.py::_master',objective='fleet sum(x)',master_sense='cover',capacity_semantics='saved strict18E1 route setup-to-disconnection any-overlap one-minute occupancy; union per route/site',initialization='No explicit MIP start for any variant',proof_scope='Finite saved pool only; no CG/pricing certificate; no changed charging physics',gurobi_version=list(gp.gurobi.version()),python=platform.python_version(),host=platform.node(),settings=dict(seed=args.seed,threads=args.threads,mip_seconds=args.mip_seconds,lp_seconds=args.lp_seconds,mipgap=.0001),variant_order=list(VARIANTS),dependencies='Immutable copied JSONL only; no upstream scheduler dependency',scheduler={k:os.environ.get(k) for k in ['SLURM_JOB_ID','SLURM_RESTART_COUNT','SLURM_JOB_PARTITION','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE']},preparation_seconds=preparation,expected_dimensions=dims)
 atomic(args.out/'manifest.json',meta);allresults=[]
 for variant in VARIANTS:
  vdir=args.out/variant;vdir.mkdir();whole=time.perf_counter();m,x,build_s=build(gp,variant,data,vdir/'mip.log',args)
  observed=dict(rows=m.NumConstrs,columns=m.NumVars,binary=m.NumBinVars,nonzeros=m.NumNZs);assert observed==dims[variant],(observed,dims[variant]);m.write(str(vdir/'model.mps.gz'))
  lp=m.relax();lp.Params.LogFile=str(vdir/'lp.log');lp.Params.TimeLimit=args.lp_seconds;lp_begin=time.perf_counter();lp.optimize();lp_wall=time.perf_counter()-lp_begin
  lpstats=stats(lp);lpvalues=[lp.getVarByName(f'x[{j}]').X for j in range(len(data['routes']))] if lp.SolCount else None
  lpreplay=replay_selection(data,lpvalues,1e-5) if lpvalues is not None else None;assert lpreplay is None or lpreplay['valid'];lp.dispose()
  events=[]
  def callback(model,where):
   if where==gp.GRB.Callback.MIPSOL:events.append(dict(runtime_s=model.cbGet(gp.GRB.Callback.RUNTIME),objective=number(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)),bound=number(model.cbGet(gp.GRB.Callback.MIPSOL_OBJBND))))
  mip_begin=time.perf_counter();m.optimize(callback);mip_wall=time.perf_counter()-mip_begin;result=stats(m)
  vals=[x[j].X for j in range(len(data['routes']))] if m.SolCount else None;selected=[j for j,v in enumerate(vals or []) if v>.5]
  actual=replay_selection(data,[int(j in set(selected)) for j in range(len(data['routes']))]) if vals is not None else None;assert actual is None or actual['valid'];assert vals is None or all(abs(v-round(v))<1e-5 for v in vals)
  proof=bool(vals is not None and result['ObjBound'] is not None and math.ceil(result['ObjBound']-1e-7)>=len(selected));fingerprint=m.Fingerprint;m.dispose()
  rec=dict(variant=variant,dimensions=observed,model_fingerprint=fingerprint,build_update_wall_s=build_s,lp_optimize_wall_s=lp_wall,lp=lpstats,lp_original_matrix_validation=lpreplay,mip_optimize_wall_s=mip_wall,mip=result,selected_indices=selected,selected_routes=[data['routes'][j] for j in selected],original_matrix_validation=actual,finite_pool_fleet_proven=proof,incumbent_events=events,variant_total_wall_s=time.perf_counter()-whole,process_cumulative_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if platform.system()=='Linux' else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,lp_log_summary=extract_log(vdir/'lp.log'),mip_log_summary=extract_log(vdir/'mip.log'))
  atomic(vdir/'result.json',rec);allresults.append(rec);atomic(args.out/'progress.json',dict(completed=[r['variant'] for r in allresults],latest=variant))
 lpobjs=[r['lp']['objective'] for r in allresults if r['lp']['Status']==gp.GRB.OPTIMAL];assert len(lpobjs)<2 or max(lpobjs)-min(lpobjs)<1e-5
 proven=[r['mip']['objective'] for r in allresults if r['finite_pool_fleet_proven']];assert len(proven)<2 or max(proven)-min(proven)<1e-5
 summary=dict(completed=True,variants=[{k:v for k,v in r.items() if k!='selected_routes'} for r in allresults],total_wall_s=time.perf_counter()-start,comparison_scope='Exact formulations of identical fixed pool; timed-out incumbents may differ; compare proven fleet objectives and LP optima only when available',output_hashes={str(f.relative_to(args.out)):sha(f) for f in args.out.rglob('*') if f.is_file()})
 atomic(args.out/'summary.json',summary);print(json.dumps(dict(completed=True,out=str(args.out),lp_objectives=lpobjs,proven_fleet_objectives=proven,total_wall_s=summary['total_wall_s']),indent=2))
if __name__=='__main__':main()
