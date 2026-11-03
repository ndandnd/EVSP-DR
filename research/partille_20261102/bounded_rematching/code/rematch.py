"""Exploratory observed-edge rematching; fixed whole-path no-goods and bounded search."""
import argparse,collections,dataclasses,hashlib,itertools,json,os,pathlib,signal,subprocess,sys,time
P=pathlib.Path
LIMIT=180
CAP=32

def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def load(p):return json.loads(P(p).read_text())
def save(p,x):
 p=P(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def no_good_coefficients(path,edges):
 """Forbid this exact path component,including both endpoints;allow extensions."""
 coefficients=collections.Counter({e:1 for e in zip(path,path[1:])})
 for e in edges:
  if e[1]==path[0]:coefficients[e]-=1
  if e[0]==path[-1]:coefficients[e]-=1
 return {e:c for e,c in coefficients.items() if c},len(path)-2

def paths_from_matching(starts,selected):
 successor=dict(selected);predecessor={v:u for u,v in selected};paths=[]
 assert len(successor)==len(predecessor)==len(selected)
 for u in sorted(set(range(len(starts)))-set(predecessor),key=lambda u:(starts[u],u)):
  path=[u]
  while path[-1] in successor:path.append(successor[path[-1]])
  paths.append(path)
 assert sorted(t for path in paths for t in path)==list(range(len(starts)))
 return paths

def worker(manifest,out):
 start=time.monotonic();m=load(manifest);r={'schema':'bounded-observed-edge-rematch-v1','case':m['case'],'exploratory':True,'validated':False,'case_wall_limit_seconds':LIMIT,'candidate_cap':CAP,'candidates_attempted':0,'safe_whole_path_nogoods':0,'heuristic_visited_matching_exclusions':0,'individual_retime_calls':0,'individual_cache_hits':0,'joint_retime_calls':0,'single_unreachable_proofs':[],'attempts':[],'phase_seconds':{},'source_hashes':{'manifest':sha(manifest),'driver':sha(__file__),'engine':sha(P(__file__).with_name('engine.py'))},'ranking':'minimize sum of observed-edge frequency-descending rank;ties by source start minute,source id,successor id;Gurobi seed0,threads1','nogoood_scope':'exact fixed-sequence event-graph path with source/sink boundaries only;no continuous-infeasibility claim','no_giro_assignment_inputs':True,'new_pricing_certificate':False}
 def elapsed():return time.monotonic()-start
 def checkpoint():
  r['elapsed_seconds']=elapsed();save(out/'receipt.json',r)
 checkpoint()
 import engine
 phase='authentication';tick=time.monotonic()
 summary,solver,data,inputs=engine.authenticate(m)
 r['phase_seconds'][phase]=time.monotonic()-tick;r['upstream']={k:summary.get(k) for k in ['graph_seconds','cg_seconds','lp_seconds','pricing_seconds','dive_settings','dive','fleet_stage','charging_stage']};r['upstream_note']='Saved-root graph/CG costs and prior held-out screening are ancestors,not included in180s;no cold/equal-end-to-end speed claim.';r['exploratory_ancestor_durations']=m['exploratory_ancestor_durations'];r['root_pricing_certified']=summary['cg_pricing_certified'];r['root_columns']=m['root_columns'];r['root_prefix_sha256']=m['root_prefix_sha256'];r['solver_commit']=m['solver_commit'];r['input_sha256']=m['input_sha256'];checkpoint()
 phase='connections';tick=time.monotonic();rows,starts,ends,arcs,sequences=engine.connect(m['pool'],m['root_columns'],m['root_prefix_sha256'],inputs['instance']);basepaths,certificate=engine.match(starts,arcs);r['phase_seconds'][phase]=time.monotonic()-tick;r['observed_edges']=len(arcs);r['trip_count']=len(rows);r['observed_graph_minimum_paths']=len(basepaths);checkpoint()
 if len(basepaths)>8:r['outcome']='observed_links_insufficient_for_eight';checkpoint();return
 target=len(starts)-8
 sys.path.insert(0,str(solver/'src'))
 import gurobipy as gp
 from gurobipy import GRB
 edges=sorted(arcs,key=lambda e:(-arcs[e],starts[e[0]],e[0],e[1]));incoming=collections.defaultdict(list);outgoing=collections.defaultdict(list)
 for e in edges:outgoing[e[0]].append(e);incoming[e[1]].append(e)
 model=gp.Model('bounded_blind_edge_rematch');model.Params.LogFile=str(out/'matching.private.log');model.Params.Threads=1;model.Params.Seed=0;model.Params.MIPGap=0
 y=model.addVars(edges,vtype=GRB.BINARY,obj={e:i+1 for i,e in enumerate(edges)},name='adjacency')
 for u in range(len(starts)):
  model.addConstr(gp.quicksum(y[e] for e in incoming[u])<=1)
  model.addConstr(gp.quicksum(y[e] for e in outgoing[u])<=1)
 model.addConstr(y.sum()==target)
 cache={};path_cuts=set()
 for attempt in range(CAP):
  if elapsed()>=LIMIT-0.5:r['outcome']='case_budget_exhausted';break
  model.Params.TimeLimit=max(0.1,LIMIT-elapsed()-0.5);tick=time.monotonic();model.optimize();solve_wall=time.monotonic()-tick
  r['phase_seconds']['matching_mip']=r['phase_seconds'].get('matching_mip',0)+solve_wall
  if not model.SolCount:
   r['outcome']='no_further_observed_matching' if model.Status==GRB.INFEASIBLE else 'matching_solver_limit_no_incumbent';r['matching_last_status']=model.Status;break
  selected=[e for e in edges if y[e].X>0.5];assert len(selected)==target;paths=paths_from_matching(starts,selected);assert len(paths)==8
  dest=out/f'candidate_{attempt:02d}';dest.mkdir();save(dest/'paths.private.json',paths)
  entry={'candidate':attempt,'rank_objective':model.ObjVal,'ranking_status':model.Status,'matching_seconds':solve_wall,'paths':8,'individual_checks':[]};r['attempts'].append(entry);r['candidates_attempted']+=1;checkpoint()
  bad=None
  for ordinal,path in enumerate(paths):
   key=tuple(path)
   if key in cache:r['individual_cache_hits']+=1;reachable=cache[key]
   else:
    r['individual_retime_calls']+=1;single=dest/f'single_{ordinal:02d}';single.mkdir();rr={'retiming_steps':[]};tick=time.monotonic()
    try:
     problem,one=engine.retime([path],solver,data,inputs,single,rr)
     cert={'edges':[[u,v,1] for u,v in zip(path,path[1:])],'matching':[list(e) for e in zip(path,path[1:])],'min_vertex_cover_left':path[:-1],'min_vertex_cover_right':[]}
     proof=engine.independent_replay(dataclasses.replace(problem,trips=tuple(path)),one,[path],cert,inputs);assert proof['validated'];save(single/'replay.private.json',proof);reachable=True
    except engine.CheckFailure as exc:
     if str(exc)!='fixed_sequence_unreachable':raise
     reachable=False;save(single/'unreachable.private.json',{'fixed_path':path,'scope':'No terminal candidate in registered fixed-sequence event graph with no other-bus reservations','engine_sha256':r['source_hashes']['engine']})
    r['phase_seconds']['individual_retime_and_replay']=r['phase_seconds'].get('individual_retime_and_replay',0)+time.monotonic()-tick;cache[key]=reachable
   entry['individual_checks'].append({'ordinal':ordinal,'reachable':reachable})
   if not reachable:bad=key;break
  if bad is not None:
   assert bad not in path_cuts
   coeff,rhs=no_good_coefficients(bad,edges);model.addConstr(gp.quicksum(c*y[e] for e,c in coeff.items())<=rhs);path_cuts.add(bad);r['safe_whole_path_nogoods']+=1;r['single_unreachable_proofs'].append({'candidate':attempt,'path_ordinal':ordinal,'path_length':len(bad),'path_sha256':hashlib.sha256(json.dumps(list(bad)).encode()).hexdigest()});entry['outcome']='excluded_exact_unreachable_path';checkpoint();continue
  joint=dest/'joint';joint.mkdir();rr={'retiming_steps':[]};r['joint_retime_calls']+=1;tick=time.monotonic()
  try:
   problem,routes=engine.retime(paths,solver,data,inputs,joint,rr)
   cert={**certificate,'matching':[list(e) for e in selected]};proof=engine.independent_replay(problem,routes,paths,cert,inputs);assert proof['validated'] and proof['fleet']==8 and proof['exact_once'];save(joint/'replay.private.json',proof)
   r.update({k:proof[k] for k in ['validated','fleet','trip_count','exact_once','grid_plug_overload_minutes','grid_terminal_kwh','grid_nonfleet_cost','continuous_nonfleet_cost']});r['selected_sha256']=sha(joint/'routes.json');r['outcome']='validated_exploratory_candidate';entry['outcome']=r['outcome'];r['phase_seconds']['joint_retime_and_replay']=r['phase_seconds'].get('joint_retime_and_replay',0)+time.monotonic()-tick;checkpoint();break
  except engine.CheckFailure as exc:
   if str(exc) not in {'fixed_sequence_unreachable','first_order_capacity_conflict'}:raise
   # Every path already passed a no-reservations single test. Do not claim path infeasibility.
   entry['outcome']='reservation_guided_retiming_miss';entry['shared_retiming_reason']=str(exc);entry['not_an_infeasibility_proof']=True
   model.addConstr(gp.quicksum(y[e] for e in selected)<=target-1);r['heuristic_visited_matching_exclusions']+=1
  r['phase_seconds']['joint_retime_and_replay']=r['phase_seconds'].get('joint_retime_and_replay',0)+time.monotonic()-tick;checkpoint()
 else:r['outcome']='candidate_cap_exhausted'
 r.setdefault('outcome','case_budget_exhausted');checkpoint()

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=P,required=True);p.add_argument('--output',type=P,required=True);p.add_argument('--start-fee',type=float,choices=[5],required=True);p.add_argument('--worker',action='store_true');a=p.parse_args()
 if a.worker:
  try:worker(a.manifest,a.output)
  except Exception as exc:
   receipt=a.output/'receipt.json';r=load(receipt) if receipt.exists() else {'validated':False};r.update(outcome='technical_or_validation_error',error_class=type(exc).__name__);save(receipt,r);save(a.output/'error.private.json',{'message':str(exc)});raise
  return
 a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 argv=[sys.executable,'-u',__file__,'--manifest',str(a.manifest),'--output',str(a.output),'--start-fee','5','--worker']
 with (a.output/'worker_stdout.private.log').open('w') as log:
  child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=child.wait(timeout=LIMIT);timedout=False
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait();timedout=True
 rp=a.output/'receipt.json';r=load(rp) if rp.exists() else {'validated':False,'case':load(a.manifest)['case']}
 r.update(parent_wall_seconds=time.monotonic()-start,parent_timeout=timedout,child_exit_code=code,search_budget_seconds=LIMIT,candidate_cap=CAP)
 if timedout and not r.get('validated'):r['outcome']='case_wall_timeout'
 r['interpretation']='Exploratory saved-root repair only. Safe path cuts retain endpoint extensions. Shared misses are heuristic enumeration,not infeasibility proofs. No cold-speed,full-model bound or GIRO-assignment claim.';save(rp,r)
 print(json.dumps({k:r.get(k) for k in ['case','outcome','validated','candidates_attempted','safe_whole_path_nogoods','heuristic_visited_matching_exclusions','parent_wall_seconds']}))
 raise SystemExit(0 if r.get('validated') else 3)
if __name__=='__main__':main()
