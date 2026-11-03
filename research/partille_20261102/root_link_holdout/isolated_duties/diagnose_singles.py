"""Bounded post-outcome diagnosis of frozen fixed passenger paths, without shared reservations.

Each original path is retimed once with the registered engine and first candidate, then
individually replayed. This does not assemble or score a new multi-bus dispatch.
"""
import argparse,dataclasses,hashlib,importlib.util,json,pathlib,signal,sys,time
P=pathlib.Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case-root',type=P,required=True);p.add_argument('--engine',type=P,required=True);p.add_argument('--start-fee',type=float,required=True,choices=[5]);a=p.parse_args()
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def read(p):return json.loads(P(p).read_text())
def dump(p,x):P(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
case=a.case_root;out=case/'single_path_diagnosis';out.mkdir(exist_ok=False)
manifest=read(case/'native_manifest.private.json');local=read(case/'local_inputs.private.json');original=read(case/'original_receipt.json');paths=read(case/'original_paths.json')
assert not original['validated'] and original['minimum_paths']==len(paths)==8
assert sha(case/'original_paths.json')==original['artifact_sha256']['paths.json']
source=P(local['solver_root']);assert {str(p.relative_to(source)):sha(p) for p in (source/'src').rglob('*.py')}==manifest['solver_source_sha256']
inputs={k:P(v) for k,v in local['inputs'].items()}
for k,v in inputs.items():assert sha(v)==manifest['input_sha256'][k]
assert sha(a.engine)==original['runner_sha256']
spec=importlib.util.spec_from_file_location('registered_recombination',a.engine);engine=importlib.util.module_from_spec(spec);sys.modules[spec.name]=engine;spec.loader.exec_module(engine)
result={'schema':'bounded-fixed-path-single-reachability-v1','case':manifest['case'],'case_limit_seconds':180,'start_fee':5,'source_execution_commit':'701b74b5518dc473e6d2a1bdda74e1ad31d891a8','source_solver_commit':manifest['solver_commit'],
 'source_hashes':{'driver':sha(__file__),'registered_engine':sha(a.engine),'native_manifest':sha(case/'native_manifest.private.json'),'original_receipt':sha(case/'original_receipt.json'),'original_paths':sha(case/'original_paths.json')},
 'inputs_sha256':manifest['input_sha256'],'physics':engine.PHYSICS,'plugs':engine.PLUGS,'paths_frozen':8,'per_path_candidate_limit':1,'shared_reservations':'none;fresh occupancy for each original path','alternative_paths_or_orders_tested':False,'dispatch_hits_added':0,'original_failure_class':original['failure_class'],'original_failure_path_ordinal':original['retiming_steps'][-1]['path_ordinal'],'paths':[],'earliest_infeasible_prefix':'not isolated;no extra prefix retimes;terminal reachability and intermediate energy effects not uniquely localized'}
def timeout(*_):raise TimeoutError('180-second case diagnostic cap')
signal.signal(signal.SIGALRM,timeout);start=time.monotonic();signal.alarm(180)
try:
 for i,path in enumerate(paths):
  dest=out/f'path_{i:02d}';dest.mkdir();tick=time.monotonic();rr={'retiming_steps':[]};entry={'path_ordinal':i,'trip_count':len(path),'single_retime_attempts':1}
  try:
   problem,selected=engine.retime([path],source,P(local['data_root']),inputs,dest,rr)
   assert len(selected)==1 and selected[0]['trips']==path
   subproblem=dataclasses.replace(problem,trips=tuple(path));cert={'edges':[[u,v,1] for u,v in zip(path,path[1:])],'matching':[list(x) for x in zip(path,path[1:])],'min_vertex_cover_left':path[:-1],'min_vertex_cover_right':[]}
   proof=engine.independent_replay(subproblem,selected,[path],cert,inputs);assert proof['validated'] and proof['exact_once'] and proof['fleet']==1
   dump(dest/'single_replay.private.json',proof)
   entry.update(outcome='individually_reachable_and_replayed',validated=True,grid_terminal_kwh=proof['grid_terminal_kwh'])
  except TimeoutError:raise
  except Exception as exc:
   entry.update(outcome='single_path_unreachable' if str(exc)=='fixed_sequence_unreachable' else 'technical_or_replay_error',validated=False,error_class=type(exc).__name__)
   dump(dest/'private_error.json',{'type':type(exc).__name__,'message':str(exc)})
  entry.update(elapsed_seconds=time.monotonic()-tick,engine_steps=rr['retiming_steps']);result['paths'].append(entry)
except TimeoutError:
 result['timeout']=True
finally:
 signal.alarm(0)
result['elapsed_seconds']=time.monotonic()-start;result['paths_attempted']=len(result['paths']);result['reachable_singles']=sum(x['validated'] for x in result['paths']);result['unreachable_singles']=sum(x['outcome']=='single_path_unreachable' for x in result['paths']);result['technical_or_replay_errors']=sum(x['outcome']=='technical_or_replay_error' for x in result['paths']);result['timing_scope']='case clock covers single retime attempts and individual replay;source/input authentication precedes it;no archive construction or global optimization'
result['interpretation']='Unreachable singles demonstrate that these first-order misses cannot be explained solely by reservations from other buses.Individually reachable singles do not establish compatibility as a dispatch.The eight-path matching certificate establishes observed adjacency suffices for an eight-path cover only;it does not establish energy reachability.'
dump(out/'aggregate.json',result);print(json.dumps({k:result[k] for k in ['case','paths_attempted','reachable_singles','unreachable_singles','technical_or_replay_errors','elapsed_seconds']}))
