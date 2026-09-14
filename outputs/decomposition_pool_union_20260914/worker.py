#!/usr/bin/env python3
"""Immutable worker for parent-mapped pool construction and finite-pool MIPs."""
from __future__ import annotations
import datetime as dt,fcntl,hashlib,json,os,re,resource,signal,socket,subprocess,sys,time
from pathlib import Path

def now(): return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name('.'+p.name+f'.tmp.{os.getpid()}')
 with t.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def req(p,h):
 got=sha(p)
 if not re.fullmatch('[0-9a-f]{64}',str(h)) or got!=h:raise ValueError(f'hash mismatch {p}: {got} != {h}')
 return got
def safe(s):
 if not re.fullmatch(r'[A-Za-z0-9_.-]+',s) or s in('.','..'):raise ValueError('unsafe id')
def check_checkout(path,commit):
 def git(*a):return subprocess.run(['git','-C',path,*a],text=True,capture_output=True)
 if git('rev-parse','HEAD').stdout.strip()!=commit:raise ValueError('execution commit mismatch')
 if git('status','--porcelain','--untracked-files=no').stdout.strip():raise ValueError('tracked execution checkout dirty')
 if git('symbolic-ref','-q','HEAD').returncode!=1:raise ValueError('execution checkout not detached')
def run(argv,attempt,cwd,limit,env,label='execution'):
 rec={'argv':argv,'cwd':cwd,'host':socket.getfqdn(),'started_utc':now(),'watchdog_s':limit}
 save(attempt/(label+'.json'),rec); start=time.monotonic(); proc=None; timed=False
 def terminate(sig,_):
  if proc:
   try:os.killpg(proc.pid,sig)
   except ProcessLookupError:pass
 old={s:signal.signal(s,terminate) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1)}
 try:
  with (attempt/(label+'.stdout.log')).open('xb') as out,(attempt/(label+'.stderr.log')).open('xb') as err:
   proc=subprocess.Popen(argv,cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True)
   while proc.poll() is None:
    if time.monotonic()-start>=limit:
     timed=True;terminate(signal.SIGTERM,None)
     try:proc.wait(30)
     except subprocess.TimeoutExpired:terminate(signal.SIGKILL,None)
    try:proc.wait(.25)
    except subprocess.TimeoutExpired:pass
  rc=proc.returncode
 finally:
  for s,h in old.items():signal.signal(s,h)
 rec.update(ended_utc=now(),wall_s=time.monotonic()-start,returncode=rc,watchdog=timed,status='finished' if rc==0 and not timed else 'failed')
 save(attempt/(label+'.json'),rec)
 if rec['status']!='finished':raise RuntimeError(f'process failed rc={rc} watchdog={timed}')
 return rec
def publish_link(src,dst):
 t=dst.with_name('.'+dst.name+f'.tmp.{os.getpid()}');t.symlink_to(src.resolve());t.replace(dst)
def register(case,token,out):
 path=Path('/home/nc437/ladder-lite/mip_preemption_study_20260911/registry.json');lock=path.with_suffix('.lock')
 with lock.open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);v=read(path);job=os.environ['SLURM_JOB_ID']
  if (job,str(out)) not in {(str(x['job_id']),x.get('result_path')) for x in v['cases']}:
   v['cases'].append({'job_id':job,'case_id':case['id'],'cohort':'decomposition_pool_union_20260914_'+case['cohort'],'is_validation':bool(case.get('is_validation')),'solver_budget_s':case['solver_budget_s'],'stage1_time_limit_s':case['stage1_s'],'result_path':str(out),'attempt_tag':token,'restart_count':os.environ.get('SLURM_RESTART_COUNT','0'),'requeue':False,'registered_utc':now()});save(path,v)

def source_completion(root,pid,manifest_hash):
 c=read(root/'cases'/pid/'completion.json')
 if c['kind']!='pool_construction' or c['manifest_sha256']!=manifest_hash or not c['usable']:raise ValueError('source completion identity')
 req(c['result_path'],c['result_sha256']);req(c['journal_path'],c['journal_sha256']);req(c['mandatory_routes_path'],c['mandatory_routes_sha256'])
 return c
def execute(root,cid):
 root=Path(root).resolve();safe(cid);m=read(root/'manifest.json');mh=sha(root/'manifest.json');case=m['cases'][cid]
 token=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');safe(token)
 croot=root/'cases'/cid;attempt=croot/'attempts'/token;attempt.mkdir(parents=True,exist_ok=False)
 state={'case_id':cid,'kind':case['kind'],'attempt':token,'manifest_sha256':mh,'started_utc':now(),'status':'preflight'};save(attempt/'state.json',state)
 with (croot/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for name,digest in m['tooling_sha256'].items():req(root/name,digest)
  req(case['parent_input_path'],case['parent_input_sha256'])
  if (croot/'completion.json').exists():
   old=read(croot/'completion.json')
   if old['manifest_sha256']!=mh:raise ValueError('existing completion manifest mismatch')
   req(old['result_path'],old['result_sha256'])
   if old.get('journal_path'):req(old['journal_path'],old['journal_sha256'])
   save(attempt/'state.json',{**state,'status':'already_complete','prior_completion':old,'ended_utc':now()});return
  try:
   if case['kind']=='pool_construction':
    spec=attempt/'spec.json';save(spec,case);out=attempt/'pool.json'
    state['status']='constructing';save(attempt/'state.json',state)
    ex=run([sys.executable,str(root/'pool_logic.py'),'construct','--spec',str(spec),'--out',str(out)],attempt,str(root),case['watchdog_s'],os.environ.copy(),'construction_execution')
    v=read(out);j=Path(v['columns_journal']);mr=attempt/'mandatory_routes.json'
    if v['artifact_kind']!='parent_mapped_partition_pool' or v['optimization_run'] is not False or v['certified_rc_optimal'] is not False or v['pool_construction']['parent_graph_constructed'] is not False:raise ValueError('construction proof scope')
    cp=attempt/'construction.json';final={**state,'status':'finished','usable':True,'optimization_run':False,'certified':False,'artifact_kind':v['artifact_kind'],'result_path':str(out),'result_sha256':sha(out),'journal_path':str(j),'journal_sha256':sha(j),'mandatory_routes_path':str(mr),'mandatory_routes_sha256':sha(mr),'construction_path':str(cp),'construction_sha256':sha(cp),'execution':ex,'ended_utc':now()};target=croot/'pool.json'
   else:
    check_checkout(case['source_code'],case['execution_commit']); sources=[source_completion(root,p,mh) for p in case['source_partitions']]
    uspec={'partitions':case['source_partitions'],'sources':[{'case_id':x['case_id'],'pool_path':x['result_path'],'pool_sha256':x['result_sha256'],'journal_path':x['journal_path'],'journal_sha256':x['journal_sha256']} for x in sources]}
    save(attempt/'union_spec.json',uspec);union=attempt/'source_pool.json'
    union_execution=run([sys.executable,str(root/'pool_logic.py'),'union','--spec',str(attempt/'union_spec.json'),'--out',str(union)],attempt,str(root),600,os.environ.copy(),'union_execution')
    union_sha=sha(union);journal=Path(str(union)+'.columns.jsonl');journal_sha=sha(journal)
    scored=[]
    for x in sources:
     c=read(x['result_path'])['pool_construction'];scored.append((c['mandatory_route_count'],c['mandatory_total_expanded_grid_cost'],x['case_id'],x))
    scored.sort();chosen=scored[0][3];start=Path(chosen['mandatory_routes_path']);start_sha=chosen['mandatory_routes_sha256'];out=attempt/'result.json'
    argv=[sys.executable,str(Path(case['source_code'])/'src'/'run_exact_pool_mip.py'),'--result',str(union),'--timelimit',str(case['solver_budget_s']),'--stage1-timelimit',str(case['stage1_s']),'--threads','4','--two-stage','--cover','--initial-partition-routes',str(start),'--verified-expanded-initial-partition','--data-dir',case['data_dir'],'--reference-data-dir',case['data_dir'],'--progress-dir',str(attempt/'progress'),'--gurobi-log',str(attempt/'gurobi.log'),'--out',str(out)]
    env=os.environ.copy();env.update({'EVSP_EXPECTED_COMMIT':case['execution_commit'],'EVSP_REQUIRE_DETACHED':'1','EVSP_MIP_EXPECTED_RESULT_SHA256':union_sha,'EVSP_MIP_EXPECTED_JOURNAL_SHA256':journal_sha,'EVSP_MIP_EXPECTED_INITIAL_PARTITION_SHA256':start_sha})
    register(case,token,out);state.update(status='solving',source_pool_sha256=union_sha,source_journal_sha256=journal_sha,source_partitions=case['source_partitions']);save(attempt/'state.json',state)
    ex=run(argv,attempt,case['source_code'],case['watchdog_s'],env,'mip_execution');v=read(out);audit=v['physical_pool_audit']
    if not v['physical_replay_validated'] or audit['rejected_columns'] or audit['deterministically_repaired'] or v['source_result_sha256']!=union_sha or v['source_journal_sha256']!=journal_sha:raise ValueError('MIP physical/hash gate')
    source_bus_counts={x[2]:x[0] for x in scored};warm=scored[0][0]
    if v.get('buses') is not None and v['buses']>warm:raise ValueError('MIP incumbent worse than admitted warm start')
    final={**state,'kind':'mip','status':'finished','usable':True,'optimization_run':True,'artifact_kind':'finite_parent_pool_mip','result_path':str(out),'result_sha256':sha(out),'source_status_sha256':union_sha,'source_pool_path':str(union),'source_pool_sha256':union_sha,'source_journal_path':str(journal),'source_journal_sha256':journal_sha,'physical_replay_validated':True,'buses':v.get('buses'),'fleet_proven':v.get('fleet_proven'),'source_partition_incumbent_buses':source_bus_counts,'best_known_source_upper_bound':warm,'warm_start_partition':chosen['case_id'],'warm_start_buses':warm,'incumbent_no_worse_than_warm_start':v.get('buses') is None or v['buses']<=warm,'union_execution':union_execution,'execution':ex,'ended_utc':now()};target=croot/'mip_result.json'
   publish_link(out,target);save(croot/'completion.json',final);save(attempt/'state.json',final)
  except BaseException as e:
   save(attempt/'state.json',{**state,'status':'failed','error':repr(e),'ended_utc':now()});raise
if __name__=='__main__':
 import argparse;p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();execute(a.root,a.case)
