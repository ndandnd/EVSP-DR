"""Bounded read-only review collection. Only synthetic DR science leaves Unicorn."""
import concurrent.futures,csv,datetime,hashlib,io,json,re,subprocess,sys,time
from pathlib import Path
BASE=Path('/home/nc437/ladder-lite');PY='/home/nc437/evsp_env/bin/python';SLURM=Path('/usr/local/slurm/slurm-25.05.5/bin');STAMP=sys.argv[1];SCRIPT_SHA=sys.argv[2]
assert re.fullmatch(r'\d{8}T\d{6}Z',STAMP)
ROOTS={'p1':BASE/'review_p1_mip_20260916','dr':BASE/'review_dr_mincharge_20260916','strict':BASE/'review_strict_c5_20260916','random':BASE/'random_trip_groups_c1_20260916','full40':BASE/'review_full40_20260916','frolunda':BASE/'review_frolunda_20260916','f6_k5':BASE/'advisor_f6_k5_reserve_20260916'}
OUT=BASE/'review_monitor_20260916'/STAMP;OUT.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def run(argv,timeout=55):
 p=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
 if p.returncode:raise RuntimeError(f'collector command failed ({p.returncode}): '+p.stderr[-2000:])
 return p.stdout

def full_collect(b):
 m=read(b/'manifest.json');rows=[]
 for cid,c in m['cases'].items():
  root=b/'cases'/cid;r={'case_id':cid,'trip_count':c['trip_count'],'input_sha256':c['input_sha256'],'cache_receipt_present':(root/'cache_result.json').exists()}
  if r['cache_receipt_present']:r['cache_receipt']=read(root/'cache_result.json')
  for mode,name in [('cg','cg.json'),('mip','mip_result.json')]:
   p=root/name;r[mode+'_canonical_result_present']=p.exists()
   if not p.exists():continue
   d=read(p);r[mode+'_path']=str(p);r[mode+'_sha256']=sha(p)
   if mode=='cg':r.update(cg_stop_reason=d.get('stop_reason'),cg_pricing_certified=d.get('certified_rc_optimal'),cg_wall_s=d.get('wall_s'),cg_iterations=d.get('iterations'),cg_final=d.get('final'),cg_final_lp_source=d.get('final_lp_source'))
   else:r.update(buses=d.get('buses'),pool_fleet_bound=d.get('fleet_bound'),fleet_proven_in_pool=d.get('fleet_proven'),physical_route_replay=d.get('physical_replay_validated'),duplicate_cleanup=d.get('duplicate_trip_removal_validated'),shared_capacity_validated=d.get('cross_route_charger_capacity_validated'))
  rows.append(r)
 return {'rows':rows,'manifest_sha256':sha(b/'manifest.json')}

def collect(name,b):
 try:
  if name=='p1':
   run([PY,str(b/'summarize.py'),'--root',str(b),'--out',str(OUT/'p1')]);result=read(OUT/'p1/results.json');script=b/'summarize.py'
   run([PY,str(b/'audit_endpoints.py'),'--root',str(b),'--out',str(OUT/'p1')])
   audit=read(OUT/'p1/endpoint_audit.json')
   for cell in audit['checks']:
    stage=cell.pop('two_stage',{})
    cell['observed_stage1_limit_s']=stage.get('stage1_time_limit_s')
    cell['observed_stage2_reserved_s']=stage.get('stage2_reserved_time_s')
   result['endpoint_audit']=audit
   result['endpoint_auditor_sha256']=sha(b/'audit_endpoints.py')
  elif name=='dr':
   # Native collector writes SE3-derived results ONLY inside remote OUT.
   run([PY,str(b/'collect.py'),'--root',str(b),'--out',str(OUT/'dr')]);script=b/'collect.py'
   with (OUT/'dr/synthetic_results.csv').open() as f:rows=list(csv.DictReader(f))
   assert all(r['tariff'] in {'peak08','peak12','peak18'} for r in rows)
   result={'rows':rows,'metadata':read(OUT/'dr/collection.json'),'publication':'Synthetic rows only; internal real-price rows retained on Unicorn and excluded from this transport.'}
  elif name=='f6_k5':
   run([PY,str(b/'collect.py'),'--root',str(b),'--out',str(OUT/'f6_k5')]);script=b/'collect.py';result=read(OUT/'f6_k5/comparison.json')
   assert all(r['peak'] in {'peak08','peak12','peak18'} for r in result['rows'])
  elif name=='full40':result=full_collect(b);script=Path(__file__)
  else:
   script=b/('collect_remote.py' if name=='strict' else 'collect.py');result=json.loads(run([PY,str(script)]))
  # Do not transport bulky job commands; scheduler registry below is explicit.
  if name=='strict':result.pop('jobs',None)
  if name=='frolunda':result.pop('job',None)
  return name,{'collection_ok':True,'collector_path':str(script),'collector_sha256':SCRIPT_SHA if name=='full40' else sha(script),'data':result}
 except Exception as exc:
  # DR failures may mention restricted price contents. No raw diagnostics leave.
  return name,{'collection_ok':False,'error':type(exc).__name__ if name=='dr' else str(exc)}

started=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=len(ROOTS)) as pool:campaigns=dict(pool.map(lambda t:collect(*t),ROOTS.items()))
registry=[];attempts=[]
def jobrows(value):
 if isinstance(value,dict):
  if 'job_id' in value and str(value['job_id']).isdigit():yield value
  else:
   for v in value.values():yield from jobrows(v)
 elif isinstance(value,list):
  for v in value:yield from jobrows(v)
for name,b in ROOTS.items():
 for r in jobrows(read(b/'jobs.json')):
  registry.append({'campaign':name,'job_id':str(r['job_id']),'case_id':r.get('case_id'),'mode':r.get('mode',r.get('kind')),'dependencies':r.get('dependencies',[])})
 private=set()
 if name=='dr':private={cid for cid,c in read(b/'manifest.json')['cases'].items() if c['publication']=='internal_only'}
 for filename in ['state.json','execution.json','command.json','COMPLETE.json']:
  for p in sorted((b/'cases').glob('*/*/*/'+filename)):
   case=p.relative_to(b/'cases').parts[0]
   if case in private:continue  # internal scientific/process details never exported
   try:d=read(p)
   except Exception:continue
   row={'campaign':name,'case_id':case,'path':str(p),'sha256':sha(p),'receipt_kind':filename}
   for key in ['attempt','status','mode','kind','stage','started_epoch','ended_epoch','started_unix','finished_unix','started_utc','ended_utc','returncode','watchdog_time_limit','signals']:
    if key in d:row[key]=d[key]
   if d.get('error'):row['worker_error']=str(d['error'])[:1500]
   attempts.append(row)
# Register the four reused seed-zero jobs even before a result exists.
for cell in read(ROOTS['p1']/'manifest.json')['reused_cells']:
 for r in jobrows(read(Path(cell['campaign'])/'jobs.json')):
  if r.get('case_id')==cell['case']:
   registry.append({'campaign':'p1','job_id':str(r['job_id']),'case_id':cell['case_id'],'mode':r.get('kind'),'dependencies':r.get('dependencies',[]),'reused':True})
ids={r['job_id'] for r in registry}
for a in attempts:
 m=re.search(r'/(\d+)_r\d+/',a['path'])
 if m:ids.add(m.group(1))
# Reused P1 endpoints are old controlled attempts, not new submissions.
for r in campaigns.get('p1',{}).get('data',{}).get('rows',[]):
 m=re.search(r'/(\d+)_r\d+/',r.get('result_path',''))
 if m:ids.add(m.group(1))
end=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
scheduler={}
commands={'squeue':[str(SLURM/'squeue'),'-u','nc437','--jobs='+','.join(sorted(ids)),'--array','--noheader','--format=%i|%T|%R|%M|%l|%C|%m|%N'],
 'sacct':[str(SLURM/'sacct'),'-u','nc437','-j',','.join(sorted(ids)),'--starttime=2026-09-16T00:00:00','--endtime='+end,'--duplicates','--parsable2','--format=JobIDRaw,JobID,State,ExitCode,ElapsedRaw,Start,End,AllocCPUS,ReqMem,MaxRSS,NodeList,Restarts']}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures={pool.submit(run,argv):key for key,argv in commands.items()}
 for future,key in [(f,k) for f,k in futures.items()]:
  try:scheduler[key]={'ok':True,'text':future.result(),'command':commands[key]}
  except Exception as exc:scheduler[key]={'ok':False,'error':str(exc),'command':commands[key]}
result={'schema':'evsp-review-focused-collection-v1','collected_utc':STAMP,'elapsed_s':time.monotonic()-started,'campaigns':campaigns,'registered_jobs':registry,'public_attempt_artifacts':attempts,'scheduler':scheduler,'private_data_policy':'No SE3 scientific outputs, costs, prices or private attempt details are transported; native DR private collection remains on Unicorn. Scheduler metadata covers all campaign jobs.','evidence_levels':'Collector success, worker receipt, scheduler state, CG certificate, pool MIP proof, physical replay and exact-once validation are distinct fields; no stage completion implies another proof.'}
(OUT/'public_snapshot.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
