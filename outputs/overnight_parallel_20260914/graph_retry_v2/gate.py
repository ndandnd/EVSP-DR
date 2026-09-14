"""Conditional graph recovery gate. Preparation only; never invokes sbatch/scontrol."""
from pathlib import Path
import argparse,fcntl,json,os,re,subprocess,time
import process_worker as w
TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def eligible(scheduler,execution):
 state=scheduler['state'].split()[0].rstrip('+')
 if state not in TERMINAL:raise ValueError('Original task is not terminal; never duplicate an active graph')
 if state=='TIMEOUT':return True
 return state=='FAILED' and scheduler['exit_code']=='124:0' and execution.get('watchdog_time_limit') is True and execution.get('status')=='time_limit'

def replacement_dependencies(original,oldgraph,newgate):
 assert re.fullmatch(r'\d+(?:_\d+)?',oldgraph) and re.fullmatch(r'\d+',newgate)
 parts=original.split(',');found=0;out=[]
 for part in parts:
  match=re.fullmatch(r'afterok:(\d+(?:_\d+)?)(?:\((?:unfulfilled|fulfilled)\))?',part)
  if not match:raise ValueError('Unexpected dependency syntax; reconcile manually')
  parent=match.group(1)
  if parent==oldgraph:parent=newgate;found+=1
  out.append('afterok:'+parent)
 if found!=1:raise ValueError('Expected exactly one original graph edge')
 return ','.join(out)

def validate_cache(cache,meta_path,case,expected_hash=None):
 meta=w.read(meta_path)
 for k,v in case['cache_identity'].items():
  if meta.get('identity',{}).get(k)!=v:raise ValueError('Graph cache identity mismatch: '+k)
 digest=w.sha(cache)
 if digest!=meta['pickle_sha256'] or (expected_hash and digest!=expected_hash):raise ValueError('Graph pickle hash mismatch')
 return meta,digest

def read_scheduler_row(job,timeout_s):
 s='/usr/local/slurm/slurm-25.05.5/bin/'
 p=subprocess.run([s+'sacct','-n','-X','-P','-j',job,'-o','JobID,JobIDRaw,State,ExitCode'],capture_output=True,text=True,timeout=timeout_s,check=True)
 rows=[x.split('|') for x in p.stdout.splitlines() if x.strip()]
 rows=[x for x in rows if x[0]==job]
 if len(rows)>1:raise ValueError('Ambiguous original scheduler allocation; refusing recovery')
 if not rows:return None
 x=rows[0]
 if len(x)<4:raise ValueError('Malformed original scheduler allocation')
 return dict(job_id=x[0],raw_job_id=x[1],state=x[2],exit_code=x[3])

def wait_scheduler_terminal(job,reader,*,clock=time.monotonic,pause=time.sleep,limit=120.0,interval=5.0):
 started=clock();deadline=started+limit;polls=0;last=None
 while True:
  remaining=deadline-clock()
  if remaining<=0:raise ValueError('Original graph accounting not terminal within120seconds: '+repr(last))
  polls+=1
  try:last=reader(job,min(30.0,remaining))
  except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as exc:last={'accounting_error':type(exc).__name__}
  # Ambiguous/malformed exact-task rows raise immediately; never choose one.
  if last and last.get('state','').split() and last['state'].split()[0].rstrip('+') in TERMINAL:
   return dict(last,accounting_polls=polls,accounting_wait_s=clock()-started)
  remaining=deadline-clock()
  if remaining<=0:raise ValueError('Original graph accounting not terminal within120seconds: '+repr(last))
  pause(min(interval,remaining))

def scheduler(job):
 return wait_scheduler_terminal(job,read_scheduler_row)

def link_absent(path,target):
 # Never overwrite an original output, including a dangling link.
 path=Path(path)
 if path.exists() or path.is_symlink():raise ValueError('Refusing to overwrite original artifact: '+str(path))
 path.symlink_to(Path(target).resolve())

def run(root,cid):
 root=Path(root);m=w.read(root/'manifest.json');c=m['cases'][cid]
 for n,d in m['tooling_sha256'].items():w.require_hash(root/n,d)
 w.require_hash(m['original_manifest_path'],m['original_manifest_sha256']);w.check_code(m['source_code'],m['execution_commit'])
 for p,d in c['static_hashes'].items():w.require_hash(p,d)
 original=Path(c['original_case_dir']);out=root/'cases'/cid
 out.mkdir(parents=True,exist_ok=True)
 with (original/'graph_recovery.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  sched=scheduler(c['original_graph_job'])
  marker=original/'cache_result.json';cache=original/'network.pkl';meta=original/'network.pkl.manifest.json'
  if marker.exists():
   prior=w.read(marker);validate_cache(cache,meta,c,prior['cache_sha256'])
   w.save(out/'completion.json',dict(kind='graph_recovery_gate',status='cache_valid_no_rebuild',graph_rebuilt=False,source_construction_kind=prior.get('graph_construction_kind','original'),source_cache_result=str(marker),source_cache_result_sha256=w.sha(marker),cache_sha256=prior['cache_sha256'],scheduler=sched,manifest_sha256=w.sha(root/'manifest.json')));return
  attempts=sorted((original/'cache').glob('*/execution.json'),key=lambda p:p.stat().st_mtime)
  if not attempts:raise ValueError('No original execution evidence')
  ep=attempts[-1];execution=w.read(ep)
  assert execution['attempt'].split('_r')[0]==sched['raw_job_id']
  assert execution['manifest_sha256']==m['original_manifest_sha256'] and execution['input_sha256']==c['input_sha256'] and execution['execution_commit']==m['execution_commit']
  if not eligible(sched,execution):raise ValueError('Only actual scheduler TIMEOUT or internal timeout124 is eligible')
  if cache.exists() or cache.is_symlink() or meta.exists() or meta.is_symlink():raise ValueError('Existing unpublished cache slot requires manual adoption audit')
  attempt=out/'attempts'/(os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0'));attempt.mkdir(parents=True,exist_ok=False)
  args=[v.replace('{attempt}',str(attempt)) for v in c['argv']]
  record=w.run_process(args,attempt,m['source_code'],m['graph_seconds'],os.environ.copy())
  target=attempt/'network.pkl';targetmeta=Path(str(target)+'.manifest.json');native,digest=validate_cache(target,targetmeta,c)
  # Recheck scheduler terminality before publishing into the previously absent graph slot.
  scheduler(c['original_graph_job'])
  provenance=dict(kind='graph_timeout_recovery',case_id=cid,attempt=attempt.name,original_failed_execution_path=str(ep),original_failed_execution_sha256=w.sha(ep),original_scheduler=sched,recovery_manifest_sha256=w.sha(root/'manifest.json'),source_commit=m['execution_commit'],input_sha256=c['input_sha256'],graph_watchdog_s=m['graph_seconds'],execution=record,cache_path=str(target),cache_sha256=digest,cache_manifest_sha256=w.sha(targetmeta))
  w.save(attempt/'recovery_provenance.json',provenance)
  link_absent(cache,target);link_absent(meta,targetmeta)
  w.save(marker,dict(case_id=cid,attempt=attempt.name,cache_sha256=digest,cache_manifest_sha256=w.sha(targetmeta),runtime_s=record['wall_s'],source_commit=m['execution_commit'],input_sha256=c['input_sha256'],graph_construction_kind='timeout_recovery',recovery_provenance_path=str(attempt/'recovery_provenance.json'),recovery_provenance_sha256=w.sha(attempt/'recovery_provenance.json'),original_failed_attempt=execution['attempt'],graph_watchdog_s=m['graph_seconds']))
  w.save(out/'completion.json',dict(kind='graph_recovery_gate',status='recovered',graph_rebuilt=True,cache_sha256=digest,recovery_provenance=provenance))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();run(a.root,a.case)
