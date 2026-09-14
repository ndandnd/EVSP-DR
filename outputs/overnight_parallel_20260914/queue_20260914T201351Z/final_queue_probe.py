"""One queue/accounting capture after authorized final-wave launch; no mutations."""
from pathlib import Path
import json,subprocess,datetime,time,hashlib
H=Path('/home/nc437/ladder-lite');S='/usr/local/slurm/slurm-25.05.5/bin/'
roots={'seed':'overnight_parallel_20260914','prefix':'retrospective_prefix_controls_20260914','decomposition_first':'decomposition_pool_union_20260914','decomposition_second':'decomposition_lp_support_union_20260914','graph_gates_v2':'graph_timeout_gates_v2_20260914'}
def read(p):return json.loads(p.read_text())
roster={};manifests={}
for family,name in roots.items():
 root=H/name;payload=read(root/'jobs.json');manifest=root/'manifest.json';manifests[family]={'path':str(manifest),'sha256':hashlib.sha256(manifest.read_bytes()).hexdigest()}
 if isinstance(payload,list):entries=[(r['case_id'],r) for r in payload]
 else:entries=list(payload['jobs'].items())
 for cid,r in entries:
  kind=r.get('kind') or ('pool_construction' if r.get('wave')=='construction' else 'mip')
  roster[str(r['job_id'])]={'family':family,'case_id':cid,'kind':kind}
for name in ['chain_extension_20260913','chain_extension_20260914']:
 for cid,stages in read(H/name/'case_jobs.json').items():
  for stage,job in stages.items():roster[str(job)]={'family':name,'case_id':cid,'kind':'graph' if stage=='cache' else stage}
q=subprocess.run([S+'squeue','--me','-h','-o','%i|%j|%T|%r|%P|%N|%E|%C|%m'],capture_output=True,text=True,timeout=30)
assert q.returncode==0,q.stderr
sampled=datetime.datetime.now(datetime.timezone.utc).isoformat()
a=subprocess.run([S+'sacct','-n','-X','-P','-j',','.join(sorted(roster)),'-o','JobID,JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,ReqMem,NodeList'],capture_output=True,text=True,timeout=40)
assert a.returncode==0,a.stderr
fs={}
for name in ['chain_extension_20260913','chain_extension_20260914']:
 root=H/name;cases={}
 for cid,c in read(root/'manifest.json')['cases'].items():
  d=root/'cases'/cid;row={'graph_published':(d/'cache_result.json').exists(),'cg_published':(d/'cg_provenance.json').exists(),'mip_published':(d/'mip_provenance.json').exists()}
  if row['graph_published']:row['cache_result']=read(d/'cache_result.json')
  paths=sorted((d/'cache').glob('*/progress.jsonl'),key=lambda p:p.stat().st_mtime)
  if paths:
   p=paths[-1];raw=p.read_text().splitlines();items=[json.loads(x) for x in raw[-3:] if x.strip()];row.update(progress_path=str(p),progress_mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat(),progress_age_s=time.time()-p.stat().st_mtime,latest_progress=items[-1] if items else None)
  cases[cid]=row
 fs[name]=cases
print(json.dumps({'sampled_utc':sampled,'queue_raw':q.stdout,'sacct_raw':a.stdout,'roster':roster,'manifests':manifests,'filesystem':fs}))
