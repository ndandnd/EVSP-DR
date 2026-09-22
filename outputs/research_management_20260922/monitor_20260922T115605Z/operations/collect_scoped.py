from pathlib import Path
import subprocess,json,datetime,hashlib,base64
OUT=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import subprocess,json,datetime,hashlib,base64
B=Path('/usr/local/slurm/slurm-25.05.5/bin');R=Path('/home/nc437/ladder-lite/chain_extension_33_40_20260921')
d={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'metadata':[]}
def run(args):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True);return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def add(key,p):
 if p.is_file():
  b=p.read_bytes();d['files'][key]={'remote':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()}
add('baseline/jobs.json',R/'jobs.json');add('baseline/manifest.json',R/'manifest.json')
ids=[str(x['job_id']) for x in json.loads((R/'jobs.json').read_text())]
d['queue']=run([B/'squeue','-r','-u','nc437','-h','-o','%i|%j|%T|%M|%R|%E'])
d['accounting']=run([B/'sacct','-D','--starttime=2026-09-21','-j',','.join(sorted(set(ids))),'-n','-P','--format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,ExitCode,NodeList,Start,End,AllocCPUS'])
# Only files in the live baseline campaign. No historical campaign collector.
for case in sorted((R/'cases').glob('*')):
 if not case.is_dir():continue
 for p in sorted(case.rglob('*')):
  if not p.is_file() or p.is_symlink():continue
  rel=p.relative_to(R);parts=rel.parts;size=p.stat().st_size
  if 'cache' in parts:
   d['metadata'].append({'remote':str(p),'bytes':size,'modified_unix':p.stat().st_mtime})
   if p.suffix=='.json' and size<500000:add('baseline/'+str(rel),p)
  elif p.name in ['COMPLETE.json','CG_COMPLETE.json','MIP_COMPLETE.json','result.json','mip.json','cg.json','status.json','checkpoint.json','command.json','execution.json'] and size<15000000:
   add('baseline/'+str(rel),p)
  elif p.suffix=='.log' and size<5000000 and any((p.parent/x).exists() for x in ['result.json','mip.json','cg.json']):add('baseline/'+str(rel),p)
d['log_tails']=[]
for p in sorted((R/'logs').glob('*')):
 if p.is_file() and p.stat().st_size:
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-1800));tail=f.read().decode(errors='replace')
  d['log_tails'].append({'path':str(p),'bytes':p.stat().st_size,'tail':tail})
print(json.dumps(d))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','nc437@unicorn-login-01.coecis.cornell.edu','python3 -'],input=REMOTE,text=True,capture_output=True)
if p.returncode:
 (OUT/'ssh_failure.txt').write_text(p.stderr);raise SystemExit('SSH/collection failed: '+p.stderr)
d=json.loads(p.stdout);records=[]
for key,v in d.pop('files').items():
 f=OUT/key;f.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(v.pop('data'));assert hashlib.sha256(b).hexdigest()==v['sha256'];f.write_bytes(b);records.append({'local':key,**v})
d['files']=records;(OUT/'collection.json').write_text(json.dumps(d,indent=2)+'\n')
for k in ['queue','accounting']:(OUT/(k+'.txt')).write_text(d[k]['stdout']);assert d[k]['returncode']==0,d[k]
import collections
q=[x.split('|') for x in d['queue']['stdout'].splitlines()];scope=[x for x in q if x[1].startswith('drX_')]
print(json.dumps({'collected_utc':d['collected_utc'],'files':len(records),'user_states':dict(collections.Counter(x[2] for x in q)),'baseline_states':dict(collections.Counter(x[2] for x in scope)),'baseline_pending_reasons':dict(collections.Counter(x[4] for x in scope if x[2]=='PENDING'))}))
