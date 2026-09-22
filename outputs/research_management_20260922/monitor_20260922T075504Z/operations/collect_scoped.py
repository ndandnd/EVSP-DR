from pathlib import Path
import subprocess,json,datetime,hashlib,base64
OUT=Path(__file__).resolve().parent
REMOTE=r'''
from pathlib import Path
import subprocess,json,datetime,hashlib,base64
B=Path('/usr/local/slurm/slurm-25.05.5/bin');R=Path('/home/nc437/ladder-lite/chain_extension_33_40_20260921');S=Path('/home/nc437/ladder-lite/strict_packed_successors_20260921')
policy=Path('/home/nc437/ladder-lite/SCAGLIONE_RESOURCE_POLICY.md').read_bytes()
d={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'policy':{'text':policy.decode(),'sha256':hashlib.sha256(policy).hexdigest()},'files':{},'metadata':[]}
def run(args):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True);return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
def add(key,p):
 if p.is_file():
  b=p.read_bytes();d['files'][key]={'remote':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()}
add('baseline/jobs.json',R/'jobs.json');add('baseline/manifest.json',R/'manifest.json')
ids=[str(x['job_id']) for x in json.loads((R/'jobs.json').read_text())]+['668433','668434']
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
  elif p.name in ['COMPLETE.json','CG_COMPLETE.json','MIP_COMPLETE.json','result.json','cg.json','status.json','checkpoint.json','command.json'] and size<15000000:
   add('baseline/'+str(rel),p)
  elif p.suffix=='.log' and size<5000000 and (p.parent/'result.json').exists():add('baseline/'+str(rel),p)
case=S/'w5_k19_18E2'
for n in ['manifest.json','jobs.json']:add('strict/'+n,S/n)
for n in ['CG_COMPLETE.json','MIP_COMPLETE.json']:add('strict/k19/'+n,case/n)
for stage,j in [('cg','668433'),('mip','668434')]:
 p=case/stage/(j+'_r0')
 for f in sorted(p.iterdir()):
  if f.is_file() and (f.suffix in ['.json','.log'] or f.name.endswith('.json.gz')) and f.stat().st_size<15000000:add('strict/k19/'+stage+'/'+f.name,f)
 # Enumerate attempt names, not pool payloads.
 d[stage+'_attempts']=[str(x.relative_to(case)) for x in (case/stage).glob('*')]
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
(OUT/'SCAGLIONE_RESOURCE_POLICY.md').write_text(d['policy']['text'])
print(json.dumps({'collected_utc':d['collected_utc'],'files':len(records),'queue':d['queue']['stdout'],'strict_files':[x['local'] for x in records if x['local'].startswith('strict/')]}))
