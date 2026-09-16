"""Read-only pilot status collection; never submits or mutates experiments."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SLURM=Path('/usr/local/slurm/slurm-25.05.5/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.loads((ROOT/'jobs.json').read_text());ids=[r['job_id'] for r in d['jobs'].values()]
start=datetime.datetime.fromtimestamp(min(r['submitted_epoch'] for r in d['jobs'].values())-60).strftime('%Y-%m-%dT%H:%M:%S')
def query(tool,args):
 p=subprocess.run([str(SLURM/tool),*args],capture_output=True,text=True,timeout=30)
 return {'ok':p.returncode==0,'text':p.stdout,'stderr':p.stderr,'returncode':p.returncode}
result={'collected_epoch':time.time(),'collector_sha256':sha(__file__),'jobs':d['jobs'],
 'squeue':query('squeue',['-j',','.join(ids),'--array','--noheader','--format=%i|%T|%R|%M|%N']),
 'sacct':query('sacct',['-u','nc437','-j',','.join(ids),'--starttime='+start,'--duplicates','--noheader','--parsable2','--format=JobIDRaw,State,ExitCode,ElapsedRaw,MaxRSS,NodeList']),
 'attempts':[],'control_gate':None,'conditional_pilots_authorized_to_submit':False,
 'full_replay_cg_mip_authorized':False}
for arm in d['jobs']:
 for p in sorted((ROOT/'runs'/arm).glob('*/execution.json')):
  row={'arm':arm,'path':str(p),'sha256':sha(p),'execution':json.loads(p.read_text())}
  endpoint=p.parent/'pilot_result.json'
  if endpoint.exists():row['result']=json.loads(endpoint.read_text());row['result_sha256']=sha(endpoint)
  elif row['execution']['status']=='failed':
   row['error_logs']={n:(p.parent/n).read_text()[-4000:] for n in ['extract.log','replay.log'] if (p.parent/n).exists()}
  result['attempts'].append(row)
gate=ROOT/'control_pass.json'
if gate.exists():
 g=json.loads(gate.read_text());result['control_gate']={'path':str(gate),'sha256':sha(gate),'data':g}
 baseline=d['jobs']['baseline']['job_id']
 completed=any(line.split('|')[:2]==[baseline,'COMPLETED'] for line in result['sacct']['text'].splitlines())
 result['conditional_pilots_authorized_to_submit']=bool(g['pass'] and g['resource_ok'] and g['unknown_count']==0 and g['pilot_manifest_sha256']==sha(ROOT/'pilot_manifest.json') and completed and len(d['jobs'])<6)
print(json.dumps(result))
