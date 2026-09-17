"""Only true dependencies: each arm's125 replay shards + its own graph cache."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;slurm=Path('/usr/local/slurm/slurm-25.05.5/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(d):
 p=root/'continuation_jobs.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
with open(root/'submit_continuation.lock','w') as f:
 fcntl.flock(f,fcntl.LOCK_EX);m=json.loads((root/'manifest.json').read_text());cm=json.loads((root/'continuation_manifest.json').read_text());replay=json.loads((root/'replay_jobs.json').read_text())['jobs']['replay']['job_id']
 d=json.loads((root/'continuation_jobs.json').read_text()) if (root/'continuation_jobs.json').exists() else {'continuation_manifest_sha256':sha(root/'continuation_manifest.json'),'jobs':{},'intents':{}}
 assert d['continuation_manifest_sha256']==sha(root/'continuation_manifest.json') and not d['intents']
 assert cm['campaign_manifest_sha256']==sha(root/'manifest.json')
 for name,h in cm['tooling_sha256'].items():assert sha(root/name)==h
 def submit(stage,case,deps,mem,cpus,wall):
  key=case+'__'+stage
  if key in d['jobs']:return d['jobs'][key]['job_id']
  cmd=[str(slurm/'sbatch'),'--parsable','--partition=default_partition','--exclude=scaglione-compute-01',f'--cpus-per-task={cpus}',f'--mem={mem}G','--time='+wall,'--requeue','--open-mode=append','--job-name=a3_'+case+'_'+stage,'--comment=EVSP_ACTION3_FULL_'+key,'--output='+str(root/'logs'/('%j_'+key+'.out')),'--error='+str(root/'logs'/('%j_'+key+'.err')),'--chdir='+str(root),'--export=ALL']
  if deps:cmd+=['--dependency=afterok:'+':'.join(deps)]
  cmd+=[str(root/'worker.sh'),str(root),stage,'--case',case]
  d['intents'][key]={'command':cmd,'dependencies':deps,'created_epoch':time.time()};save(d)
  p=subprocess.run(cmd,capture_output=True,text=True)
  if p.returncode:raise RuntimeError(p.stderr)
  job=p.stdout.strip().split(';')[0];assert job.isdigit();d['jobs'][key]={'job_id':job,'dependencies':deps,'command':cmd,'submitted_epoch':time.time()};del d['intents'][key];save(d)
  sc=subprocess.check_output([str(slurm/'scontrol'),'show','job',job,'--oneliner'],text=True);assert 'ExcNodeList=scaglione-compute-01' in sc;d['jobs'][key]['scontrol_initial']=sc.strip();save(d);return job
 cache={name:submit('cache',name,[],spec['cache_memory_gb'],1,'40:00:00') for name,spec in cm['cases'].items()}
 assembly={}
 for arm in m['arms']:
  case=next(name for name,spec in cm['cases'].items() if spec['arm']==arm);idx=m['arms'].index(arm)
  assembly[arm]=submit('assemble',case,[f'{replay}_{idx+6*i}' for i in range(125)],24,1,'04:00:00')
 for name,spec in cm['cases'].items():
  cg=submit('cg',name,[cache[name],assembly[spec['arm']]],spec['cache_memory_gb'],8,'06:00:00')
  submit('mip',name,[cg],24,8,'01:15:00')
 print(json.dumps(d))
