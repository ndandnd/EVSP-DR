"""Submit exact full-coverage sharding and ONE globally throttled replay array."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;slurm=Path('/usr/local/slurm/slurm-25.05.5/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(d):
 p=root/'replay_jobs.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
with open(root/'submit_replay.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);m=json.loads((root/'manifest.json').read_text())
 d=json.loads((root/'replay_jobs.json').read_text()) if (root/'replay_jobs.json').exists() else {'manifest_sha256':sha(root/'manifest.json'),'jobs':{},'intents':{}}
 assert d['manifest_sha256']==sha(root/'manifest.json') and not d['intents']
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=m['code'],text=True).strip()==m['execution_commit']
 for name,h in m['tooling_sha256'].items():assert sha(root/name)==h
 assert sha(root/'prepared/manifest.json')==m['prepared_manifest_sha256'];(root/'logs').mkdir(exist_ok=True)
 for stage in ['prepare','replay']:
  if stage in d['jobs']:continue
  cmd=[str(slurm/'sbatch'),'--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=1',
       '--mem='+('8G' if stage=='prepare' else '4G'),'--time='+('01:00:00' if stage=='prepare' else '06:00:00'),
       '--requeue' if stage=='replay' else '--no-requeue','--open-mode=append','--job-name=a3full_'+stage,
       '--comment=EVSP_ACTION3_FULL_'+stage,'--output='+str(root/'logs'/('%A_%a_'+stage+'.out')),'--error='+str(root/'logs'/('%A_%a_'+stage+'.err')),'--chdir='+str(root),'--export=ALL']
  if stage=='replay':cmd+=['--array=0-749%50','--dependency=afterok:'+d['jobs']['prepare']['job_id']]
  cmd+=[str(root/'worker.sh'),str(root),stage]
  d['intents'][stage]={'command':cmd,'created_epoch':time.time()};save(d);r=subprocess.run(cmd,capture_output=True,text=True)
  if r.returncode:raise RuntimeError(r.stderr)
  job=r.stdout.strip().split(';')[0];assert job.isdigit();d['jobs'][stage]={'job_id':job,'command':cmd,'submitted_epoch':time.time()};del d['intents'][stage];save(d)
  sc=subprocess.check_output([str(slurm/'scontrol'),'show','job',job,'--oneliner'],text=True);assert 'ExcNodeList=scaglione-compute-01' in sc
  d['jobs'][stage]['scontrol_initial']=sc.strip();save(d)
 print(json.dumps(d))
