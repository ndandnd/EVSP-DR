"""Idempotent pilot-only submission, with evidence gate for the other five arms."""
import argparse,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--remaining',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent;slurm=Path('/usr/local/slurm/slurm-25.05.5/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
with open(root/'submit.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);m=json.loads((root/'pilot_manifest.json').read_text())
 ledger=root/'jobs.json';d=json.loads(ledger.read_text()) if ledger.exists() else {'pilot_manifest_sha256':sha(root/'pilot_manifest.json'),'jobs':{},'intents':{}}
 assert d['pilot_manifest_sha256']==sha(root/'pilot_manifest.json') and not d['intents'],'manifest changed or unresolved sbatch intent'
 assert sha(root/'prepared/manifest.json')==m['prepared_manifest_sha256']
 pin=json.loads((root/'prepared/manifest.json').read_text())
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=m['code'],text=True).strip()==pin['execution_commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=m['code'],text=True).strip()
 for name,h in pin['tooling_sha256'].items():assert sha(root/'prepared'/name)==h
 for name,h in m['tooling_sha256'].items():assert sha(root/name)==h
 if a.remaining:
  gate=json.loads((root/'control_pass.json').read_text());assert gate['pass'] and gate['resource_ok'] and gate['unknown_count']==0
  assert gate['pilot_manifest_sha256']==sha(root/'pilot_manifest.json')
  baseline=d['jobs']['baseline']['job_id']
  status=subprocess.check_output([str(slurm/'sacct'),'-j',baseline,'--noheader','--parsable2','--format=JobIDRaw,State'],text=True)
  assert any(line.split('|')[:2]==[baseline,'COMPLETED'] for line in status.splitlines()),'control scheduler allocation has not completed successfully'
  arms=m['conditional_arms'];d['remaining_gate_sha256']=sha(root/'control_pass.json')
 else:arms=['baseline']
 (root/'logs').mkdir(exist_ok=True)
 for arm in arms:
  if arm in d['jobs']:continue
  cmd=[str(slurm/'sbatch'),'--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=1','--mem=16G','--time=01:30:00','--no-requeue','--job-name=sfPilot_'+arm,'--comment=EVSP_ADVISOR_SINGLE_FACTOR_PILOT_'+arm,'--output='+str(root/'logs'/('%j_'+arm+'.out')),'--error='+str(root/'logs'/('%j_'+arm+'.err')),'--chdir='+str(root),'--export=ALL',str(root/'worker.sh'),str(root),arm]
  d['intents'][arm]={'command':cmd,'created_epoch':time.time()};save(ledger,d)
  proc=subprocess.run(cmd,capture_output=True,text=True)
  if proc.returncode:raise RuntimeError(proc.stderr)
  job=proc.stdout.strip().split(';')[0];assert job.isdigit(),'ambiguous sbatch reply; reconcile retained intent'
  d['jobs'][arm]={'job_id':job,'command':cmd,'submitted_epoch':time.time()};del d['intents'][arm];save(ledger,d)
  sc=subprocess.check_output([str(slurm/'scontrol'),'show','job',job,'--oneliner'],text=True)
  assert 'ExcNodeList=scaglione-compute-01' in sc
  d['jobs'][arm]['scontrol_initial']=sc.strip();save(ledger,d)
 print(json.dumps(d))
