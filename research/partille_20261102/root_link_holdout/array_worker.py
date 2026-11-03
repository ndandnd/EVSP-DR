"""Launch one arm in an immutable, non-reused attempt directory; never submit retries."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
state=json.loads((a.root/'submission_state.json').read_text())
assert state['authorization']['status']=='approved'
row=state['task_map'][a.index];assert row['index']==a.index
assert 0<=a.index<10
attempt='job'+os.environ['SLURM_ARRAY_JOB_ID']+'_'+str(a.index)+'_r'+os.environ.get('SLURM_RESTART_COUNT','0')
out=a.root/'results'/row['case']/row['arm']/attempt
out.mkdir(parents=True,exist_ok=False)
code=a.root/'code';manifest=a.root/'manifests'/(row['case']+'.json')
for rel,h in state['artifact_sha256'].items():assert sha(a.root/rel)==h,rel
receipt={'schema':'heldout-root-arm-attempt-v1','case':row['case'],'arm':row['arm'],'task_index':a.index,'attempt':attempt,
 'manifest_sha256':sha(manifest),'registration_commit':state['authorization']['registration_commit'],
 'array_job_id':os.environ['SLURM_ARRAY_JOB_ID'],'slurm_job_id':os.environ['SLURM_JOB_ID'],
 'restart_count':int(os.environ.get('SLURM_RESTART_COUNT','0')),'cpus':int(os.environ['SLURM_CPUS_PER_TASK']),
 'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','retry_owner':state['retry_policy']['owner']}
dump(out/'wrapper_receipt.json',receipt)
script='run_recombination.py' if row['arm']=='repair' else 'root_control.py'
argv=[sys.executable,'-u',str(code/script),'--manifest',str(manifest),'--output',str(out/'solve'),'--start-fee','5']
if row['arm']=='repair':argv+=['--retime-seconds','180']
tick=time.monotonic()
with (out/'stdout.private.log').open('w') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,cwd=code)
receipt.update(status='terminal',exit_code=r.returncode,arm_wall_seconds=time.monotonic()-tick,
 ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
rp=out/'solve/receipt.json'
if rp.exists():
 rr=json.loads(rp.read_text());receipt.update(scientific_receipt_sha256=sha(rp),scientific_outcome=rr.get('outcome'),validated=rr.get('validated',False))
dump(out/'wrapper_receipt.json',receipt)
raise SystemExit(r.returncode)
