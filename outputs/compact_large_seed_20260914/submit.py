"""Locked, intent-journaled submission; uncertain sbatch never retried blindly."""
from pathlib import Path
import argparse,fcntl,json,math,subprocess
import worker as w
S='/usr/local/slurm/slurm-25.05.5/bin/'
def main(root):
 b=Path(root)
 with (b/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  m=w.read(b/'manifest.json');v=w.read(b/'validation.json')
  assert v['status']=='passed' and v['manifest_sha256']==w.sha(b/'manifest.json')
  for n,h in m['tooling_sha256'].items():w.require_hash(b/n,h)
  # Exactly one bounded scheduler health gate per invocation, before any mutation.
  health=subprocess.run([S+'squeue','-h','-u','nc437','-o','%i|%T'],capture_output=True,text=True,timeout=30)
  w.save(b/'scheduler_health.json',dict(returncode=health.returncode,stdout=health.stdout,stderr=health.stderr,utc=w.now()))
  assert health.returncode==0,'Scheduler health gate failed; no submissions'
  jobs=w.read(b/'jobs.json') if (b/'jobs.json').exists() else [];done={x['case_id']:x['job_id'] for x in jobs}
  if (b/'submission_intent.json').exists():assert w.read(b/'submission_intent.json')['status']=='recorded','Reconcile uncertain sbatch first'
  for cid,c in sorted(m['cases'].items(),key=lambda x:(x[1]['kind']!='cg',x[0])):
   if cid in done:continue
   deps=[done[c['source_case']]] if c.get('source_case') else []
   mins=math.ceil(c['resources']['allocation_s']/60)
   argv=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task='+str(c['resources']['cpus']),'--mem='+c['resources']['mem'],'--time='+f'{mins//60}:{mins%60:02d}:00','--requeue','--job-name=drLargeSeed_'+cid,'--output='+str(b/'logs/%x_%j.out'),'--error='+str(b/'logs/%x_%j.err')]
   if deps:argv+=['--dependency=afterok:'+':'.join(deps),'--kill-on-invalid-dep=yes']
   argv +=[str(b/'worker.sub'),str(b),cid]
   intent=dict(case_id=cid,argv=argv,status='submitting',utc=w.now());w.save(b/'submission_intent.json',intent)
   result=subprocess.run(argv,capture_output=True,text=True,timeout=45,check=True);job=result.stdout.strip().split(';')[0];assert job.isdigit()
   jobs.append(dict(case_id=cid,job_id=job,kind=c['kind'],dependencies=deps,argv=argv,submitted_utc=w.now()));done[cid]=job
   w.save(b/'jobs.json',jobs);w.save(b/'case_jobs.json',done);intent.update(status='recorded',job_id=job);w.save(b/'submission_intent.json',intent)
  print(json.dumps(dict(submitted=len(jobs),independent=sum(not x['dependencies'] for x in jobs))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
