"""Submit a validated immutable independent-MIP manifest exactly once per case."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,math,os,subprocess
S='/usr/local/slurm/slurm-25.05.5/bin/'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with t.open('w') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def main(root):
 b=Path(root)
 with (b/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  m=read(b/'manifest.json');v=read(b/'validation.json')
  assert v['status']=='passed' and v['manifest_sha256']==sha(b/'manifest.json')
  for n,h in m['tooling_sha256'].items():assert sha(b/n)==h
  jobs=read(b/'jobs.json') if (b/'jobs.json').exists() else [];done={x['case_id']:x['job_id'] for x in jobs}
  if (b/'submission_intent.json').exists():assert read(b/'submission_intent.json')['status']=='recorded','Reconcile uncertain sbatch before retrying'
  for cid,c in sorted(m['cases'].items(), key=lambda item: item[0]):
   if cid in done:continue
   assert c['arm'] in ('cg','fixed')
   mins=math.ceil(c['resources']['allocation_s']/60)
   argv=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=32G',
    '--time='+f'{mins//60}:{mins%60:02d}:00','--requeue','--job-name=dr3m_'+cid,'--output='+str(b/'logs/%x_%j.out'),
    '--error='+str(b/'logs/%x_%j.err'),str(b/'worker.sub'),str(b),cid]
   intent=dict(case_id=cid,argv=argv,status='submitting',utc=now());save(b/'submission_intent.json',intent)
   job=subprocess.check_output(argv,text=True).strip().split(';')[0];assert job.isdigit()
   jobs.append(dict(case_id=cid,job_id=job,kind=c['arm'],treatment='fresh_minimum_charge_DR',dependencies=[],argv=argv,submitted_utc=now()))
   done[cid]=job;save(b/'jobs.json',jobs);save(b/'case_jobs.json',done)
   intent.update(status='recorded',job_id=job);save(b/'submission_intent.json',intent)
  rows=[]
  for j in jobs:
   raw=subprocess.check_output([S+'scontrol','show','job',j['job_id'],'-o'],text=True)
   x=dict(t.split('=',1) for t in raw.split() if '=' in t)
   assert x['Partition']=='default_partition' and x['ExcNodeList']=='scaglione-compute-01' and x['NumCPUs']=='8'
   assert x['Dependency'] in ('(null)','(none)','')
   rows.append(dict(case_id=j['case_id'],job_id=j['job_id'],state=x['JobState'],reason=x.get('Reason'),nodes=x.get('NodeList'),raw=raw))
  counts={s:sum(x['state']==s for x in rows) for s in sorted({x['state'] for x in rows})}
  save(b/'scheduler_verification.json',dict(status='passed',collected_utc=now(),counts=counts,rows=rows,all_independent=True))
  print(json.dumps({'submitted':len(jobs),'counts':counts}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();main(a.root)
