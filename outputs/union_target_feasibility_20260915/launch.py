"""Submit bounded native fixtures; production additionally requires root review artifact."""
from pathlib import Path
import argparse,fcntl,subprocess
import common as w
ROOT=Path(__file__).resolve().parent;S='/usr/local/slurm/slurm-25.05.5/bin/'
def launch(production=False):
 m=w.read(ROOT/'manifest.json');mh=w.sha(ROOT/'manifest.json');mode='production' if production else 'validation'
 if production:
  review=w.read(ROOT/'root_production_review.json')
  if review.get('approved') is not True or review.get('manifest_sha256')!=mh:raise ValueError('matching root production review required')
  validation=w.read(ROOT/'native_validation.json')
  if validation['status']!='passed' or validation['manifest_sha256']!=mh:raise ValueError('native validation must pass')
 with (ROOT/(mode+'_launch.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  jobsfile=ROOT/(mode+'_jobs.json')
  if jobsfile.exists() or (ROOT/(mode+'_submission_intent.json')).exists():raise ValueError('launch already attempted; inspect before retry')
  cases=[c for c in m['cases'].values() if c['is_validation']!=production]
  w.save(ROOT/(mode+'_submission_intent.json'),{'utc':w.now(),'manifest_sha256':mh,'cases':[c['id'] for c in cases]});jobs=[]
  for c in cases:
   name='utf_'+c['id'];cmd=[S+'sbatch','--parsable','--open-mode=append','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=24G','--time='+('02:00:00' if production else '00:30:00'),'--requeue','--job-name='+name,'--output='+str(ROOT/'logs'/(name+'_%j.out')),'--error='+str(ROOT/'logs'/(name+'_%j.err')),str(ROOT/'worker.sub'),str(ROOT),c['id'],'mip']
   jid=subprocess.check_output(cmd,text=True).strip().split(';')[0];j={'case_id':c['id'],'job_id':jid,'argv':cmd,'submitted_utc':w.now()};jobs.append(j);w.save(jobsfile,jobs)
   raw=subprocess.check_output([S+'scontrol','show','job',jid,'-o'],text=True);j['effective_scontrol_at_submission']=raw;w.save(jobsfile,jobs)
   x=dict(t.split('=',1) for t in raw.split() if '=' in t)
   assert x['Partition']=='default_partition' and x['ExcNodeList']=='scaglione-compute-01' and x['NumCPUs']=='8' and x['MinMemoryNode']=='24G' and x['Requeue']=='1'
  print(jobsfile)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--production',action='store_true');a=p.parse_args();launch(a.production)
