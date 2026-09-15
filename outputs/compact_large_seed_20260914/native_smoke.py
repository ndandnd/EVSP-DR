"""Create isolated native two-arm CG+own-MIP compatibility validation."""
from pathlib import Path
import copy,json,shutil,subprocess
import worker as w
B=Path(__file__).resolve().parent;S='/usr/local/slurm/slurm-25.05.5/bin/'
def main():
 m=w.read(B/'manifest.json');b=B.parent/(B.name+'_smoke');b.mkdir();(b/'logs').mkdir()
 for n in ['worker.py','worker.sub']:shutil.copy2(B/n,b/n)
 sm=copy.deepcopy(m);sm['cases']={}
 for arm in ['core','core512']:
  cid='c5_k25_'+arm
  for key in [cid,cid+'_mip']:
   c=copy.deepcopy(m['cases'][key]);c['is_validation']=True;c['watchdog_s']=2400;c['resources']['allocation_s']=2700
   if c['kind']=='cg':
    c['argv'][c['argv'].index('--max-iters')+1]='1';c['argv'][c['argv'].index('--wall-limit-s')+1]='1800';c['solver_budget_s']=1800
   else:
    c['argv'][c['argv'].index('--timelimit')+1]='30';c['argv'][c['argv'].index('--stage1-timelimit')+1]='20';c['solver_budget_s']=30
   sm['cases'][key]=c
 w.save(b/'manifest.json',sm);jobs={}
 for cid,c in sorted(sm['cases'].items(),key=lambda x:(x[1]['kind']!='cg',x[0])):
  args=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem='+c['resources']['mem'],'--time=00:45:00','--no-requeue','--job-name=compact_smoke_'+cid,'--output='+str(b/'logs/%x_%j.out'),'--error='+str(b/'logs/%x_%j.err')]
  if c.get('source_case'):args+=['--dependency=afterok:'+jobs[c['source_case']],'--kill-on-invalid-dep=yes']
  args +=[str(b/'worker.sub'),str(b),cid]
  w.save(b/'submission_intent.json',dict(case_id=cid,argv=args,status='submitting'))
  job=subprocess.check_output(args,text=True,timeout=45).strip().split(';')[0];assert job.isdigit();jobs[cid]=job;w.save(b/'jobs.json',jobs);w.save(b/'submission_intent.json',dict(case_id=cid,job_id=job,status='recorded'))
 print(json.dumps(jobs))
if __name__=='__main__':main()
