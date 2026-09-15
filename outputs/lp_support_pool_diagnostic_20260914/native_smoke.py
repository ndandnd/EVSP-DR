from pathlib import Path
import copy,json,shutil,subprocess
import worker as w
B=Path(__file__).resolve().parent;V=B.with_name(B.name+'_smoke');S='/usr/local/slurm/slurm-25.05.5/bin/'
def main():
 assert not V.exists();V.mkdir();(V/'cases').mkdir();(V/'logs').mkdir();m=w.read(B/'manifest.json');cases={}
 for cid in ['c3_k08_support_only','c3_k08_integer_positive_added','c3_k08_integer_matched_zero_added']:
  c=copy.deepcopy(m['cases'][cid]);c.update(is_validation=True,solver_budget_s=30,watchdog_s=2400,stage1_budget_s=15);c['argv'][c['argv'].index('--timelimit')+1]='30';c['argv'][c['argv'].index('--stage1-timelimit')+1]='15';cases[cid]=c
 for n in m['tooling_sha256']:shutil.copy2(B/n,V/n)
 vm=copy.deepcopy(m);vm.update(cases=cases,is_validation=True,production_manifest_sha256=w.sha(B/'manifest.json'));w.save(V/'manifest.json',vm);jobs=[]
 for cid in cases:
  argv=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=24G','--time=0:45:00','--no-requeue','--job-name=lpSmoke_'+cid,'--output='+str(V/'logs/%x_%j.out'),'--error='+str(V/'logs/%x_%j.err'),str(V/'worker.sub'),str(V),cid]
  w.save(V/'submission_intent.json',dict(status='submitting',case_id=cid,argv=argv));j=subprocess.check_output(argv,text=True,timeout=45).strip().split(';')[0];jobs.append(dict(case_id=cid,job_id=j,argv=argv));w.save(V/'jobs.json',jobs);w.save(V/'submission_intent.json',dict(status='recorded',case_id=cid,job_id=j))
 print(jobs)
if __name__=='__main__':main()
