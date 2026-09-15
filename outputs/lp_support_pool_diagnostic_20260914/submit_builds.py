from pathlib import Path
import fcntl,subprocess
import worker as w
B=Path(__file__).resolve().parent;S='/usr/local/slurm/slurm-25.05.5/bin/'
with (B/'build_launch.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);ix=w.read(B/'source_index.json');jobs=w.read(B/'build_jobs.json') if (B/'build_jobs.json').exists() else []
 if (B/'build_intent.json').exists():assert w.read(B/'build_intent.json')['status']=='recorded'
 health=subprocess.run([S+'squeue','-h','-u','nc437','-o','%i|%T'],capture_output=True,text=True,timeout=30);assert health.returncode==0;w.save(B/'build_health.json',{'utc':w.now(),'stdout':health.stdout})
 for cid in ix['groups']:
  if any(j['case_id']==cid for j in jobs):continue
  argv=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=2','--mem=8G','--time=1:00:00','--no-requeue','--job-name=lpPrep_'+cid,'--output='+str(B/'logs/%x_%j.out'),'--error='+str(B/'logs/%x_%j.err'),'--wrap','/home/nc437/evsp_env/bin/python '+str(B/'construct.py')+' '+cid]
  w.save(B/'build_intent.json',dict(status='submitting',case_id=cid,argv=argv));job=subprocess.check_output(argv,text=True,timeout=45).strip().split(';')[0];assert job.isdigit();jobs.append(dict(case_id=cid,job_id=job,argv=argv));w.save(B/'build_jobs.json',jobs);w.save(B/'build_intent.json',dict(status='recorded',case_id=cid,job_id=job))
 print(jobs)
