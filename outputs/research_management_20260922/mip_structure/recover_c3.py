from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent
script=r'''
from pathlib import Path
import json,subprocess,fcntl,time,os
root=Path('/home/nc437/ladder-lite/mip_structure_20260922');receipt=root/'c3_recovery.json';intent=root/'c3_recovery_SUBMIT_INTENT.json'
with (root/'c3_recovery.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if receipt.exists():d=json.loads(receipt.read_text())
 else:
  assert not intent.exists(),'Unresolved recovery submission intent'
  cmd=['sbatch','--parsable','--job-name=mstruct_c3_k15_fresh_prep_v2','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=32G','--time=02:00:00','--requeue','--open-mode=append','--output='+str(root/'slurm/%x-%j.out'),'--error='+str(root/'slurm/%x-%j.err'),str(root/'code_v2/worker_recovery.sh'),str(root),'prepare','c3_k15_fresh']
  intent.write_text(json.dumps({'argv':cmd,'created_unix':time.time()},indent=2)+'\n');r=subprocess.run(cmd,capture_output=True,text=True,check=True);job=r.stdout.strip().split(';')[0];assert job.isdigit();d={'replacement_preparation_job':job,'failed_preparation_job':'729457','diagnostic_job':'729731','argv':cmd,'repairs':[]};receipt.write_text(json.dumps(d,indent=2)+'\n')
 for job in ['729458','729459','729460','729461','729462']:
  if any(x['job']==job for x in d['repairs']):continue
  before=subprocess.check_output(['scontrol','show','job','-o',job],text=True);assert 'JobState=PENDING' in before and 'Reason=JobHeld' not in before
  r=subprocess.run(['scontrol','update','JobId='+job,'Dependency=afterok:'+d['replacement_preparation_job']],capture_output=True,text=True,check=True);after=subprocess.check_output(['scontrol','show','job','-o',job],text=True);assert 'afterok:'+d['replacement_preparation_job'] in after
  d['repairs'].append({'job':job,'before':before,'after':after,'stdout':r.stdout,'stderr':r.stderr});temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');os.replace(temp,receipt)
 d['replacement_scontrol']=subprocess.check_output(['scontrol','show','job','-o',d['replacement_preparation_job']],text=True);d['diagnostic_sacct']=subprocess.check_output(['sacct','-S','2026-09-22','-j','729731','--format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList','-P'],text=True);receipt.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn',"bash -lc 'python3 -'"],input=script,text=True,capture_output=True,check=True);(P/'c3_recovery.json').write_text(r.stdout);d=json.loads(r.stdout);print(json.dumps({'replacement':d['replacement_preparation_job'],'repaired':[x['job'] for x in d['repairs']],'scontrol':d['replacement_scontrol']}))
