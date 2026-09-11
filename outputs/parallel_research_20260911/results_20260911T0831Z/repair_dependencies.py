"""Repair only pending corresponding-case dependencies; preserve jobs and solver settings."""
import subprocess,json,datetime,hashlib
from pathlib import Path
root=Path('/home/nc437/ladder-lite/covering_complement75_20260911_21fbecb')
bin='/usr/local/slurm/slurm-25.05.5/bin/'
def run(*args):
 r=subprocess.run(args,capture_output=True,text=True);return {'command':list(args),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
def require(r):
 if r['returncode']:raise RuntimeError(r)
 return r['stdout']
account=run(bin+'sacct','-S','2026-09-10T23:00:00','-u','nc437','-X','-n','-P','-j','810454','-o','JobID,State,ExitCode,Elapsed')
states={a[0]:a for x in require(account).splitlines() if len(a:=x.split('|'))>=4}
q=run(bin+'squeue','--me','-r','-h','-o','%i|%T|%R')
matrix=[x.split('\t') for x in (root/'matrix.tsv').read_text().splitlines()]
audit={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reason':'Corresponding CG tasks are COMPLETED, but aftercorr freeze dependencies remain unfulfilled. Explicitly discharge verified completed predecessors; use per-task afterok for unfinished prerequisites. No scientific changes or new jobs.','accounting':account,'queue_before':q,'changes':[]}
p=root/'downstream/dependency_repair_20260911T0831Z.json'
if p.exists():
 audit['prior_pass']=json.loads(p.read_text())
def save():p.write_text(json.dumps(audit,indent=2)+'\n')
save()
for line in sorted(require(q).splitlines(),key=lambda x:(not x.startswith('810587_'),x)):
 job,state,reason=line.split('|',2)
 if state!='PENDING' or job.split('_')[0] not in ['810587','812766','812767']:continue
 idx=int(job.split('_')[1]); row=matrix[idx]; assert int(row[0])==idx
 before=run(bin+'scontrol','show','job',job);s=require(before)
 if 'JobState=PENDING' not in s:continue
 assert 'scaglione-compute-01' in s
 record={'job':job,'case':row[1],'before':before}
 if job.startswith('810587_'):
  predecessor=f'810454_{idx}';st=states[predecessor]
  if st[1]=='COMPLETED' and st[2]=='0:0':
   source=root/'cg'/f'M__{row[1]}__cover__{row[7]}.json';raw=source.read_bytes();d=json.loads(raw)
   assert d.get('stop_reason') and isinstance(d.get('final'),dict)
   journal=Path(str(source)+'.columns.jsonl');assert journal.is_file() and journal.stat().st_size>0
   output=root/'snapshots'/source.name;assert not output.exists()
   record['prerequisite']={'job':predecessor,'state':st,'source':str(source),'sha256':hashlib.sha256(raw).hexdigest(),'stop_reason':d['stop_reason'],'journal_bytes':journal.stat().st_size}
   dep=''
  elif st[1] in ['RUNNING','PENDING']:dep='afterok:'+predecessor
  else:continue
 else:dep=f'afterok:810587_{idx}'
 if ('Dependency='+dep+'(' in s) if dep else ('Dependency=(null)' in s):continue
 record['new_dependency']=dep;record['update']=run(bin+'scontrol','update','JobId='+job,'Dependency='+dep);audit['changes'].append(record);save()
 record['after']=run(bin+'scontrol','show','job',job);save()
 if record['update']['returncode'] and ('Dependency='+dep+'(' not in record['after']['stdout']): require(record['update'])
audit['queue_after']=run(bin+'squeue','--me','-r','-h','-o','%i|%T|%R');save();print(json.dumps({'changes':len(audit['changes']),'record':str(p)}))
