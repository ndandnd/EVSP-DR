#!/usr/bin/env python3
from pathlib import Path
import datetime as dt,hashlib,importlib,os,json,subprocess,sys
r=Path('/home/nc437/ladder-lite/giro_zero_start_fee_20260913');plan=json.loads((r/'plan.json').read_text());code=Path(plan['execution_repo']);sys.path.insert(0,str(code/'scripts/event_uniform_envelope'));import launch_giro_zero_fee as launch
oldraw=(r/'jobs.json').read_bytes();old=json.loads(oldraw);assert len(old['jobs'])==12 and not (r/'jobs.initial_failed.json').exists()
assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==plan['commit']
worker=r/'worker_recovery.sh';assert 'CODE_ROOT='+str(code) in worker.read_text();assert 'dirname' not in worker.read_text()
records=[];slurm='/usr/local/slurm/slurm-25.05.5/bin/'
for job in old['jobs']:
 if job['stage']!='mip':continue
 sc=subprocess.check_output([slurm+'scontrol','show','job',job['job_id']],text=True);assert 'JobState=PENDING' in sc
 subprocess.run([slurm+'scancel',job['job_id']],check=True)
 records.append({'job_id':job['job_id'],'disposition':'cancelled_before_start; upstream worker path failed before solver','scontrol_before':sc})
(r/'jobs.initial_failed.json').write_bytes(oldraw)
(r/'cancelled_invalid_mips.json').write_text(json.dumps(records,indent=2)+'\n')
ledger={'schema':'evsp-dr-terminal-energy-fee-submission-v1','started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'plan_sha256':hashlib.sha256((r/'plan.json').read_bytes()).hexdigest(),'worker_sha256':hashlib.sha256(worker.read_bytes()).hexdigest(),'supersedes_initial_jobs_sha256':hashlib.sha256(oldraw).hexdigest(),'reason':'Explicit code path survives Slurm spool copying; numerical code, data, fee, budgets and physics unchanged.','jobs':[]}
def save():
 p=r/'jobs.recovery.tmp';p.write_text(json.dumps(ledger,indent=2)+'\n');os.replace(p,r/'jobs.json')
save();cells={c['id']:c for c in plan['cells']}
for tariff in launch.TARIFFS:
 parents=[]
 for fee in launch.FEES:
  cell=cells[f'{tariff}_fee{int(fee)}'];cmd=launch.sbatch_frontier(plan,cell,worker);jid=subprocess.check_output(cmd,text=True).strip().split(';')[0];parents.append(jid);ledger['jobs'].append({'pair_id':cell['id'],'stage':'frontier','job_id':jid,'command':cmd});save()
 for fee in launch.FEES:
  cell=cells[f'{tariff}_fee{int(fee)}'];cmd=launch.sbatch_mip(plan,cell,worker,parents);jid=subprocess.check_output(cmd,text=True).strip().split(';')[0];ledger['jobs'].append({'pair_id':cell['id'],'stage':'mip','job_id':jid,'dependency':parents,'command':cmd});save()
for job in ledger['jobs']:
 sc=subprocess.check_output([slurm+'scontrol','show','job',job['job_id']],text=True);assert 'ExcNodeList=scaglione-compute-01' in sc and 'Partition=default_partition' in sc
 if job['stage']=='mip':assert all('afterok:'+x in sc for x in job['dependency'])
 job['scontrol']=sc;save()
ledger['finished_utc']=dt.datetime.now(dt.timezone.utc).isoformat();save();print(json.dumps({'jobs':[{k:j[k] for k in ['job_id','pair_id','stage']} for j in ledger['jobs']]}))
