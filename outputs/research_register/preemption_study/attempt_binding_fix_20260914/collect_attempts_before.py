"""Run on Unicorn. Keep allocation attempts and samples; never alter jobs."""
from pathlib import Path
import subprocess,json,csv,io,datetime,hashlib,collections
ROOT=Path.home()/'ladder-lite'/'mip_preemption_study_20260911'
BIN='/usr/local/slurm/slurm-25.05.5/bin/'
def call(args):
 p=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
registry=json.loads((ROOT/'registry.json').read_text())
jobids=sorted(set(str(x['job_id']).split('_')[0] for x in registry['cases']))
fields='JobID,JobIDRaw,DBIndex,SLUID,JobName,Partition,QOS,Submit,Eligible,Start,End,ElapsedRaw,CPUTimeRAW,TotalCPU,AllocCPUS,ReqMem,NodeList,State,ExitCode,Reason,Restarts,Priority,TimelimitRaw'
r=call([BIN+'sacct','--array','-D','-X','-P','-u','nc437','--starttime=2026-09-10T23:00:00','-j',','.join(jobids),'--format='+fields]) if jobids else {'returncode':0,'stdout':'','stderr':''}
rows=list(csv.DictReader(io.StringIO(r['stdout']),delimiter='|')) if r['returncode']==0 else []
lookup={x['job_id']:x for x in registry['cases']}
accepted=[]
for row in rows:
 job=row['JobID']; case=lookup.get(job)
 if row['Submit'] < '2026-09-10T23:00:00': continue
 if case is None: continue # exclude array parent aggregates and other jobs
 result_path=Path(case['result_path']) if case.get('result_path') else None
 row['result_exists']=bool(result_path and result_path.exists())
 row['result_path']=str(result_path) if result_path else None
 if row['result_exists']:
  raw=result_path.read_bytes(); result=json.loads(raw);row['result_sha256']=hashlib.sha256(raw).hexdigest();row['reported_physical_replay_validated']=result.get('physical_replay_validated');row['reported_fleet']=result.get('buses')
 row['case_id']=case['case_id'];row['cohort']=case['cohort'];row['scientific_budget_s']=case.get('solver_budget_s')
 row['attempt_key']='|'.join(row.get(k,'') for k in ('SLUID','DBIndex','JobID','Submit','Start','Restarts'))
 state=row['State'].split()[0].split('+')[0]
 row['state_normalized']=state
 row['started']=row['Start'] not in ('','Unknown','None','N/A')
 try: row['eligible_to_start_seconds']=(datetime.datetime.fromisoformat(row['Start'])-datetime.datetime.fromisoformat(row['Eligible'])).total_seconds()
 except (ValueError,TypeError): row['eligible_to_start_seconds']=None
 row['confirmed_preemption']=state=='PREEMPTED'
 row['classification']=('preempted' if state=='PREEMPTED' else 'scheduler_completed' if state=='COMPLETED' else 'running_censored' if state in ('RUNNING','COMPLETING','SUSPENDED') else 'not_started' if not row['started'] else 'other_termination')
 accepted.append(row)
accepted=list({r['attempt_key']:r for r in accepted}.values())
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
summary={}
for cohort in sorted({r['cohort'] for r in registry['cases']}):
 a=[r for r in accepted if r['cohort']==cohort];ended=[r for r in a if r['started'] and r['classification'] not in ('running_censored','not_started')];pre=[r for r in ended if r['confirmed_preemption']]
 n=len(ended);k=len(pre)
 # Wilson interval is descriptive only; node/day correlations violate IID.
 z=1.959963984540054
 if n:
  p=k/n;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*((p*(1-p)/n+z*z/(4*n*n))**0.5)/den;ci=[max(0,center-half),min(1,center+half)]
 else: ci=None
 num=lambda r,key:int(r.get(key) or 0)
 summary[cohort]={'registered_tasks':sum(r['cohort']==cohort for r in registry['cases']),'accounting_missing_task_ids':[r['job_id'] for r in registry['cases'] if r['cohort']==cohort and r['job_id'] not in {x['JobID'] for x in a}],'attempts':len(a),'started':sum(r['started'] for r in a),'ended_started_attempts':n,'confirmed_preemptions':k,'preemption_fraction_among_ended':k/n if n else None,'descriptive_wilson95':ci,'state_counts':dict(collections.Counter(r['state_normalized'] for r in a)),'confirmed_preempted_allocation_seconds':sum(num(r,'ElapsedRaw') for r in pre),'confirmed_preempted_allocated_cpu_seconds':sum(num(r,'CPUTimeRAW') for r in pre),'allocation_exposure_seconds':sum(num(r,'ElapsedRaw') for r in a)}
priority=call([BIN+'sprio','-l','-j',','.join(jobids)]) if jobids else {}
config=call([BIN+'scontrol','show','config'])
config['stdout']='\n'.join(l for l in config['stdout'].splitlines() if any(k in l for k in ('Preempt','PriorityType','PriorityWeight','PriorityDecay')))
out={'timestamp_utc':now,'registry_sha256':hashlib.sha256((ROOT/'registry.json').read_bytes()).hexdigest(),'registry':registry,'accounting':r,'attempts':accepted,'summary':summary,'priority_sample':priority,'live_queue':call([BIN+'squeue','-r','-h','-j',','.join(jobids),'-o','%i|%T|%P|%Q|%N|%R']),'scheduler_config':config,'fair_share_sample':call([BIN+'sshare','-u','nc437','-P','-l']),'interpretation':'Scheduler COMPLETED is not a validated scientific result. PREEMPTED is the only automatically confirmed preemption state. Other cancellations, failures and timeouts remain separate. Running attempts are censored. Lost allocation time is not necessarily lost solver time or unusable incumbents. Observational cohorts differ; no causal partition comparison or stationary one-hour survival probability is claimed.'}
print(json.dumps(out))
