"""Scoped append-only snapshots of original and recovery attempts, with sacct fallback."""
from pathlib import Path
import subprocess,json,datetime,re
P=Path(__file__).resolve().parent
original=json.loads((P/'jobs.json').read_text());records={k:{**v,'role':'original'} for k,v in original.items()}
v2=json.loads((P/'c3_recovery.json').read_text());records['c3_k15_fresh__prepare_v2']={'job_id':v2['replacement_preparation_job'],'role':'superseded_failed_preparation','scontrol_at_submission':v2['replacement_scontrol']}
records['cost_diagnostic']={'job_id':v2['diagnostic_job'],'role':'diagnostic'}
recovery=json.loads((P/'recovery_v3.json').read_text()) if (P/'recovery_v3.json').exists() else None
if recovery:
 for k,v in recovery['jobs'].items():records[k+'__v3']={**v,'role':'active_recovery'}
records['loader_benchmark']={'job_id':(P/'loader_benchmark_v3_job.txt').read_text().strip(),'role':'diagnostic'}
script='jobs='+repr({k:v['job_id'] for k,v in records.items()})+r'''
from pathlib import Path
import json,subprocess
root=Path('/home/nc437/ladder-lite/mip_structure_20260922');out={'jobs':{},'attempts':{}}
for k,j in jobs.items():
 r=subprocess.run(['scontrol','show','job','-o',j],capture_output=True,text=True);out['jobs'][k]={'id':j,'scontrol':r.stdout.strip(),'scontrol_stderr':r.stderr}
out['sacct']=subprocess.check_output(['sacct','-S','2026-09-22','-j',','.join(jobs.values()),'--format=JobID,State,ExitCode,ElapsedRaw,Start,End,MaxRSS,AllocCPUS,ReqMem,NodeList','-P'],text=True)
for p in root.glob('results/*/*/*'):
 d={n:json.loads((p/n).read_text()) for n in ['execution.json','COMPLETE.json','result.json','phase_loading.json','phase_loaded.json','phase_model_ready.json'] if (p/n).exists()};d['gurobi_log_exists']=(p/'gurobi.log').exists()
 if d['gurobi_log_exists']:d['gurobi_log_tail']=(p/'gurobi.log').read_text()[-2200:]
 out['attempts'][str(p)]=d
out['prepared_gates']={p.stem:json.loads(p.read_text()) for p in (root/'prepared').glob('*.json')}
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn',"bash -lc 'python3 -'"],input=script,text=True,capture_output=True,check=True);d=json.loads(r.stdout)
lines=d['sacct'].splitlines();acct={x.split('|')[0]:dict(zip(lines[0].split('|'),x.split('|'))) for x in lines[1:]}
def field(s,k):
 m=re.search(r'(?:^| )'+re.escape(k)+r'=(\S+)',s);return m.group(1) if m else None
checks=[]
for k,x in d['jobs'].items():
 rec=records[k];s=x['scontrol'];a=acct.get(x['id'],{});x.update(role=rec['role'],accounting=a,state=field(s,'JobState') or a.get('State'),state_source='scontrol' if s else 'sacct');argv=rec.get('argv',[]);historical=rec.get('scontrol_at_submission','')
 resource=s or historical
 if argv:
  settings=all(v in argv for v in ['--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=32G','--time=02:00:00'])
  source='recorded_submission_argv'
 elif resource:
  diagnostic=rec['role']=='diagnostic';settings=field(resource,'ExcNodeList')=='scaglione-compute-01' and field(resource,'Partition')=='default_partition' and field(resource,'NumCPUs')==('1' if diagnostic else '8');source='live_or_preserved_scontrol'
 else:settings=None;source='diagnostic completion retained separately'
 depok=True;expected=None
 if rec['role']=='active_recovery' and '__prepare' not in k:
  case=k.split('__')[0]
  if case=='c3_k15_fresh':
   expected=recovery['jobs']['c3_k15_fresh__prepare']['job_id'];depok='--dependency=afterok:'+expected in argv and (not s or 'afterok:'+expected in s or case in d['prepared_gates'])
  else:depok=case in d['prepared_gates']
 checks.append({'key':k,'job_id':x['id'],'state':x['state'],'role':rec['role'],'settings_passed':settings,'settings_evidence':source,'dependency_valid':depok,'expected_preparation':expected})
d['checks']=checks;d['checks_passed']=all(c['settings_passed'] is not False and c['dependency_valid'] for c in checks);d['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();dest=P/'snapshots'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ.json');dest.parent.mkdir(exist_ok=True);dest.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps({'snapshot':str(dest),'passed':d['checks_passed'],'checks':checks,'completed_trial_artifacts':sum('result.json' in a and '/prepare/' not in p for p,a in d['attempts'].items())},indent=2))
