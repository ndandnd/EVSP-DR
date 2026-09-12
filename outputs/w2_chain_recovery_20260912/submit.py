from pathlib import Path
import json,subprocess,datetime,hashlib
B=Path('/home/nc437/ladder-lite/w2_chain_recovery_20260912');SL='/usr/local/slurm/slurm-25.05.5/bin/';records=[]
assert not (B/'submission.json').exists()
acct=subprocess.check_output([SL+'sacct','-S','2026-09-12','-j','15687','--format=JobID,State,ExitCode','-P'],text=True);assert '15687|COMPLETED|0:0' in acct
m=json.loads((B/'manifest.json').read_text())
for r in m['files']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
def submit(name,manifest,mem,wall,dependency=None):
 a=[SL+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem='+mem,'--time='+wall,'--no-requeue','--job-name='+name,'--output='+str(B/'logs/%x_%j.out'),'--error='+str(B/'logs/%x_%j.err')]
 if dependency:a+=['--dependency=afterok:'+dependency]
 a += [str(B/'worker.sub'),str(B/manifest)]
 p=subprocess.run(a,capture_output=True,text=True,check=True);job=p.stdout.strip().split(';')[0];assert job.isdigit()
 r={'job_id':job,'name':name,'manifest':str(B/manifest),'manifest_sha256':hashlib.sha256((B/manifest).read_bytes()).hexdigest(),'argv':a,'dependency':dependency,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};records.append(r);(B/'submission.json').write_text(json.dumps(records,indent=2)+'\n')
 state=subprocess.check_output([SL+'scontrol','show','job',job],text=True);assert 'ExcNodeList=scaglione-compute-01' in state and 'Partition=default_partition' in state;(B/f'scontrol_{job}.txt').write_text(state);return job
m14=submit('rec2k14M','mip14_manifest.json','24G','02:00:00');c15=submit('rec2k15CG','cg15_manifest.json','96G','04:45:00');m15=submit('rec2k15M','mip15_manifest.json','24G','02:00:00',c15)
p=Path('/home/nc437/ladder-lite/mip_preemption_study_20260911/registry.json');v=json.loads(p.read_text());before=hashlib.sha256(p.read_bytes()).hexdigest();known={x['job_id'] for x in v['cases']}
for k,j in [(14,m14),(15,m15)]:
 assert j not in known
 v['cases'].append({'job_id':j,'case_id':f'w2_k{k}_recovery','cohort':'default_chain2_recovery_3600','solver_budget_s':3600,'result_path':str(B/f'cases/w2_k{k}/mip/onehour/result.json'),'attempt_tag':'chain2_recovery_a01','requeue':False})
t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
(B/'preemption_registration.json').write_text(json.dumps({'registry':str(p),'before_sha256':before,'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'added_job_ids':[m14,m15]},indent=2)+'\n')
print(json.dumps({'jobs':records,'states':{r['job_id']:(B/f"scontrol_{r['job_id']}.txt").read_text() for r in records}}))
