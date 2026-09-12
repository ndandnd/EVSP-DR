from pathlib import Path
import subprocess,json,datetime,hashlib
B=Path('/home/nc437/ladder-lite/queue_recovery_20260912');S='/usr/local/slurm/slurm-25.05.5/bin/'
v=json.loads((B/'ready_mips_manifest.json').read_text());assert not(B/'ready_mips_submission.json').exists()
args=[S+'sbatch','--parsable','--array='+','.join(str(x['index']) for x in v['cases'])+'%50','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=24G','--time=02:00:00','--requeue','--job-name=dr_readyM','--output='+str(B/'logs/%x_%A_%a.out'),'--error='+str(B/'logs/%x_%A_%a.err'),str(B/'ready_mip_worker.sub')]
r=subprocess.run(args,capture_output=True,text=True,check=True);job=r.stdout.strip().split(';')[0];assert job.isdigit()
record={'job_id':job,'argv':args,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256((B/'ready_mips_manifest.json').read_bytes()).hexdigest(),'task_ids':[f"{job}_{x['index']}" for x in v['cases']],'replacement_map':{x['replaces_pending_job']:f"{job}_{x['index']}" for x in v['cases']}}
(B/'ready_mips_submission.json').write_text(json.dumps(record,indent=2)+'\n')
state=subprocess.check_output([S+'scontrol','show','job',job],text=True);assert 'ExcNodeList=scaglione-compute-01' in state and 'Partition=default_partition' in state;(B/'ready_mips_scontrol.txt').write_text(state)
p=Path('/home/nc437/ladder-lite/mip_preemption_study_20260911/registry.json');registry=json.loads(p.read_text())
for c in v['cases']:registry['cases'].append({'job_id':f"{job}_{c['index']}",'case_id':c['case_id'],'cohort':'default_ready_pool_recovery_3600','solver_budget_s':3600,'result_path':c['result_path'],'attempt_tag':'ready_pool_dependency_recovery_a01','parent_job_id':c['parent_job_id'],'replaces_pending_job':c['replaces_pending_job'],'requeue':True})
t=p.with_suffix('.tmp');t.write_text(json.dumps(registry,indent=2)+'\n');t.replace(p)
# Original entries never optimized and cannot launch while their dependencies are failed.
# Remove only those nine exact pending entries after replacement IDs are durably recorded.
cancel=[]
for c in v['cases']:
 j=c['replaces_pending_job'];st=subprocess.check_output([S+'scontrol','show','job',j,'-o'],text=True)
 assert 'JobState=PENDING' in st and 'DependencyNeverSatisfied' in st
 rr=subprocess.run([S+'scancel',j],capture_output=True,text=True);cancel.append({'job_id':j,'returncode':rr.returncode,'stderr':rr.stderr,'replacement':record['replacement_map'][j]})
(B/'ready_mips_old_cancellations.json').write_text(json.dumps(cancel,indent=2)+'\n')
print(json.dumps(record))
