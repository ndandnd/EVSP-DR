import hashlib,json,subprocess,datetime
from pathlib import Path
base=Path('/home/nc437/ladder-lite/w2_k14_import_fix_20260912')
assert not (base/'submission.json').exists()
(base/'logs').mkdir(exist_ok=True)
# Parent is completed and its immutable inputs are frozen; Slurm has purged its
# controller record, so an afterok edge cannot be created for this historical ID.
parent_state=subprocess.check_output(['/usr/local/slurm/slurm-25.05.5/bin/sacct','-j','949701','--format=JobID,State,ExitCode','-P'],text=True)
assert '949701|COMPLETED|0:0' in parent_state
manifest=json.loads((base/'manifest.json').read_text())
assert manifest['source_parent_status_sha256']=='9d4487510dd56ec09edfb646584530ba4795bfb0db1994ffb3bafda3164486d1'
manifest['dependencies']['scheduler_dependency']=None
manifest['dependencies']['satisfied_parent_accounting']=parent_state
manifest['dependencies']['reason_no_scheduler_edge']='Parent completed successfully and inputs frozen; completed job purged from Slurm controller, afterok rejected (test-only: Job dependency problem).'
(base/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(base/'initial_submission_rejected.json').write_text(json.dumps({'job_created':False,'attempted_dependency':'afterok:949701','test_only_error':'allocation failure: Job dependency problem','parent_sacct':parent_state,'scontrol_error':'Invalid job id specified','squeue_matching_jobs':[]},indent=2)+'\n')

argv=['/usr/local/slurm/slurm-25.05.5/bin/sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=96G','--time=04:45:00','--no-requeue','--job-name=w2k14_importfix','--output='+str(base/'logs/%x_%j.out'),'--error='+str(base/'logs/%x_%j.err'),str(base/'worker.sub')]
result=subprocess.run(argv,capture_output=True,text=True,check=True)
job=result.stdout.strip().split(';')[0];assert job.isdigit()
record={'job_id':job,'argv':argv,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),'worker_sub_sha256':hashlib.sha256((base/'worker.sub').read_bytes()).hexdigest(),'prepare_script_sha256':hashlib.sha256((base/'prepare_remote.py').read_bytes()).hexdigest(),'sbatch_stdout':result.stdout,'sbatch_stderr':result.stderr}
(base/'submission.json').write_text(json.dumps(record,indent=2)+'\n')
state=subprocess.check_output(['/usr/local/slurm/slurm-25.05.5/bin/scontrol','show','job',job],text=True)
(base/'submission_verification.txt').write_text(state)
assert 'ExcNodeList=scaglione-compute-01' in state
assert 'Partition=default_partition' in state
print(json.dumps(record));print(state)
