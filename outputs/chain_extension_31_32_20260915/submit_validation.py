"""Submit the bounded native fixture only; no production jobs."""
import datetime
import subprocess
import campaign as c
assert c.read(c.B/'input_validation.json')['status']=='passed'
assert not (c.B/'validation_submission.json').exists()
args=[c.SLURM+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01',
 '--cpus-per-task=2','--mem=8G','--time=00:15:00','--no-requeue','--job-name=drX31_native_validate',
 '--output='+str(c.B/'logs/native_validation_%j.out'),'--error='+str(c.B/'logs/native_validation_%j.err'),str(c.B/'validate.sub')]
response=subprocess.run(args,check=True,capture_output=True,text=True)
job=response.stdout.strip().split(';')[0];assert job.isdigit()
c.save(c.B/'validation_submission.json',{'job_id':job,'argv':args,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'research_result':False})
state=subprocess.check_output([c.SLURM+'scontrol','show','job',job,'-o'],text=True)
assert 'ExcNodeList=scaglione-compute-01' in state and 'Partition=default_partition' in state
c.save(c.B/'validation_scheduler_initial.json',{'job_id':job,'scontrol':state})
print(job,flush=True)
