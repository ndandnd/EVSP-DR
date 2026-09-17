"""One authorized full40 CG replacement; preserve graph and downstream MIP hold."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess
base=Path('/home/nc437/ladder-lite/review_full40_20260916');version=base/'scaglione_12h_v1';slurm=Path('/usr/local/slurm/slurm-25.05.5/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def show(job):return subprocess.check_output([str(slurm/'scontrol'),'show','job',job,'-o'],text=True).strip()
def save(value):
 p=version/'submission_receipt.json';tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
with (version/'submission.lock').open('w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert not (version/'submission_receipt.json').exists(),'Submission receipt exists: inspect it, do not duplicate'
 deployment=json.loads((version/'deployment_receipt.json').read_text());m=json.loads((version/'manifest.json').read_text())
 for name,h in deployment['files_sha256'].items():assert sha(version/name)==h
 for name,h in m['tooling_sha256'].items():assert sha(base/name)==h
 old=show('341405');mip=show('341406');graph=show('341404_0')
 assert 'Reason=JobHeldUser' in old and 'Dependency=afterok:341404_0' in old
 assert 'Reason=JobHeldUser' in mip and 'Dependency=afterok:341405' in mip
 cmd=[str(slurm/'sbatch'),'--parsable','--partition=scaglione','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem=120G','--time=12:00:00','--requeue','--kill-on-invalid-dep=yes','--dependency=afterok:341404_0','--job-name=dr40_12hCG','--comment=EVSP_FULL40_12H_SCAGLIONE_20260916','--chdir='+str(base),'--output='+str(base/'logs/dr40_12hCG_%j.out'),'--error='+str(base/'logs/dr40_12hCG_%j.err'),str(version/'worker.sub'),'cg','c1_full40']
 receipt={'authorized_action':'replace full40 CG only; exact12h scheduler allocation on Scaglione; retain held downstreamMIP','manifest_sha256':sha(version/'manifest.json'),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'before':{'341405':old,'341406':mip,'341404_0':graph},'submission_intent':cmd,'scientific_cg_s':42300,'scheduler_wall_s':43200,'memory_GiB':120,'cpus':8,'state':'submission_intent_saved'}
 save(receipt)
 result=subprocess.run(cmd,text=True,capture_output=True)
 receipt.update(sbatch_returncode=result.returncode,sbatch_stdout=result.stdout,sbatch_stderr=result.stderr);save(receipt)
 result.check_returncode();job=result.stdout.strip().split(';')[0];assert job.isdigit();receipt['replacement_job_id']=job;receipt['state']='submitted';save(receipt)
 state=show(job);receipt['replacement_scontrol']=state;save(receipt)
 for token in ['Partition=scaglione','ExcNodeList=scaglione-compute-01','TimeLimit=12:00:00','MinMemoryNode=120G','NumCPUs=8','Dependency=afterok:341404_0']:
  assert token in state,(token,state)
 # Rewire before cancelling the old prerequisite: KillOnInvalidDep must not
 # cancel the intentionally held downstream MIP when its old CG is retired.
 receipt['mip_rewire_intent']={'job':'341406','dependency':'afterok:'+job,'hold_preserved':True};save(receipt)
 subprocess.run([str(slurm/'scontrol'),'update','JobId=341406','Dependency=afterok:'+job],check=True,capture_output=True,text=True)
 after_mip=show('341406');assert 'Reason=JobHeldUser' in after_mip and 'Priority=0 ' in after_mip and ('Dependency=afterok:'+job) in after_mip
 receipt['mip_after_rewire']=after_mip;receipt['state']='submitted_mip_rewired_still_held';save(receipt)
 receipt['superseded_cg_cancel_intent']='341405';save(receipt)
 subprocess.run([str(slurm/'scancel'),'341405'],check=True,capture_output=True,text=True)
 receipt['superseded_cg_after']=show('341405');assert 'JobState=CANCELLED' in receipt['superseded_cg_after']
 receipt['graph_after']=show('341404_0');receipt['replacement_after']=show(job);receipt['mip_after']=show('341406');receipt['state']='complete';receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(receipt)
 print(json.dumps(receipt,indent=2))
