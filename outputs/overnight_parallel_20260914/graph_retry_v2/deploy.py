"""Reviewed conditional gate deployment; append-only audit for every mutation."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess
import gate
import process_worker as w
S='/usr/local/slurm/slurm-25.05.5/bin/'
def fields(job):
 raw=subprocess.check_output([S+'scontrol','show','job',job,'-o'],text=True,timeout=20)
 return dict(x.split('=',1) for x in raw.split() if '=' in x),raw

def main(root):
 b=Path(root)
 with (b/'deployment.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (b/'deployment_ledger.jsonl').exists():raise ValueError('Existing deployment ledger: reconcile it; no blind retry')
  m=w.read(b/'manifest.json');native=w.read(b/'validation.json')
  assert native['status']=='passed' and native['manifest_sha256']==w.sha(b/'manifest.json')
  for n,h in m['tooling_sha256'].items():w.require_hash(b/n,h)
  w.require_hash(m['original_manifest_path'],m['original_manifest_sha256'])
  part=subprocess.check_output([S+'scontrol','show','partition','default_partition','-o'],text=True,timeout=20)
  assert 'MaxTime=UNLIMITED' in part,'Review changed partition time limit before deployment'
  def event(kind,**value):
   record=dict(kind=kind,utc=w.now(),**value)
   with (b/'deployment_ledger.jsonl').open('a') as f:f.write(json.dumps(record,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
  event('frozen_deployment',manifest_sha256=w.sha(b/'manifest.json'),deploy_sha256=w.sha(Path(__file__)),native_validation_sha256=w.sha(b/'validation.json'))
  w.require_hash(m['v1_manifest_path'],m['v1_manifest_sha256'])
  original_jobs=w.read(Path(m['original_manifest_path']).parent/'case_jobs.json')
  before={};selection=[]
  for cid,c in sorted(m['cases'].items()):
   cg,raw=fields(c['original_cg_job']);mip,mipraw=fields(original_jobs[cid]['mip']);v1,v1raw=fields(c['v1_gate_job']);assert v1['JobState']=='PENDING','V1 gate no longer pending; reconcile without cancelling it';before[cid]=dict(cg=cg,raw=raw,mip=mip,mip_raw=mipraw,v1_gate=v1,v1_raw=v1raw)
   if cg['JobState']!='PENDING':event('skip_existing_not_pending',case_id=cid,state=cg['JobState']);continue
   if cg.get('Reason')=='DependencyNeverSatisfied':raise ValueError('Late invalid dependency requires replacement DAG')
   # Validate graph-edge syntax now without introducing a job ID.
   gate.replacement_dependencies(cg['Dependency'],c['v1_gate_job'],'999999999')
   selection.append(cid)
  w.save(b/'before_dependencies.json',before)
  jobs=[];after={}
  for cid in selection:
   c=m['cases'][cid];cg,raw=fields(c['original_cg_job'])
   if cg['JobState']!='PENDING':event('skip_became_not_pending',case_id=cid,state=cg['JobState']);continue
   gate.replacement_dependencies(cg['Dependency'],c['v1_gate_job'],'999999999')
   args=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=2','--mem=64G','--time=1-00:30:00','--requeue','--kill-on-invalid-dep=yes','--dependency=afterany:'+c['original_graph_job'],'--job-name=drGraphGate_'+cid,'--output='+str(b/'logs/%x_%j.out'),'--error='+str(b/'logs/%x_%j.err'),str(b/'worker.sub'),str(b),cid]
   event('submit_intent',case_id=cid,argv=args)
   job=subprocess.check_output(args,text=True,timeout=45).strip().split(';')[0];assert job.isdigit()
   event('submit_recorded',case_id=cid,job_id=job)
   jobs.append(dict(case_id=cid,job_id=job,kind='graph_recovery_gate',original_graph_job=c['original_graph_job'],cg_job_id=c['original_cg_job'],dependency='afterany:'+c['original_graph_job'],argv=args));w.save(b/'jobs.json',jobs)
   current,raw=fields(c['original_cg_job'])
   if current['JobState']!='PENDING':
    event('skip_retarget_became_not_pending',case_id=cid,state=current['JobState'],gate_job_id=job);continue
   old=current['Dependency'];new=gate.replacement_dependencies(old,c['v1_gate_job'],job)
   event('retarget_intent',case_id=cid,cg_job_id=c['original_cg_job'],before_dependency=old,after_dependency=new)
   subprocess.run([S+'scontrol','update','JobId='+c['original_cg_job'],'Dependency='+new],check=True,capture_output=True,text=True,timeout=30)
   final,raw=fields(c['original_cg_job']);actual=final['Dependency']
   def strip(x):return x.replace('(unfulfilled)','').replace('(fulfilled)','')
   assert set(strip(actual).split(','))==set(new.split(','))
   event('retarget_recorded',case_id=cid,cg_job_id=c['original_cg_job'],before_dependency=old,after_dependency=actual)
   own,ownraw=fields(job)
   assert own['Partition']=='default_partition' and own['ExcNodeList']=='scaglione-compute-01' and own['NumCPUs']=='2' and own['MinMemoryNode']=='64G' and own['TimeLimit']=='1-00:30:00'
   assert c['original_graph_job'] in own['Dependency'] and own['Dependency'].startswith('afterany:')
   mip,mipraw=fields(original_jobs[cid]['mip'])
   assert mip['Dependency']==before[cid]['mip']['Dependency'],'Unexpected change to own-CG MIP dependency'
   after[cid]=dict(cg=final,cg_raw=raw,gate=own,gate_raw=ownraw,mip=mip,mip_raw=mipraw)
   # Cancel only the replaced pending V1 gate, after both CG and MIP edge verification.
   stale,staleraw=fields(c['v1_gate_job']);assert stale['JobState']=='PENDING','V1 gate became active; do not cancel automatically'
   event('cancel_v1_intent',case_id=cid,v1_gate_job=c['v1_gate_job'],v2_gate_job=job)
   subprocess.run([S+'scancel','--state=PENDING',c['v1_gate_job']],check=True,capture_output=True,text=True,timeout=30)
   stopped,stoppedraw=fields(c['v1_gate_job']);assert stopped['JobState']=='CANCELLED'
   event('cancel_v1_recorded',case_id=cid,v1_gate_job=c['v1_gate_job'],v2_gate_job=job,state=stopped['JobState'])
   after[cid]['v1_gate']=stopped;after[cid]['v1_raw']=stoppedraw
  w.save(b/'after_dependencies.json',after);w.save(b/'case_jobs.json',{j['case_id']:j['job_id'] for j in jobs})
  event('deployment_complete',gates=len(jobs),retargeted=len(after),before_cg_count=len(before),all_other_edges_preserved=True)
  w.save(b/'deployment_validation.json',dict(status='passed',manifest_sha256=w.sha(b/'manifest.json'),gate_count=len(jobs),retargeted_cg_count=len(after),all_other_dependency_edges_preserved=True,held_jobs_untouched=True,original_running_jobs_untouched=True,ledger_sha256=w.sha(b/'deployment_ledger.jsonl'),utc=w.now()))
  print(json.dumps(dict(gates=len(jobs),retargeted=len(after))))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
