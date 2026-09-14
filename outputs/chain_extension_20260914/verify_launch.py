"""Read-only post-submission checks and local cohort export; no scheduler mutation."""
import json,subprocess,datetime
from pathlib import Path
import campaign as c
v=c.read(c.B/'manifest.json'); jobs=c.read(c.B/'jobs.json'); mapping=c.read(c.B/'case_jobs.json')
assert len(jobs)==37 and len(mapping)==18
ids=sorted(v['cases']);graph=jobs[0]['job_id'];cohort=[];states={}
for row in jobs:
 text=subprocess.check_output([c.SLURM+'scontrol','show','job',row['job_id'],'-o'],text=True)
 assert 'ExcNodeList=scaglione-compute-01' in text and 'Partition=default_partition' in text
 assert 'MinMemoryNode='+v['resources'][row['mode']]['mem'] in text
 states[row['job_id']]=text
 if row['mode']=='cache':assert 'ArrayTaskThrottle=18' in text
byid={x['job_id']:x for x in jobs}
for chain,chain_ids in v['warm_chains'].items():
 previous=None
 for n,cid in enumerate(chain_ids):
  m=mapping[cid];cg=byid[m['cg']];mip=byid[m['mip']]
  assert m['cache']==graph+'_'+str(ids.index(cid))
  expected=[m['cache']]+([previous] if previous else [])
  if n==0:
   p=v['initial_parents'][chain]
   assert cg['dependencies'] in [expected, expected+[p['parent_job_id']]]
   if cg['dependencies']==expected:c.authenticate_parent(p['status'],p['csv_sha256'])
  else:assert cg['dependencies']==expected
  assert mip['dependencies']==[m['cg']]
  previous=m['cg']
  cohort.append({'job_id':m['mip'],'case_id':cid,'cohort':'default_chain_extension_3600','solver_budget_s':3600,'result_path':str(c.B/'cases'/cid/'mip_result.json'),'attempt_tag':'chain_extension_20260914','requeue':True})
squeue=subprocess.check_output([c.SLURM+'squeue','-r','-j',','.join(x['job_id'] for x in jobs),'-h','-o','%i|%T|%R|%m|%C|%P'],text=True)
sacct=subprocess.check_output([c.SLURM+'sacct','-j',','.join(x['job_id'] for x in jobs),'--format=JobID,State,ExitCode,Elapsed,ReqMem,AllocCPUS','-n','-P'],text=True)
c.save(c.B/'launch_verification.json',{'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':c.sha(c.B/'manifest.json'),'jobs_sha256':c.sha(c.B/'jobs.json'),'case_jobs_sha256':c.sha(c.B/'case_jobs.json'),'production_tasks':54,'independent_graphs':18,'cg_jobs':18,'mip_jobs':18,'all_resources_exclusions_dependencies_verified':True,'scontrol':states,'squeue':squeue,'sacct':sacct})
c.save(c.B/'mip_registry_additions.json',cohort)
print(json.dumps({'graph_array':graph,'jobs':len(jobs),'cases':len(mapping),'squeue':squeue}))
