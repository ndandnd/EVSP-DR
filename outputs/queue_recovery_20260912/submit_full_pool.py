from pathlib import Path
import json,subprocess,datetime,hashlib
B=Path('/home/nc437/ladder-lite/full_pool_recovery_20260912');S='/usr/local/slurm/slurm-25.05.5/bin/';v=json.loads((B/'manifest.json').read_text());assert not (B/'jobs.json').exists();jobs=[];mapping={}
assert (B/'remote_tests_passed.json').exists()
def submit(mode,cid,parent=None):
 c=v['cases'][cid];wall='08:45:00' if mode=='cg' and c['cg_seconds']==28800 else '04:45:00' if mode=='cg' else '02:00:00'
 a=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem='+('96G' if mode=='cg' else '24G'),'--time='+wall,'--no-requeue','--kill-on-invalid-dep=yes','--job-name=drF_'+cid+('_CG' if mode=='cg' else '_M'),'--output='+str(B/'logs/%x_%j.out'),'--error='+str(B/'logs/%x_%j.err')]
 if parent:a+=['--dependency=afterok:'+parent]
 a += [str(B/'worker.sub'),mode,cid]
 rr=subprocess.run(a,capture_output=True,text=True,check=True);j=rr.stdout.strip().split(';')[0];assert j.isdigit()
 item={'job_id':j,'mode':mode,'case_id':cid,'dependency':parent,'argv':a,'submitted_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};jobs.append(item);(B/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n')
 st=subprocess.check_output([S+'scontrol','show','job',j,'-o'],text=True);assert 'ExcNodeList=scaglione-compute-01' in st and 'Partition=default_partition' in st;item['scontrol']=st;(B/'jobs.json').write_text(json.dumps(jobs,indent=2)+'\n');return j
# Submit the six independent roots first; only later sizes wait for previous CG.
prev={}
for chain,ids in v['warm_chains'].items():prev[chain]=submit('cg',ids[0]);mapping[ids[0]]={'cg':prev[chain]}
for chain,ids in v['warm_chains'].items():
 for pos,cid in enumerate(ids):
  if pos:prev[chain]=submit('cg',cid,prev[chain]);mapping[cid]={'cg':prev[chain]}
  mapping[cid]['mip']=submit('mip',cid,prev[chain])
(B/'case_jobs.json').write_text(json.dumps(mapping,indent=2)+'\n')
p=Path('/home/nc437/ladder-lite/mip_preemption_study_20260911/registry.json');reg=json.loads(p.read_text())
for r in jobs:
 if r['mode']=='mip':reg['cases'].append({'job_id':r['job_id'],'case_id':r['case_id'],'cohort':'default_full_pool_indexed_3600','solver_budget_s':3600,'result_path':str(B/'cases'/r['case_id']/'mip_result.json'),'attempt_tag':'full_pool_indexed_a01','requeue':False})
t=p.with_suffix('.tmp');t.write_text(json.dumps(reg,indent=2)+'\n');t.replace(p)
print(json.dumps({'cg_jobs':sum(x['mode']=='cg' for x in jobs),'mip_jobs':sum(x['mode']=='mip' for x in jobs),'independent_roots':[mapping[ids[0]]['cg'] for ids in v['warm_chains'].values()],'case_jobs':mapping}))
