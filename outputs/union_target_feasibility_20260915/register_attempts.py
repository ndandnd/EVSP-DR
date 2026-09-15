"""Merge only this experiment's exact restart paths under established registry lock."""
from pathlib import Path
import fcntl,hashlib,json
import common as w
from collect_status import collect,ROOT
CAMPAIGN='union_target_feasibility_20260915'
def register():
 status=collect();mh=status['manifest']['sha256'];records=[]
 for group in ['production','validation']:
  for case in status[group]:
   for a in case['attempts']:
    rp=a['result']['path'];assert Path(rp).is_relative_to(ROOT/'cases'/case['case_id']/'attempts')
    records.append({'job_id':case['job_id'],'case_id':case['case_id'],'campaign':CAMPAIGN,'campaign_case_id':CAMPAIGN+'/'+case['case_id'],'campaign_root':str(ROOT),'campaign_manifest_sha256':mh,'cohort':'validation_union_target_feasibility_20260915' if case['is_validation'] else 'default_union_target_feasibility_3600','is_validation':case['is_validation'],'solver_budget_s':case['solver_budget_s'],'result_path':rp,'attempt_tag':a['attempt_tag'],'restart_count':str(a['restart_count']),'requeue':True,'registered_utc':w.now(),'result_path_scope':'Exact per-restart target-feasibility result, possibly not yet published','optimization_kind':'constant_objective_target_feasibility','proof_scope':'validated finite-pool target-cap feasibility only; not fleet or charging optimization','target_cap':case['target_cap']})
 path=ROOT.parent/'mip_preemption_study_20260911/registry.json'
 with path.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);raw=path.read_bytes();data=json.loads(raw);known={(str(r['job_id']),r.get('result_path')):r for r in data['cases']};added=[]
  for row in records:
   key=(row['job_id'],row['result_path'])
   if key in known:
    for field in ['case_id','cohort','restart_count','solver_budget_s','campaign_manifest_sha256']:assert str(known[key].get(field))==str(row[field]),(key,field)
    continue
   data['cases'].append(row);known[key]=row;added.append(row)
  if added:w.save(path,data)
  receipt={'utc':w.now(),'registry_path':str(path),'before_sha256':hashlib.sha256(raw).hexdigest(),'after_sha256':w.sha(path),'manifest_sha256':mh,'helper_sha256':w.sha(__file__),'proposed_records':len(records),'added':len(added),'additions':added,'no_scheduler_changes':True,'cohorts':['default_union_target_feasibility_3600','validation_union_target_feasibility_20260915']}
  w.save(ROOT/'registry_receipts'/(receipt['utc'].replace(':','')+'.json'),receipt);w.save(ROOT/'latest_registry_merge.json',receipt)
 w.save(ROOT/'attempt_registry_additions.json',{'cases':records,'manifest_sha256':mh});print(json.dumps({k:v for k,v in receipt.items() if k!='additions'}));return receipt
if __name__=='__main__':register()
