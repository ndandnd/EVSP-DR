"""Produce private fixture/production MIP registry additions; never merge them."""
from pathlib import Path
import json
import common as w
B=Path(__file__).resolve().parent
def export():
 records=[];cohorts=[]
 for root,validation in [(B/'native_fixture',True),(B,False)]:
  if not (root/'jobs.json').exists():continue
  m=w.read(root/'manifest.json');mh=w.sha(root/'manifest.json');cohort=('validation_compact_pool_union_20260915' if validation else 'default_compact_pool_union_20260915');cohorts.append(cohort)
  for j in w.read(root/'jobs.json'):
   c=m['cases'][j['case_id']]
   if c['kind']!='mip':continue
   attempts=sorted((root/'cases'/j['case_id']/'attempts').glob(j['job_id']+'_r*'))
   if not attempts:attempts=[root/'cases'/j['case_id']/'attempts'/(j['job_id']+'_r0')]
   for attempt in attempts:
    token=attempt.name;state_path=attempt/'state.json';state=w.read(state_path) if state_path.exists() else {};result=state.get('result_path',str(attempt/'result.json'))
    records.append({'job_id':j['job_id'],'case_id':j['case_id'],'campaign_case_id':'compact_pool_union_20260915/'+('native_fixture/' if validation else '')+j['case_id'],'campaign':'compact_pool_union_20260915','campaign_root':str(root),'campaign_manifest_sha256':mh,'cohort':cohort,'is_validation':validation,'solver_budget_s':c['solver_budget_s'],'stage1_time_limit_s':c['stage1_s'],'result_path':result,'attempt_tag':token,'restart_count':token.rsplit('_r',1)[1],'requeue':not validation,'registered_utc':w.now(),'registration_scope':'Private proposed additions only; root merges with established shared-registry lock','result_path_scope':'Attempt-owned published path' if state.get('result_path') else 'Stable unique attempt output path before publication','source_case':c['source_case'],'treatment':c['treatment']})
 v={'schema':'evsp-private-mip-registry-additions-v1','campaign':'compact_pool_union_20260915','generated_utc':w.now(),'shared_registry_modified':False,'cohorts':cohorts,'cases':records};w.save(B/'mip_registry_additions.json',v);print(json.dumps({'records':len(records),'validation':sum(r['is_validation'] for r in records),'production':sum(not r['is_validation'] for r in records),'shared_registry_modified':False}))
if __name__=='__main__':export()
