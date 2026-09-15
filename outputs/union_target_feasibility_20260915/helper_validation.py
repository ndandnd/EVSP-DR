from pathlib import Path
import copy,json
import common as w
from collect_status import collect,proof,ROOT
m=w.read(ROOT/'manifest.json');v=collect();checks=[]
assert len(v['production'])==7 and len(v['validation'])==2
for group in ['production','validation']:
 for row in v[group]:
  assert row['startup']['full_size_license_probe_pass_marker']
  for a in row['attempts']:assert a['full_size_license_probe_pass_inferred_from_worker_entry'] and a['result']['path'].endswith('/'+a['attempt_tag']+'/result.json')
checks.append('7production_and2validation_separate_all_started_full_license_probe')
for row in v['validation']:
 assert row['completion_identity_verified']
 c=m['cases'][row['case_id']];result=w.read(row['published_result_path']);assert proof(result,c)==row['proof_classification']
 for key,value in [('proof_classification','fleet_optimal'),('target_cap',c['target_cap']+1),('source_result_sha256','0'*64)]:
  bad=copy.deepcopy(result);bad[key]=value
  try:proof(bad,c)
  except (AssertionError,ValueError):checks.append(row['case_id']+'_reject_'+key)
  else:raise AssertionError('did not reject '+key)
receipts=sorted((ROOT/'registry_receipts').glob('*.json'));first=w.read(receipts[0]);last=w.read(receipts[-1])
assert first['added']==9 and last['added']==0 and last['before_sha256']==last['after_sha256'];checks.append('registry9added_then0_idempotent')
report={'status':'passed','utc':w.now(),'manifest_sha256':w.sha(ROOT/'manifest.json'),'helper_sha256':{n:w.sha(ROOT/n) for n in ['collect_status.py','register_attempts.py','helper_validation.py']},'checks':checks,'production_job_ids':[r['job_id'] for r in v['production']],'validation_job_ids':[r['job_id'] for r in v['validation']],'no_job_mutations':True,'scope':'Live artifact collection and metadata merge; no new native solve and no production proof inferred from startup'}
w.save(ROOT/'helper_validation.json',report);w.save(ROOT/'collection_status.json',v);print(json.dumps(report))
