"""Read-only compact stage records and provenance for review monitoring."""
import json,hashlib,time,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.load(open(root/'manifest.json'));rows=[]
for case in manifest['cases']:
 base=root/'cases'/case['id'];row={'case':case['id'],'global_k':case['global_target_k'],'group':case['group'],'group_reference_duties':case['group_reference_duties'],'input_sha256':case['input_sha256']}
 for mode in ['cg','mip']:
  p=base/f'{mode}.json'
  if not p.exists():row[mode]={'status':'no_canonical_result'};continue
  d=json.load(open(p));entry={'path':str(p),'sha256':sha(p),'provenance':d.get('provenance')}
  if mode=='cg':
   entry.update({k:d.get(k) for k in ['status','stop_reason','certified_rc_optimal','terminal_exact_min_reduced_cost','final','runtime_s','network_build_s','physics','inheritance','pool_sha256']})
   entry['iterations']=len(d['iterations']);entry['total_pricing_s']=sum(x.get('pricing_s',0) for x in d['iterations']);entry['total_iteration_lp_s']=sum(x.get('lp_solve_s',0) for x in d['iterations'])
  else:
   result=d['result'];entry['result']={k:v for k,v in result.items() if k not in ['selected_indices']};entry['duplicate_service_audit']=d['duplicate_service_audit'];entry['capacity_enforced_in_mip']=d['capacity_enforced_in_mip']
  row[mode]=entry
 rows.append(row)
print(json.dumps({'schema':'review-strict-c5-collection-v1','collected_unix':time.time(),'manifest_sha256':sha(root/'manifest.json'),'cases':rows,'jobs':json.load(open(root/'jobs.json')) if (root/'jobs.json').exists() else None,'scope':'group component statuses; union rows require both group components, see manifest.unions'}))
