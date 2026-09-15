from pathlib import Path
import json,hashlib,fcntl,os,datetime
b=Path('/home/nc437/ladder-lite/chain_extension_31_32_20260915');r=b.parent/'mip_preemption_study_20260911/registry.json';proposed=json.loads((b/'mip_registry_additions.json').read_text())
with r.with_suffix('.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);raw=r.read_bytes();data=json.loads(raw);known={(str(x['job_id']),x.get('result_path')):x for x in data['cases']};added=[]
 for x in proposed:
  assert Path(x['result_path']).is_relative_to(b) and x['cohort']=='default_chain_extension_3600'
  key=(str(x['job_id']),x['result_path'])
  if key in known:
   assert all(known[key][n]==x[n] for n in ['case_id','cohort','solver_budget_s']);continue
  data['cases'].append(x);known[key]=x;added.append(x['job_id'])
 if added:
  tmp=r.with_name(r.name+'.extension3132.'+str(os.getpid()))
  with tmp.open('w') as out:json.dump(data,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
  tmp.replace(r)
 receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_sha256=hashlib.sha256(raw).hexdigest(),after_sha256=hashlib.sha256(r.read_bytes()).hexdigest(),added_job_ids=added,registered_cases=len(data['cases']),no_scheduler_mutation=True)
 (b/'mip_registry_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
