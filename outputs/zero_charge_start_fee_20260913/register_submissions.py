#!/usr/bin/env python3
"""Append fee jobs to the allocation/preemption study without changing jobs."""
from pathlib import Path
import datetime as dt,hashlib,json,os
base=Path('/home/nc437/ladder-lite');root=base/'zero_charge_start_fee_20260913'
p=base/'mip_preemption_study_20260911/registry.json';raw=p.read_bytes();registry=json.loads(raw)
known={str(r['job_id']) for r in registry['cases']};added=[]
m=json.loads((root/'manifest.json').read_text())
for job in json.loads((root/'jobs.json').read_text())['jobs']:
 jid=job['job_id'];case=job['case_id'];arm=job['order'][0]
 added.append(dict(job_id=jid,case_id=job['pair_id'],cohort='default_fee_CG7200_MIP3600_allocation14400',solver_budget_s=10800,mip_solver_budget_s=3600,cg_solver_budget_s=7200,allocation_budget_s=14400,input_sha256=m['inputs'][case]['input_sha256'],charge_start_cost=m['arms'][arm]['charge_start_cost'],result_path=str(root/'cases'/job['pair_id']/(jid+'_r0')/arm/'mip/result.json'),attempt_tag='fee_sensitivity_first_attempt',interpretation='Combined CG and MIP allocation. Do not pool its preemption rate with standalone one-hour MIPs.'))
groot=base/'giro_zero_start_fee_20260913'
plan=json.loads((groot/'plan.json').read_text());cells={c['id']:c for c in plan['cells']}
for job in json.loads((groot/'jobs.json').read_text())['jobs']:
 if job['stage']!='mip':continue
 cell=cells[job['pair_id']]
 added.append(dict(job_id=job['job_id'],case_id=job['pair_id'],cohort='default_GIRO_fee_MIP3600_allocation7200',solver_budget_s=3600,allocation_budget_s=7200,input_sha256=cell['instance_sha256'],charge_start_cost=cell['fee'],result_path=str(Path(cell['frontier_dir'])/'joint/comparison.json'),attempt_tag='fee_terminal_fair_first_attempt',dependency=job['dependency'],interpretation='One-hour MIP optimization, two-hour allocation includes pool replay and physical validation.'))
new=[x for x in added if x['job_id'] not in known]
if not new:print('No new attempts');raise SystemExit
before=root/'preemption_registry_before.json'
if not before.exists():before.write_bytes(raw)
registry['cases'].extend(new);registry['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
assert p.read_bytes()==raw,'Registry changed; merge against the newer version before retrying'
tmp=p.with_name('registry.fee.tmp')
with tmp.open('x') as stream:json.dump(registry,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
os.replace(tmp,p)
receipt={'added':new,'before_sha256':hashlib.sha256(raw).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'updated_utc':registry['updated_utc']}
(root/'preemption_registration.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Registered attempts',len(new))
