"""Freeze review P1 experiments, no submissions or optimization."""
from pathlib import Path
import copy,json
import worker as w
B=Path(__file__).resolve().parent; H=B.parent
COMMIT='6830caa225856903d1157ef8587863c7ae21ad53'
CODE=H/'execution/6830caa2'

def comparison_record(rp):
 r=w.read(rp)
 assert r['physical_replay_validated']
 p=r['physical_pool_audit']
 assert p['rejected_columns']==p['deterministically_repaired']==p['added_giro_route_count']==0
 return dict(result_path=str(rp),result_sha256=w.sha(rp),buses=r['buses'],fleet_bound=r.get('fleet_bound'),fleet_proven=r.get('fleet_proven'),ordered_pool_sha256=p['mip_ordered_pool_sha256'],pool_columns=r['pool_columns'],mip_start={k:v for k,v in r['mip_start'].items() if k!='solver_acceptance'},mip_provenance=r['mip_provenance'],two_stage=r['two_stage'],physics=r['physics'])

def main():
 assert not (B/'manifest.json').exists()
 template=next(iter(w.read(H/'continuation_gaps15_20260916/manifest.json')['cases'].values()))
 cases={};reuse=[]
 def add(cid,chain,k,item,findings,sp,ip,data,csv,rp,seed,fleet=10800):
  s=w.read(sp);jp=w.journal_path(sp,s);original=comparison_record(rp)
  assert s['provenance']['instance_sha256']==w.sha(ip)
  r=w.read(rp);assert r['source_journal_sha256']==w.sha(jp)
  # Canonical status JSONs can differ only by path; source journal and input are authoritative.
  assert r['source_result_sha256']==w.sha(sp) or w.read(r['source_result'])==s
  c=copy.deepcopy(template)
  for field in ['latest_original_result','latest_original_sha256','latest_original_buses','latest_original_target_matched','matched_original_audit','selection_snapshot','selected_as_gap_at_snapshot']:c.pop(field,None)
  total=fleet+1800
  c.update(id=cid,chain=chain,target_k=k,target_duties=k,review_item=item,findings=findings,seed=seed,treatment='review_saved_pool_seed_time_control',source_code=str(CODE),execution_commit=COMMIT,data_dir=str(data),csv=csv,input_path=str(ip),input_sha256=w.sha(ip),source_status=str(sp),source_status_sha256=w.sha(sp),source_journal_sha256=w.sha(jp),source_cg_commit=s['provenance']['git_commit'],original_case=cid.split('_seed')[0],solver_budget_s=total,stage1_budget_s=fleet,watchdog_s=total+2700,resources=dict(cpus=8,mem='24G',allocation_s=total+3600),changed_factor='Fleet search allowance and explicit Gurobi Seed. Unchanged ordered column pool and greedy start policy.',comparator=original,interpretation='Finite-pool MIP repeat; no new CG. Seed0 default and explicitly set 0 are numerically the same solver parameter. Node and timing can differ.',static_hashes={str(data/n):w.sha(data/n) for n in ['Ref_dict.csv','par_ref_dhd.csv','hourly_prices_flat.csv']})
  c['static_hashes'][str(rp)]=w.sha(rp)
  c['argv']=['/home/nc437/evsp_env/bin/python',str(CODE/'src/run_exact_pool_mip.py'),'--result','{source_status}','--data-dir',str(data),'--reference-data-dir',str(data),'--cover','--two-stage','--timelimit',str(total),'--stage1-timelimit',str(fleet),'--threads','8','--mipgap','0.0001','--seed',str(seed),'--gurobi-log','{attempt_dir}/gurobi.log','--out','{out}']
  cases[cid]=c
 C=H/'cumulative_budget_20260913';cm=w.read(C/'manifest.json')
 for chain in range(1,7):
  key=f'c{chain}_k15';src=cm['cases'][key];cp=C/'cases'/key
  sp=Path(w.read(cp/'base/completion.json')['result_path']);rp=Path(w.read(cp/'mip_base/completion.json')['result_path']);ip=Path(src['input_remote_path']);data=H/'full_pool_recovery_20260912/code/data'
  for seed in range(3):add(f'fresh_c{chain}_k15_seed{seed}',chain,15,7,['F5'],sp,ip,data,src['csv'],rp,seed)
 E=H/'chain_extension_31_32_20260915';em=w.read(E/'manifest.json');data=E/'code/data'
 for chain,root in [(1,'continuation_gaps15_20260916'),(3,'continuation_gaps12_20260916'),(4,'continuation_gaps14_20260916'),(5,'continuation_gaps12_20260916')]:
  key=f'w{chain}_k32';e=em['cases'][key];sp=(E/'cases'/key/'cg.json').resolve();rp=(E/'cases'/key/'mip_result.json').resolve();ip=data/e['csv']
  old=w.read(H/root/'manifest.json')['cases'][key+'_longmip'];original=comparison_record(rp)
  assert old['input_sha256']==w.sha(ip) and old['source_journal_sha256']==w.sha(w.journal_path(sp,w.read(sp)))
  assert old['solver_budget_s']==12600 and old['stage1_budget_s']==10800 and old['execution_commit']=='871d057e1067411f09581e37d78f7c1ca43f68bb'
  entry=dict(case_id=f'warm_{key}_seed0',review_item=9,findings=['F2','F5'],chain=chain,k=32,seed=0,campaign=str(H/root),case=key+'_longmip',manifest_sha256=w.sha(H/root/'manifest.json'),source_journal_sha256=old['source_journal_sha256'],comparison=original)
  cp=H/root/'cases'/(key+'_longmip')/'completion.json'
  if cp.exists():
   comp=w.read(cp); rr=Path(comp['result_path']); result=comparison_record(rr)
   assert result['ordered_pool_sha256']==original['ordered_pool_sha256'] and result['mip_start']==original['mip_start']
   assert result['mip_provenance']['gurobi_parameters']['Seed']==0
   assert result['two_stage']['stage1_time_limit_s']==10800
   entry.update(state='completed_reused',result=result)
  else:entry.update(state='existing_job_in_progress',jobs=w.read(H/root/'jobs.json'))
  reuse.append(entry)
  for seed in [1,2]:add(f'warm_{key}_seed{seed}',chain,32,9,['F2','F5'],sp,ip,data,e['csv'],rp,seed)
 key='w5_k31';e=em['cases'][key]
 add('warm_w5_k31_seed0_12h',5,31,8,['F2','F4'],(E/'cases'/key/'cg.json').resolve(),data/e['csv'],data,e['csv'],(E/'cases'/key/'mip_result.json').resolve(),0,43200)
 D=Path('/share/scaglione/nc437/evsp-dr')/B.name;D.mkdir(exist_ok=False);(D/'cases').mkdir();(B/'cases').symlink_to(D/'cases');(B/'logs').mkdir()
 manifest=dict(schema='evsp-independent-review-p1-v1',prepared_utc=w.now(),review_sha256=w.sha(B/'REVIEW.md'),execution_commit=COMMIT,base_commit='871d057e1067411f09581e37d78f7c1ca43f68bb',cases=cases,reused_cells=reuse,policy_sha256=w.sha(H/'SCAGLIONE_RESOURCE_POLICY.md'),storage_root=str(D),tooling_sha256={n:w.sha(B/n) for n in ['worker.py','worker.sub','prepare.py','submit.py']},scientific_settings=dict(master_sense='cover',fleet_stage='minimize sum x',cost_stage='sum x <= fleet incumbent; minimize original full objective',electricity_tariff='hourly_prices_flat.csv',battery_kwh=240,charge_kw=240,reserve_kwh=0,charge_start_cost=5,capacity='unconstrained',terminal_soc='unconstrained',mip_start='same deterministic greedy pool partition; no incumbent import',column_pool='exact original ordered pool; physical replay gate unchanged'),status='prepared_not_submitted')
 w.save(B/'manifest.json',manifest)
 checks=[]
 for cid,c in cases.items():
  w.preflight(B,manifest,c);checks.append(dict(case=cid,status='passed',source_journal_sha256=c['source_journal_sha256']))
 w.save(B/'validation.json',dict(status='passed',manifest_sha256=w.sha(B/'manifest.json'),checks=checks,reused_cells=len(reuse),new_cells=len(cases)))
 print(json.dumps(dict(cases=len(cases),reused_cells=len(reuse),manifest_sha256=w.sha(B/'manifest.json'))))
if __name__=='__main__':main()
