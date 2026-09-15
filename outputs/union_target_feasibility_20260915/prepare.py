from pathlib import Path
import copy,difflib
import common as w
from target_solver import model_digest,patched_source
ROOT=Path(__file__).resolve().parent
PRIOR=Path('/home/nc437/ladder-lite/compact_pool_union_20260915')
def main():
 old=w.read(PRIOR/'manifest.json');cases={}
 for cid in ['c1_k15_union','c1_k20_union','c2_k20_union','c2_k25_union','c3_k20_union','c4_k25_union','c5_k20_union','c5_k25_union']:
  orig=old['cases'][cid];built=w.read(PRIOR/'cases'/orig['source_case']/'completion.json');resultpath=PRIOR/'cases'/cid/'mip_result.json';v=w.read(resultpath);pa=v['physical_pool_audit'];construction=w.read(built['construction_path'])
  c=copy.deepcopy(orig);c.update(id=cid.replace('_union','_target'),source_result=built['result_path'],source_result_sha256=built['result_sha256'],source_journal=built['journal_path'],source_journal_sha256=built['journal_sha256'],native_pool_columns=pa['base_pool_column_count'],native_pool_ordered_sha256=pa['base_pool_ordered_sha256'],source_union_pool_set=construction['union_pool_set'],source_production_result=str(resultpath),source_production_result_sha256=w.sha(resultpath),source_production_fleet=v['buses'],source_production_bound=v['fleet_bound'],target_cap=orig['target_k'],is_validation=False,solver_budget_s=3600,watchdog_s=6900,resources={'cpus':8,'mem':'24G','allocation_s':7200},model_sha256=model_digest())
  c['static_hashes'][str(resultpath)]=w.sha(resultpath);c['static_hashes'][built['construction_path']]=built['construction_sha256']
  for key in ['stage1_s','treatment','source_case','kind']:c.pop(key,None)
  if cid!='c1_k20_union':cases[c['id']]=c
  if cid=='c1_k15_union':
   f=copy.deepcopy(c);f.update(id='validation_feasible',target_cap=v['mip_start']['validated_bus_count'],is_validation=True,solver_budget_s=60,watchdog_s=1700,resources={'cpus':8,'mem':'24G','allocation_s':1800},validation_expected='target_feasible_in_validated_finite_pool',fixture_rationale='Cap equals independently witnessed native greedy partition fleet in this exact production pool; no supplied start.');cases[f['id']]=f
  if cid=='c1_k20_union':
   assert v['fleet_proven'] and v['buses']==21 and v['fleet_bound']>20
   f=copy.deepcopy(c);f.update(id='validation_infeasible',target_cap=1,is_validation=True,solver_budget_s=60,watchdog_s=1700,resources={'cpus':8,'mem':'24G','allocation_s':1800},validation_expected='target_infeasible_in_validated_finite_pool',fixture_rationale='Cap 1 is below independently proved finite-pool minimum 21; deliberately easy solver classification fixture, not a repeat production target-20 solve.');cases[f['id']]=f
 for c in cases.values():
  for p,h in {**c['static_hashes'],c['source_result']:c['source_result_sha256'],c['source_journal']:c['source_journal_sha256'],c['input_path']:c['input_sha256']}.items():w.require_hash(p,h)
 native=Path(next(iter(cases.values()))['source_code'])/'src/run_exact_pool_mip.py';text=native.read_text();(ROOT/'native_adapter.diff').write_text(''.join(difflib.unified_diff(text.splitlines(True),patched_source(text).splitlines(True),fromfile='native871d057',tofile='target_adapter')))
 m={'schema':'union-target-feasibility-manifest-v1','created_utc':w.now(),'source_manifest_sha256':w.sha(PRIOR/'manifest.json'),'source_commit':'871d057e1067411f09581e37d78f7c1ca43f68bb','production_authorized':False,'no_augmentation':True,'initialization':'unchanged native greedy policy','resources_policy':'default_partition; exclude scaglione-compute-01; all seven independent production cases eligible; requeue preserves attempts, tree restarts','tooling_sha256':{n:w.sha(ROOT/n) for n in ['common.py','worker.py','worker.sub','target_solver.py']},'cases':cases}
 if (ROOT/'manifest.json').exists():raise ValueError('refuse manifest overwrite')
 w.save(ROOT/'manifest.json',m)
if __name__=='__main__':main()
