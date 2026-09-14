from pathlib import Path
import csv,importlib.util,json,subprocess
import worker as w
import seed_logic
B=Path(__file__).resolve().parent
C=B.parent/'cumulative_budget_20260913'
CODE=B.parent/'full_pool_recovery_20260912/code'
D=Path('/share/scaglione/nc437/evsp-dr/overnight_parallel_20260914')
def mod(p,n):
 s=importlib.util.spec_from_file_location(n,p);v=importlib.util.module_from_spec(s);s.loader.exec_module(v);return v

def main():
 assert not (B/'manifest.json').exists()
 cm=mod(C/'campaign.py','cm');old=mod(B.parent/'overnight_diagnostics_20260914/campaign.py','old')
 cm.code_check();cv=w.read(C/'manifest.json');rows=[json.loads(l) for l in (B/'design/source_inventory.jsonl').read_text().splitlines()]
 D.mkdir(parents=True,exist_ok=True);(D/'cases').mkdir(exist_ok=True)
 if not (B/'cases').exists():(B/'cases').symlink_to(D/'cases',target_is_directory=True)
 (B/'logs').mkdir(exist_ok=True);cases={};pairs=[]
 for r in rows:
  src=cv['cases'][r['id']];assert src['k'] in (8,10,15)
  w.require_hash(r['parent'],r['parent_sha']);w.require_hash(r['mip'],r['mip_sha']);w.require_hash(r['source'],r['source_sha'])
  original=w.read(r['parent']);parent=w.read(r['source']);mip=w.read(r['mip'])
  assert mip['source_result_sha256']==r['source_sha']
  assert {k for k in original.keys()|parent.keys() if original.get(k)!=parent.get(k)}<={'columns_journal','terminal_pool_snapshot'}
  assert original['final_lp']==parent['final_lp']
  parentinput=CODE/'data'/parent['csv'];w.require_hash(parentinput,parent['provenance']['instance_sha256'])
  def ids(p):
   with open(p) as f:return {x['Ordered_Trip_ID'] for x in csv.DictReader(f)}
  assert ids(parentinput)<ids(src['input_remote_path'])
  assert f'_k{src["k"]-1:02d}_' in parent['csv']
  arms=seed_logic.choose(parent,mip)
  audit=dict(pair_id=r['id'],parent_k=src['k']-1,target_k=src['k'],parent_status_path=r['parent'],parent_status_sha256=r['parent_sha'],
   mip_bound_parent_path=r['source'],mip_bound_parent_sha256=r['source_sha'],parent_mip_path=r['mip'],parent_mip_sha256=r['mip_sha'],
   parent_journal_path=parent['columns_journal'],parent_journal_sha256=mip['source_journal_sha256'],parent_journal_hash_scope='Existing native MIP execution binding; journal not consumed by subset construction.',
   selected_sequence_count=len(arms['integer']),parent_lp_support_count=len(parent['final_lp']['positive_routes']),parent_input_path=str(parentinput),parent_input_sha256=parent['provenance']['instance_sha256'],
   historical_costs=dict(parent_cg_wall_s=parent.get('wall_s'),parent_mip_runtime_s=mip.get('runtime_s'),parent_mip_gurobi_wall_s=mip.get('gurobi_optimize_wall_s'),
    ancestry_paths=src['ancestry_paths'][:-1],ancestry_sha256=src['ancestry_sha256'][:-1],cg_accounting=src['cg_accounting'][:-1],
    ancestry_cumulative_native_cg_wall_s=sum(x.get('native_wall_s',0) for x in []),source_cumulative_budget_case=src),
   count_scope='Equal unique parent trip-set sequences selected. Child replay accepted/added/replaced counts must be reported separately; no guarantee of equal added columns.')
  audit['historical_costs'].pop('ancestry_cumulative_native_cg_wall_s')
  pairs.append(audit)
  for arm,records in arms.items():
   cid=r['id']+'_'+arm;p=B/'seeds'/cid;p.mkdir(parents=True,exist_ok=True)
   journal=p/'sequences.jsonl';journal.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in records))
   seed=p/'seed.json';w.save(seed,seed_logic.status(parent,journal,arm,audit))
   c=old.common(cid,'cg',src['chain'],src['k'],'previous_k_'+arm,CODE,cm.COMMIT,CODE/'data',src['csv'],src['input_sha256'])
   args=cm.cg_args(src,'{out}',14400)+['--inherit-event-pool-from',str(seed),'--inherit-event-pool-workers','1','--inherit-max-columns','0','--inherit-time-limit-s','0']
   c.update(argv=args,solver_budget_s=14400,watchdog_s=16200,resources={'cpus':8,'mem':'96G','allocation_s':18000},
    cache_manifest_path=src['cache']+'.manifest.json',cache_manifest_sha256=w.sha(src['cache']+'.manifest.json'),pair_id=r['id'],seed=audit,seed_status_path=str(seed),seed_status_sha256=w.sha(seed),seed_journal_sha256=w.sha(journal),
    changed_factor='Previous-k sequence selection only; integer-selected versus count-matched highest positive LP weight',interpretation='Previous-instance seed-content comparison; historical fresh/full-pool endpoints are retrospective context only.')
   c['static_hashes'].update({str(p):w.sha(p) for p in [seed,journal,Path(r['parent']),Path(r['source']),Path(r['mip']),parentinput,CODE/'src/config.py']})
   cases[cid]=c
   child=old.mip_case(cid+'_mip',src['chain'],src['k'],CODE/'data',src['csv'],src['input_sha256'],'previous_k_'+arm,12600,10800)
   child.update(source_case=cid,source_cg_commit=cm.COMMIT,pair_id=r['id'],seed=audit);cases[child['id']]=child
 assert len(cases)==72
 manifest=dict(schema='evsp-previous-k-seed-content-v1',prepared_utc=w.now(),cases=cases,pairs=pairs,storage_root=str(D),
  baseline_physics=cv['settings']|{'inherited_columns':True,'initial_pool':'singletons plus replayed previous-k sequence subset'},
  execution_budgets={'cg_s':14400,'mip_s':12600,'stage1_s':10800},
  interpretation=['No current-k solution used as a seed.','Equal selected parent sequence counts; audit actual replay acceptance and child additions separately.',
   'Derived seed artifacts are not CG endpoints and carry no pricing certificate.','Parent CG and MIP cost is retained and must be included in end-to-end comparisons.',
   'Both pair arms use same input/cache/code/model and budget; historical full-pool/fresh results are context, not matched timing controls.'],
  resources={'independent_cg':36,'own_cg_dependent_mip':36,'partition':'default_partition','exclude':'scaglione-compute-01'},
  tooling_sha256={n:w.sha(B/n) for n in ['worker.py','worker.sub','seed_logic.py','prepare.py','submit.py']},policy_sha256=w.sha(B.parent/'SCAGLIONE_RESOURCE_POLICY.md'))
 for k in ['mip_s','stage1_s','inherited_columns']:manifest['baseline_physics'].pop(k,None)
 w.save(B/'manifest.json',manifest)
 for c in cases.values():
  if c['kind']=='cg':w.preflight(B,manifest,c)
  else:
   w.check_code(c['source_code'],c['execution_commit']);w.require_hash(c['input_path'],c['input_sha256'])
 w.save(B/'preflight.json',dict(status='passed',manifest_sha256=w.sha(B/'manifest.json'),cg_cases=36,mip_static_checks=36,dependent_source_checks='deferred until own CG completion'))
 print(json.dumps(dict(prepared=len(cases),manifest_sha256=w.sha(B/'manifest.json'))))
if __name__=='__main__':main()
