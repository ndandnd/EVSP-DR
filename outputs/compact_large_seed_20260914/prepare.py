"""Frozen k19/k24 parent cores and 512-incidence fillers, a0e0 native caches."""
from pathlib import Path
import copy,csv,json
import worker as w
import seed_logic
B=Path(__file__).resolve().parent
OLD=B.parent/'compact_seed_support_20260914';EXT=B.parent/'chain_extension_20260913';CODE=EXT/'code'
D=Path('/share/scaglione/nc437/evsp-dr/compact_large_seed_20260914')
def table(p):
 with open(p) as f:return {r['Ordered_Trip_ID']:r for r in csv.DictReader(f)}
def main():
 assert not (B/'manifest.json').exists()
 old=w.read(OLD/'manifest.json');ext=w.read(EXT/'manifest.json');w.check_code(CODE,ext['execution_commit']);cases={};pairs=[]
 D.mkdir(parents=True,exist_ok=True);(D/'cases').mkdir(exist_ok=True);(B/'cases').symlink_to(D/'cases') if not (B/'cases').exists() else None;(B/'logs').mkdir(exist_ok=True)
 for chain in range(1,7):
  for k in [20,25]:
   pid=f'c{chain}_k{k:02d}';cid_old=f'w{chain}_k{k:02d}';prev=f'w{chain}_k{k-1:02d}';c0=ext['cases'][cid_old];pc=ext['cases'][prev];pr=EXT/'cases'/prev
   status=pr/'cg.json';parent=w.read(status);mip_path=pr/'mip_result.json';mip=w.read(mip_path);journal=Path(parent['columns_journal']);prov=w.read(pr/'cg_provenance.json');mp=w.read(pr/'mip_provenance.json')
   canonical_mip=mip;canonical_mip_sha=w.sha(mip_path);mip_path=Path(mp['result_path']);mip=w.read(mip_path);assert mip==canonical_mip,'Canonical MIP differs semantically from immutable published attempt'
   w.require_hash(status,prov['result_sha256']);w.require_hash(journal,prov['journal_sha256']);w.require_hash(mip_path,mp['result_sha256'])
   assert mip['source_result_sha256']==prov['result_sha256'] and mip['source_journal_sha256']==prov['journal_sha256'];assert mip['physical_replay_validated'] is True
   assert mip['physical_pool_audit']['rejected_columns']==0 and mip['physical_pool_audit']['deterministically_repaired']==0
   assert parent['provenance']['git_commit']==ext['execution_commit'] and parent['provenance']['instance_sha256']==pc['input_sha256'] and parent['csv']==pc['csv']
   parent_input=CODE/'data'/pc['csv'];child_input=CODE/'data'/c0['csv'];w.require_hash(parent_input,pc['input_sha256']);w.require_hash(child_input,c0['input_sha256']);assert c0['previous_input_sha256']==pc['input_sha256']
   ptab=table(parent_input);ctab=table(child_input);assert ptab.keys()<ctab.keys()
   for key,r in ptab.items():assert {n:v for n,v in r.items() if n!='count_trip_id'}=={n:v for n,v in ctab[key].items() if n!='count_trip_id'}
   cache_meta=w.read(c0['cache']+'.manifest.json');cache_done=w.read(EXT/'cases'/cid_old/'cache_result.json');w.require_hash(c0['cache']+'.manifest.json',cache_done['cache_manifest_sha256']);assert cache_meta['identity']['git_commit']==ext['execution_commit'] and cache_meta['identity']['instance_sha256']==c0['input_sha256'] and cache_meta['pickle_sha256']==cache_done['cache_sha256']
   with open(journal) as f:arms,selection=seed_logic.choose(parent,mip,(json.loads(l) for l in f if l.strip()))
   history=[]
   for pk in range(16,k):
    hp=EXT/'cases'/f'w{chain}_k{pk:02d}'/'cg.json';hv=w.read(hp);history.append(dict(path=str(hp.resolve()),sha256=w.sha(hp),native_wall_s=hv.get('wall_s'),execution_commit=(hv.get('provenance') or {}).get('git_commit')))
   a=dict(pair_id=pid,parent_k=k-1,target_k=k,parent_status_path=str(status.resolve()),parent_status_sha256=prov['result_sha256'],parent_mip_path=str(mip_path.resolve()),parent_mip_sha256=mp['result_sha256'],canonical_parent_mip_sha256=canonical_mip_sha,canonical_parent_mip_semantic_equality=True,parent_journal_path=str(journal),parent_journal_sha256=prov['journal_sha256'],parent_input_path=str(parent_input),parent_input_sha256=pc['input_sha256'],selection=selection,previous_stable_trip_nesting=True,physical_trip_attributes_unchanged_except_count_trip_id=True,historical_costs=dict(parent_cg_wall_s=parent.get('wall_s'),parent_mip_runtime_s=mip.get('runtime_s'),parent_mip_gurobi_wall_s=mip.get('gurobi_optimize_wall_s'),extension_history=history,baseline_k15_parent=ext['initial_parents'][str(chain)],source_extension_manifest_sha256=w.sha(EXT/'manifest.json')),count_scope='Exact selected native incidences; measure replay accepted and child added/replaced separately')
   pairs.append(a)
   for arm,rs in arms.items():
    cid=pid+'_'+arm;p=B/'seeds'/cid;p.mkdir(parents=True);seed_j=p/'sequences.jsonl';seed_j.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs));audit=a|selection['arms'][arm];seed=p/'seed.json';w.save(seed,seed_logic.status(parent,seed_j,arm,audit))
    c=copy.deepcopy(old['cases']['c1_k08_'+arm]);c.update(id=cid,chain=chain,target_k=k,target_duties=k,treatment='previous_k_'+arm,source_code=str(CODE),execution_commit=ext['execution_commit'],data_dir=str(CODE/'data'),csv=c0['csv'],input_path=str(child_input),input_sha256=c0['input_sha256'],pair_id=pid,seed=audit,seed_status_path=str(seed),seed_status_sha256=w.sha(seed),seed_journal_sha256=w.sha(seed_j),cache_manifest_path=c0['cache']+'.manifest.json',cache_manifest_sha256=w.sha(c0['cache']+'.manifest.json'),cache_pickle_sha256=cache_meta['pickle_sha256'],target_graph_cost=dict(original_build_s=cache_meta.get('original_build_s'),preparation_publication=cache_done,scope='Common preexisting target graph; build cost recorded separately from parent CG/MIP and child native wall budget'),cache_hash_check_scope='Native require-mode consumer verifies full pickle hash before use; producer publication and manifest verified during prepare',interpretation='Within-pair initialization comparison on a0e0; upstream history common. Across-size small cohort used e091 and is not a single-code comparison.')
    args=c['argv'];args[1]=str(CODE/'src/exact_pricer_expanded.py')
    for flag,value in [('--csv',c0['csv']),('--event-network-cache',c0['cache']),('--inherit-event-pool-from',str(seed))]:args[args.index(flag)+1]=value
    c['static_hashes']={str(CODE/'data'/name):digest for name,digest in ext['data_sha256'].items()};c['static_hashes'].update({str(CODE/'src/config.py'):w.sha(CODE/'src/config.py'),str(seed):w.sha(seed),str(seed_j):w.sha(seed_j),str(parent_input):pc['input_sha256'],str(status.resolve()):prov['result_sha256'],str(mip_path.resolve()):mp['result_sha256']});cases[cid]=c
    child=copy.deepcopy(old['cases']['c1_k08_'+arm+'_mip']);child.update(id=cid+'_mip',chain=chain,target_k=k,target_duties=k,treatment='previous_k_'+arm,source_case=cid,source_cg_commit=ext['execution_commit'],pair_id=pid,seed=audit,data_dir=str(CODE/'data'),csv=c0['csv'],input_path=str(child_input),input_sha256=c0['input_sha256'])
    for flag in ['--data-dir','--reference-data-dir']:child['argv'][child['argv'].index(flag)+1]=str(CODE/'data')
    child['static_hashes']={str(CODE/'data'/name):digest for name,digest in ext['data_sha256'].items()};child['static_hashes'][str(Path(child['source_code'])/'src/config.py')]=w.sha(Path(child['source_code'])/'src/config.py');cases[child['id']]=child
 m=copy.deepcopy(old);m.update(schema='evsp-compact-large-seed-support-v1',prepared_utc=w.now(),cases=cases,pairs=pairs,storage_root=str(D),frozen_previous_campaign_manifest_sha256=w.sha(OLD/'manifest.json'),frozen_extension_manifest_sha256=w.sha(EXT/'manifest.json'),baseline_physics=ext['scientific_settings'],interpretation=['Same a0e0 source in both large arms and native a0e0 graph caches; no identity adaptation.','Within-pair initialization comparison; no single-code across-size claim.','Previous-k integer plus all positive LP support core, filled512 arm adds distinct native incidences.','Parent CG/MIP/history costs retained; source MIP historical budget differs from new MIP budget.','No current-k optimized/GIRO routes; no transferred pricing certificate.'],resources={'independent_cg':24,'own_cg_dependent_mip':24,'partition':'default_partition','exclude':'scaglione-compute-01'},tooling_sha256={n:w.sha(B/n) for n in ['worker.py','worker.sub','seed_logic.py','prepare.py','submit.py','native_smoke.py','test_design.py']},policy_sha256=w.sha(B.parent/'SCAGLIONE_RESOURCE_POLICY.md'))
 for key in ['mip_seconds','stage1_seconds']:m['baseline_physics'].pop(key,None)
 m['baseline_physics']['inherit_workers']=1
 m['baseline_physics']['initial_pool']='singletons plus native replayed previous-k compact sequences'
 assert len(cases)==48;w.save(B/'manifest.json',m)
 for c in cases.values():
  if c['kind']=='cg':w.preflight(B,m,c)
  else:w.check_code(c['source_code'],c['execution_commit']);w.require_hash(c['input_path'],c['input_sha256'])
 w.save(B/'preflight.json',dict(status='passed',manifest_sha256=w.sha(B/'manifest.json'),cg_cases=24,mip_static_checks=24));print(w.sha(B/'manifest.json'))
if __name__=='__main__':main()
