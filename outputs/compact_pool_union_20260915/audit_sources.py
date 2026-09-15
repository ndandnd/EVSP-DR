"""Read-only compact-pair source/duplicate audit; does not submit or build pools."""
from pathlib import Path
import datetime,hashlib,json,subprocess
B=Path('/home/nc437/ladder-lite');OUT=B/'compact_pool_union_20260915'
MIP='871d057e1067411f09581e37d78f7c1ca43f68bb'
CASES=['c1_k15','c1_k20','c2_k20','c2_k25','c3_k20','c4_k25','c5_k20','c5_k25']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(p,h):assert sha(p)==h,str(p)
def opt(a,k):return a[a.index(k)+1] if k in a else None
rows=[]
for base in CASES:
 campaign=B/('compact_seed_support_20260914' if base=='c1_k15' else 'compact_large_seed_20260914')
 manifest=read(campaign/'manifest.json');mh=sha(campaign/'manifest.json');sources=[];values=[]
 for arm in ['core','core512']:
  cid=base+'_'+arm;cg=read(campaign/'cases'/cid/'completion.json');mc=read(campaign/'cases'/(cid+'_mip')/'completion.json')
  assert cg['status']==mc['status']=='finished' and cg['usable'] and mc['usable']
  # Source manifest has a metadata-freeze history; bind actual execution marker and current manifest separately.
  req(cg['result_path'],cg['result_sha256']);req(mc['result_path'],mc['result_sha256'])
  req(campaign/'cases'/cid/'cg.json',cg['result_sha256']);req(campaign/'cases'/(cid+'_mip')/'mip_result.json',mc['result_sha256'])
  v=read(cg['result_path']);m=read(mc['result_path']);pa=m['physical_pool_audit'];a=mc['execution']['argv'];c=manifest['cases'][cid]
  assert m['source_result_sha256']==cg['result_sha256'] and m['source_journal_sha256']==cg['journal_sha256']
  assert m['physical_replay_validated'] is True and pa['rejected_columns']==pa['deterministically_repaired']==0
  assert pa['base_pool_column_count']==pa['post_augmentation_columns'] and pa['base_pool_ordered_sha256']==pa['augmented_pool_ordered_sha256']
  assert mc['execution_commit']==MIP and a[1]==str(B/'execution'/MIP/'src/run_exact_pool_mip.py')
  assert '--cover' in a and '--two-stage' in a and '--initial-partition-routes' not in a and '--extra-routes' not in a
  assert (opt(a,'--timelimit'),opt(a,'--stage1-timelimit'),opt(a,'--threads'))==('12600','10800','8')
  assert v['provenance']['instance_sha256']==c['input_sha256']==pa['input_hashes']['instance_sha256']
  assert v['provenance']['git_commit']==c['execution_commit']
  assert m['stage2_fleet_constraint']=='at_most'
  req(c['input_path'],c['input_sha256'])
  for p,h in c['static_hashes'].items():req(p,h)
  k=int(base.split('_k')[1]);bound=m['fleet_bound'];routes=m['selected_routes']
  assert len(routes)==m['buses'] and all(r['master_cost_semantics']=='expanded_grid_cost' and abs(r['cost']-r['expanded_grid_cost'])<1e-6 for r in routes)
  values.append(v)
  sources.append(dict(arm=arm,case_id=cid,campaign=str(campaign),current_manifest_sha256=mh,cg_execution_manifest_sha256=cg['manifest_sha256'],mip_execution_manifest_sha256=mc['manifest_sha256'],cg_completion_path=str(campaign/'cases'/cid/'completion.json'),cg_completion_sha256=sha(campaign/'cases'/cid/'completion.json'),status_path=cg['result_path'],status_sha256=cg['result_sha256'],journal_path=v['columns_journal'],journal_sha256=cg['journal_sha256'],journal_size_bytes=Path(v['columns_journal']).stat().st_size,journal_rehash_scope='Not reread during lightweight audit; marker and bound native MIP hashes agree; compute construction must rehash before and after',mip_completion_path=str(campaign/'cases'/(cid+'_mip')/'completion.json'),mip_completion_sha256=sha(campaign/'cases'/(cid+'_mip')/'completion.json'),mip_path=mc['result_path'],mip_sha256=mc['result_sha256'],mip_provenance=m['mip_provenance'],buses=m['buses'],fleet_bound=bound,fleet_proven=m['fleet_proven'],target_excluded_by_bound=bound is not None and bound>k+1e-6,expanded_grid_objective=m['mip_obj'],selected_route_set_sha256=m['selected_route_set_sha256'],cg_wall_s=v['wall_s'],cg_certificate=v['certified_rc_optimal'],cg_stop_reason=v['stop_reason'],cg_commit=c['execution_commit'],native_pool_columns=pa['base_pool_column_count'],native_pool_ordered_sha256=pa['base_pool_ordered_sha256'],physical_pool_audit=pa,source_cg_ancestry=c['seed'].get('historical_costs',c['seed'].get('parent',{})),source_manifest_seed=c['seed'],resources=c['resources'],data_dir=c['data_dir'],input_path=c['input_path'],input_sha256=c['input_sha256'],static_hashes=c['static_hashes'],mip_argv=a))
 for key in ['csv','trip_ids','g_kwh','charge_kw','min_soc_frac','soc_step','block_min','time_model','master_sense','prices_csv']:
  assert values[0][key]==values[1][key],key
 for key in ['git_commit','instance_sha256','prices_sha256','reference_sha256','deadhead_sha256']:
  assert values[0]['provenance'][key]==values[1]['provenance'][key],key
 assert all(s['buses']>k for s in sources)
 best=min(sources,key=lambda s:(s['buses'],s['target_excluded_by_bound'],s['expanded_grid_objective'],s['arm']))
 viable=[s for s in sources if not s['target_excluded_by_bound']]
 control=(min(viable,key=lambda s:(s['buses'],s['expanded_grid_objective'],s['arm']))['arm'] if viable else None)
 rows.append(dict(pair_id=base,target_k=k,sources=sources,chosen_witness_arm=best['arm'],control_arm=control,witness_tie_policy='Minimum validated fleet; prefer a target-viable source on fleet tie; then smaller expanded cost, then arm name',combined_child_cg_wall_s=sum(s['cg_wall_s'] for s in sources),same_computation_performance_claim=False,source_start_membership='Not parsed: native unconditional augmentation blocked design first'))
# Bounded inventory: immediate campaign manifests and JSON receipts/status for names containing union or compact.
existing=[]
for d in sorted(B.iterdir()):
 if not d.is_dir() or not any(t in d.name for t in ['union','compact']) or d==OUT:continue
 for name in ['manifest.json','jobs.json','case_jobs.json','selection.json']:
  p=d/name
  if not p.is_file():continue
  v=read(p);record=dict(path=str(p),sha256=sha(p))
  if name=='manifest.json':
   cases=v.get('cases',{});record['case_ids']=list(cases) if isinstance(cases,dict) else [c.get('id') for c in cases]
   record['compact_pair_references']=[b for b in CASES if b in p.read_text()]
  existing.append(record)
q=subprocess.check_output(['/usr/local/slurm/slurm-25.05.5/bin/squeue','-u','nc437','-h','-o','%i|%j|%T|%E'],text=True)
(B/'compact_pool_union_20260915').mkdir(exist_ok=True)
(OUT/'queue_at_audit.txt').write_text(q)
v=dict(audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='source_checks_passed_design_blocked_before_pool_construction',native_script_path=str(B/'execution'/MIP/'src/run_exact_pool_mip.py'),native_script_sha256=sha(B/'execution'/MIP/'src/run_exact_pool_mip.py'),policy_sha256=sha(B/'SCAGLIONE_RESOURCE_POLICY.md'),pairs=rows,existing_campaign_inventory=existing,duplicate_scope='Immediate union and compact campaign manifests/receipts plus complete current user queue; excludes unavailable historical scheduler jobs',queue_sha256=sha(OUT/'queue_at_audit.txt'),no_submissions=True)
(OUT/'source_audit.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps(dict(status=v['status'],pairs=[dict(pair_id=r['pair_id'],outcomes=[(s['arm'],s['buses'],s['fleet_bound'],s['fleet_proven']) for s in r['sources']],witness=r['chosen_witness_arm'],control=r['control_arm']) for r in rows],native_script_sha256=v['native_script_sha256'],inventory_count=len(existing)),indent=2))
