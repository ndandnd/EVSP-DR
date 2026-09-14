"""Prepare20 independent pool diagnostics. Does not submit jobs or run CG."""
from pathlib import Path
import copy,hashlib,importlib.util,json,math,shutil

B=Path('/home/nc437/ladder-lite/parallel_pool_followup_20260914')
D=Path('/share/scaglione/nc437/evsp-dr/parallel_pool_followup_20260914')
OLD=B.parent/'overnight_diagnostics_20260914'
EXT=B.parent/'chain_extension_20260913'
CUM=B.parent/'cumulative_budget_20260913'
REP=B.parent/'mip_repeatability_20260914'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

IDENTITY=('csv','trip_ids','g_kwh','charge_kw','min_soc_frac','soc_step','block_min','time_model','master_sense','prices_csv')
PROVENANCE=('instance_sha256','prices_sha256','reference_sha256','deadhead_sha256','git_commit')

def check_identity(values):
 first=values[0]
 for v in values:
  for key in IDENTITY:
   if key not in first or key not in v or v[key]!=first[key]:raise ValueError('Union identity mismatch: '+key)
  for key in PROVENANCE:
   if not first['provenance'].get(key) or v['provenance'].get(key)!=first['provenance'][key]:raise ValueError('Union provenance mismatch: '+key)
 if first['master_sense']!='cover' or first['time_model']!='event':raise ValueError('Only baseline event covering unions are supported')

def union_records(record_groups,trips):
 allowed=set(trips);pool={};input_count=0
 for records in record_groups:
  for r in records:
   input_count+=1
   t=r['trips'];cost=float(r['cost'])
   if not t or len(t)!=len(set(t)) or not set(t)<=allowed or not math.isfinite(cost):raise ValueError('Invalid union column')
   key=frozenset(t)
   if key not in pool or cost<float(pool[key]['cost'])-1e-9:pool[key]=r
 return list(pool.values()),input_count

def bind(status_path):
 status_path=Path(status_path).resolve();v=read(status_path);j=Path(v['columns_journal'])
 if not j.is_absolute():j=status_path.parent/j
 return dict(status_path=str(status_path),status_sha256=sha(status_path),journal_path=str(j),journal_sha256=sha(j),value=v)

def attach(c,source):
 c.pop('source_case',None)
 c.update(source_status=source['status_path'],source_status_sha256=source['status_sha256'],
          source_journal_sha256=source['journal_sha256'],source_cg_commit=source['value']['provenance']['git_commit'])

def long_template(old,name,treatment):
 c=copy.deepcopy(old);c.update(id=name,treatment=treatment,solver_budget_s=12600,stage1_budget_s=10800,
  watchdog_s=15300,resources=dict(cpus=8,mem='24G',allocation_s=16200))
 a=c['argv'];a[a.index('--timelimit')+1]='12600';a[a.index('--stage1-timelimit')+1]='10800'
 c['interpretation']='Identical baseline covering physics, default seed, greedy pool initializer and eight threads; three-hour fleet search plus30-minute charging allowance. New tree; proof scope finite pool. No timing-causality claim.'
 return c

def main():
 assert not (B/'manifest.json').exists(),'Do not overwrite prepared manifest'
 old=read(OLD/'manifest.json');ext=read(EXT/'manifest.json');rep=read(REP/'manifest.json');selection=read(B/'selection.json')
 D.mkdir(parents=True,exist_ok=True);(D/'cases').mkdir(exist_ok=True);(D/'pools').mkdir(exist_ok=True)
 for part in ['cases','pools']:
  if not (B/part).exists():(B/part).symlink_to(D/part,target_is_directory=True)
  assert (B/part).resolve()==(D/part).resolve()
 (B/'logs').mkdir(exist_ok=True)
 for n in ['worker.py','worker.sub']:
  assert sha(OLD/n)==old['tooling_sha256'][n];shutil.copy2(OLD/n,B/n)
 spec=importlib.util.spec_from_file_location('frozen_worker',B/'worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
 cases={};constructions={}
 for base in selection['union_inputs']:
  for arm in ['c200','complementary']:
   if base+'_'+arm in selection['proved_treatment_pools']:continue
   source=bind(OLD/'cases'/(base+'_'+arm)/'cg.json')
   name=base+'_'+arm+'_longmip'
   c=long_template(old['cases'][base+'_'+arm+'_mip'],name,'unchanged_treatment_pool_longer_search')
   c.update(original_case=base,comparator=str(OLD/'cases'/(base+'_'+arm+'_mip')/'mip_result.json'))
   attach(c,source);cases[name]=c
 for original in selection['new_extension_gaps']:
  e=ext['cases'][original];source=bind(EXT/'cases'/original/'cg.json');latest=read(EXT/'cases'/original/'mip_result.json')
  c=long_template(rep['cases']['w1_k19_newgap_longmip'],original+'_longmip','new_extension_gap_longer_search')
  c.update(chain=e['chain'],target_k=e['k'],target_duties=e['k'],csv=e['csv'],input_path=str(EXT/'code/data'/e['csv']),
    input_sha256=e['input_sha256'],original_case=original,comparator=str(EXT/'cases'/original/'mip_result.json'),
    latest_original_buses=latest['buses'],latest_original_target_matched=latest['buses']<=e['k'])
  for key in ['latest_original_result','latest_original_sha256','selection_snapshot']:c.pop(key,None)
  attach(c,source);cases[c['id']]=c
 c=long_template(old['cases']['w4_k19_resume8h_mip'],'w4_k19_resumed_pool_longmip','unchanged_continued_pool_longer_search')
 attach(c,bind(OLD/'cases/w4_k19_resume8h/cg.json'));cases[c['id']]=c
 for base in ([] if selection.get('existing_only') else selection['union_inputs']):
  parent_paths=[CUM/'cases'/base/'base/completion.json',OLD/'cases'/(base+'_c200')/'completion.json',OLD/'cases'/(base+'_complementary')/'completion.json']
  sources=[]
  for p in parent_paths:
   comp=read(p);s=bind(comp['result_path'])
   assert comp['result_sha256']==s['status_sha256'] and comp['journal_sha256']==s['journal_sha256']
   sources.append(s)
  check_identity([s['value'] for s in sources])
  def records(p):
   with Path(p).open() as f:
    for line in f:
     if line.strip():yield json.loads(line)
  routes,input_count=union_records([records(s['journal_path']) for s in sources],sources[0]['value']['trip_ids'])
  out=B/'pools'/base;out.mkdir(exist_ok=False);journal=out/'union.columns.jsonl'
  with journal.open('x') as f:
   for r in routes:f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
  construction=dict(schema='evsp-finite-pool-union-v1',kind='pool_construction',optimization_run=False,
   construction_code_sha256=sha(B/'prepare.py'),sources=[{k:v for k,v in s.items() if k!='value'} for s in sources],
   source_order=['original','c200','complementary'],input_records=input_count,union_columns=len(routes),
   journal_path=str(journal),journal_sha256=sha(journal),
   deduplication='Native baseline covering identity: frozenset(trips); retain lower recorded cost by more than1e-9; ties retain first source. Entire unchanged route record retained.',
   full_model_lp_certified=False,certificate_claim='No union CG or LP solve was performed; source certificates remain source-only.',
   physical_validation='No new routes or costs synthesized. Native MIP physical gate replays union before solving; selected routes replayed again. No shared-capacity or heterogeneous-power claim.')
  # Compatibility input for the existing MIP loader: final.iter is explicitly
  # inherited metadata, not a count of union CG iterations; no final LP retained.
  status=copy.deepcopy(sources[0]['value'])
  for key in ['final_lp','route_values','trip_duals','objective','weighted_lp_objective','pricing_certificate_scope','iterations','history','last_iteration']:
   status.pop(key,None)
  status.update(schema='evsp-finite-pool-union-mip-input-v1',artifact_kind='finite_pool_union',optimization_run=False,
   columns_journal=str(journal),certified_rc_optimal=False,stop_reason='constructed_pool_not_cg',
   final={'artificials':0,'iter':int(sources[0]['value']['final']['iter']),'pool_columns':len(routes)},
   iteration_semantics='Compatibility-only source CG iteration metadata; no union CG executed',
   pool_construction=construction,wall_s=0)
  status['final']['iteration_origin']='original_constituent_only'
  save(out/'union.json',status);save(out/'construction.json',construction)
  c=long_template(old['cases'][base+'_c200_mip'],base+'_union_longmip','three_way_existing_column_union')
  c.update(original_case=base,pool_construction_path=str(out/'construction.json'),pool_construction_sha256=sha(out/'construction.json'),source_artifact_kind='finite_pool_union')
  c['static_hashes'][str(out/'construction.json')]=sha(out/'construction.json')
  attach(c,bind(out/'union.json'));cases[c['id']]=c;constructions[base]=construction
 expected=14 if selection.get('existing_only') else 20
 assert len(cases)==expected and all(not c.get('source_case') for c in cases.values())
 manifest=dict(schema='evsp-parallel-pool-followup-v1',cases=cases,selection=selection,selection_sha256=sha(B/'selection.json'),
  tooling_sha256={n:sha(B/n) for n in ['worker.py','worker.sub']},physics=old['physics'],pool_constructions=constructions,
  policy_sha256=sha(B.parent/'SCAGLIONE_RESOURCE_POLICY.md'),storage_root=str(D),
  interpretation=('14 independent existing-pool MIP allocations;6union preparations deferred to separate compute allocations.' if selection.get('existing_only') else '20 independent MIP allocations:14 existing-pool searches and6union searches; no newCG endpoint. Union construction must be collected separately from CG certificates.'),
  source_worker_note='Exact frozen worker reused; existing diagnostics cohort in registry, unique output paths identify followup campaign.')
 save(B/'manifest.json',manifest)
 checks=[]
 for cid,c in cases.items():
  source=w.preflight(B,manifest,c);a=w.expand_argv(c['argv'],'/validation/result.json','/validation',source['path'])
  assert '--cover' in a and '--two-stage' in a and a[a.index('--threads')+1]=='8'
  assert a[a.index('--timelimit')+1]=='12600' and a[a.index('--stage1-timelimit')+1]=='10800'
  checks.append(dict(case_id=cid,status='passed',source_status_sha256=source['status_sha256'],source_journal_sha256=source['journal_sha256']))
 save(B/'validation.json',dict(status='passed',manifest_sha256=sha(B/'manifest.json'),checks=checks,
  prepared_only=True,submitted_jobs=0,physical_gate='Existing tested native MIP runner performs full pool replay before optimization; no new pool route generation or cost reconstruction in preparer.'))
 print(json.dumps({'prepared':len(cases),'existing_pool':14,'union':len(constructions),'submitted':0}))

if __name__=='__main__':main()
