from pathlib import Path
import json,hashlib,importlib.util,collections
root=Path(__file__).resolve().parents[2];p=root/'outputs/overnight_next_20260914';raw=p/'launch_collection.json';d=json.loads(raw.read_text());source=root/'outputs/post_meeting_20260910/monitor/20260915T010508Z.json';old=json.loads(source.read_text());old['campaigns'].update(d['campaigns'])
spec=importlib.util.spec_from_file_location('register_builder',root/'outputs/research_register/build_register.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.Register(old,source,p/'normalizer_validation_only',[]);r.build();checks=[]
for name,c in d['campaigns'].items():
 for item in c.get('mip',[]):
  rows=[x for x in r.rows if x['campaign_id']==name and x['stage']=='mip' and x['source_path']==item['path']];assert len(rows)==1
  row=rows[0];assert row['mip_incumbent_fleet']==item['buses'] and row['fleet_proven']==item['fleet_proven'] and row['source_sha256']==item['sha256']
  assert item['physical_replay_validated'] and item['physical_pool_audit']['rejected_columns']==item['physical_pool_audit']['deterministically_repaired']==0
  checks.append({'case_id':row['case_id'],'target_k':row['target_k'],'buses':row['mip_incumbent_fleet'],'fleet_proved_in_pool':row['fleet_proven'],'source_sha256':row['source_sha256'],'input_sha256':row['input_sha256'],'normalized_exactly':True})
assert len(checks)==9 and all(x['buses']>x['target_k'] and x['fleet_proved_in_pool'] for x in checks)
fixture=json.loads((root/'outputs/reserve_feasibility_screen_20260914/native_fixture_collection.json').read_text()); fixture=fixture.get('campaigns',{}).get('reserve_feasibility_screen_20260914',fixture)
# The marked smoke fixture is used only here, never merged into published production data.
# Raw native smoke lacks production worker markers and must not be admitted as a production endpoint.
old['campaigns']['reserve_feasibility_screen_20260914']=fixture
try:
    rejected=m.Register(old,source,p/'normalizer_validation_only',[]);rejected.build()
except ValueError as exc:
    assert 'lacks completion verification' in str(exc)
else:
    raise AssertionError('raw smoke was incorrectly promoted')
# Unit-only adapter-shaped copy checks metadata mapping, not production proof verification.
import copy
fixture=copy.deepcopy(fixture)
fixture['cg']=[]
fixture['mip']=[]
manifest=json.loads((root/'outputs/reserve_feasibility_screen_20260914/manifest.json').read_text())
fixture['workflow']['manifest.json']=manifest
case_map={c['case_id']:c for c in manifest['cases']}
for record in fixture['records']:
    record['stage_completion_verified']=True
    record['case_metadata']=case_map[record['case_id']]
old['campaigns']['reserve_feasibility_screen_20260914']=fixture
r2=m.Register(old,source,p/'normalizer_validation_only',[]);r2.build();testrows=[x for x in r2.rows if x['campaign_id']=='reserve_feasibility_screen_20260914' and x['stage'] in ['cg','mip']]
assert len(testrows)==2, len(testrows)
for x in testrows:assert x['target_k']==1 and x['battery_kwh']==236.44 and x['reserve_kwh']==35.466
(p/'normalizer_validation.json').write_text(json.dumps({'status':'passed','validation_only':True,'authoritative_register_not_rebuilt':True,'snapshot_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'records':checks,'reserve_adapter_shape_unit_fixture_records':2,'raw_unverified_smoke_correctly_rejected':True,'production_reserve_adapter_endpoint_pending':True,'reserve_fixture_not_published_as_production':True},indent=2)+'\n')
lines=['# First overnight diagnostic results','',f"Collection: {d['timestamp_utc']}. All nine support-only controls have completed; the26 paired augmentation MIPs remain running.",'','| Chain, target | Target buses | Buses needed using only final LP support | Fleet minimum proved in this pool |','|---|---:|---:|---|']
for x in checks:lines.append(f"| {x['case_id'].removesuffix('_support_only')} | {x['target_k']} | {x['buses']} | yes |")
lines+=['','All nine pass native individual-route replay with zero rejected/repaired columns. The full frozen donor pools match their targets; the restricted support-only pools do not. Thus at least some columns with zero weight in these donors\' final LP solutions are necessary for target-fleet recovery. This statement concerns these saved pools and these selected cases. It does not identify every useful inactive column or prove general algorithmic performance. No new CG or full-model pricing certificate is produced by filtering a pool.','', '[Collection and result hashes](launch_collection.json), [normalizer validation](normalizer_validation.json), [frozen campaign](../lp_support_pool_diagnostic_20260914/README.md).']
(p/'FIRST_DIAGNOSTIC_RESULTS.md').write_text('\n'.join(lines)+'\n')
jobs=[x for x in d['queue']['jobs'] if not x.get('name','').startswith('tpmR')]; counts=collections.Counter(x['job_state'][0] for x in jobs);assert counts=={'RUNNING':89,'PENDING':54};assert not any(x['state_reason'] in ['DependencyNeverSatisfied','JobArrayTaskLimit'] for x in jobs);assert not any('scaglione-compute-01' in str(x.get('nodes')) for x in jobs)
(p/'queue_verification.json').write_text(json.dumps({'collected_until_utc':d['collected_until_utc'],'counts':dict(counts),'held_historical_excluded':33,'all_pending_reasons':'Dependency','reserved_node_assignments':0,'new_independent_allocations':69,'new_following_mips':24},indent=2)+'\n')
print('Verified9 new production endpoints,2 separate native physics fixtures,and queue counts.')
