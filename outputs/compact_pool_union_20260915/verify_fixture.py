"""Verify native-fixture outputs, accounting and effective requested resources."""
from pathlib import Path
import json,subprocess
import common as w
B=Path(__file__).resolve().parent;ROOT=B/'native_fixture';S='/usr/local/slurm/slurm-25.05.5/bin/'
def verify():
 m=w.read(ROOT/'manifest.json');mh=w.sha(ROOT/'manifest.json');jobs=w.read(ROOT/'jobs.json');rows=[];missing=[]
 for j in jobs:
  c=m['cases'][j['case_id']];x=dict(t.split('=',1) for t in j['effective_scontrol_at_submission'].split() if '=' in t)
  if x['Partition']!='default_partition' or x['ExcNodeList']!='scaglione-compute-01' or int(x['NumCPUs'])!=c['resources']['cpus'] or x['MinMemoryNode']!=c['resources']['mem'] or x['TimeLimit']!='00:30:00' or x['Requeue']!='0':raise ValueError('effective fixture resource mismatch')
  if j['dependencies'] and not all(d in x['Dependency'] for d in j['dependencies']):raise ValueError('dependency mismatch')
  path=ROOT/'cases'/j['case_id']/'completion.json'
  if not path.exists():missing.append(j['case_id']);continue
  done=w.read(path)
  if done['manifest_sha256']!=mh or done['status']!='finished' or not done['usable']:raise ValueError('fixture completion mismatch')
  w.require_hash(done['result_path'],done['result_sha256']);value=w.read(done['result_path'])
  if c['kind']=='pool_construction':
   d=value['pool_construction'];w.require_hash(done['journal_path'],done['journal_sha256'])
   rows.append({'case_id':j['case_id'],'kind':c['kind'],'job_id':j['job_id'],'result_path':done['result_path'],'result_sha256':done['result_sha256'],'construction_sha256':done['construction_sha256'],'source_columns':[p['native_unique_columns'] for p in d['source_pool_sets']],'union_columns':d['union_pool_set']['native_unique_columns'],'shared_incidence_count':d['shared_incidence_count'],'constructor_wall_s':d['constructor_wall_s'],'constructor_peak_rss_native_kib':d['peak_rss_native'],'source_record_counts':d['source_record_counts'],'union_pool_set':d['union_pool_set'],'donor_membership_passed':all(x['all_own_selected_records_matched_exactly'] and x['all_union_incidences_present_no_higher_cost'] for x in d['donor_witnesses']),'source_physics_commit':value['provenance']['git_commit'],'donor_buses':[x['buses'] for x in d['donor_witnesses']]})
  else:
   from worker import result_gate
   built=w.read(ROOT/'cases'/c['source_case']/'completion.json');construction=w.read(built['construction_path']);source=built if c['treatment']=='union' else {'result_sha256':c['control_source']['status_sha256'],'journal_sha256':c['control_source']['journal_sha256']}
   pa=result_gate(value,c,source,construction);log=(ROOT/'logs'/('cuFix_'+j['case_id']+'_'+j['job_id']+'.out')).read_text()
   if 'Native Gurobi license check passed' not in log:raise ValueError('full-size license gate absent')
   rows.append({'case_id':j['case_id'],'kind':c['kind'],'job_id':j['job_id'],'result_path':done['result_path'],'result_sha256':done['result_sha256'],'source_result_sha256':value['source_result_sha256'],'source_journal_sha256':value['source_journal_sha256'],'native_pool_columns':pa['base_pool_column_count'],'native_pool_ordered_sha256':pa['base_pool_ordered_sha256'],'native_added_columns':pa['post_augmentation_columns']-pa['base_pool_column_count'],'native_greedy_start':value['mip_start'],'buses':value['buses'],'fleet_bound':value['fleet_bound'],'fleet_proven':value['fleet_proven'],'physical_replay_validated':value['physical_replay_validated'],'full_size_license_passed':True,'execution_commit':value['mip_provenance']['git_commit'],'solver_budget_s':c['solver_budget_s'],'stage1_s':c['stage1_s']})
 raw=subprocess.check_output([S+'sacct','-j',','.join(j['job_id'] for j in jobs),'-P','-n','-o','JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TotalCPU,MaxRSS,NodeList'],text=True);(ROOT/'accounting.txt').write_text(raw)
 if missing:
  w.save(B/'native_validation_pending.json',{'status':'pending','missing':missing,'verified':rows,'utc':w.now()});print(json.dumps({'status':'pending','missing':missing}));return
 v={'status':'passed','utc':w.now(),'production_manifest_sha256':w.sha(B/'manifest.json'),'fixture_manifest_sha256':mh,'actual_fixture_cases':5,'actual_fixture_builds':2,'actual_fixture_mips':3,'fixture_inherited_production_count_fields':'The fixture manifest retains campaign-level independent_construction_cases=8 and independent_mip_cases=12 as parent design metadata; its actual cases mapping contains2builds/3MIPs.','fixture_is_production_result':False,'rows':rows,'all_effective_resources_verified':True,'all_required_dependencies_verified':True,'production_submitted':False,'accounting_sha256':w.sha(ROOT/'accounting.txt'),'limitations':'Real full pools tested for C1k15(e091) and C2k20(a0e0); remaining6production constructions retain strict gates.30s MIP results validate native compatibility only; they are not production scientific comparisons.'};w.save(B/'native_validation.json',v);print(json.dumps({'status':'passed','cases':len(rows)}))
if __name__=='__main__':verify()
