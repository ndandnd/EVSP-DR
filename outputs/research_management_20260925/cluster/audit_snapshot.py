"""Derive compact tables and consistency checks from the read-only snapshot."""
from pathlib import Path
import csv, hashlib, json

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT/'snapshot.json').read_text())
details = json.loads((ROOT/'dive_details.json').read_text())
checks = []

def check(name, value):
    checks.append({'check': name, 'passed': bool(value)})

def csvfile(name, rows):
    with (ROOT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def accounting(section):
    return {x.split('|')[0]: x.split('|') for x in d[section]['stdout'].splitlines() if '.' not in x.split('|')[0]}

acc = accounting('full40_sacct')
rows = []
for cid, a in sorted(d['full40_artifacts'].items()):
    cg, m, gate = a['cg.json']['content'], a['mip_result.json']['content'], a['cg_provenance.json']['content']
    info = d['full40_manifest']['content']['cases'][cid]
    ids = d['full40_case_jobs']['content'][cid]
    check(cid+' authenticated source result', a['cg.json']['sha256'] == gate['result_sha256'] == m['source_result_sha256'])
    check(cid+' bound source journal', gate['journal_sha256'] == m['source_journal_sha256'])
    check(cid+' input matches frozen manifest', info['input_sha256'] == cg['provenance']['instance_sha256'] == m['physical_pool_audit']['input_hashes']['instance_sha256'])
    check(cid+' execution commits', cg['provenance']['git_commit'] == d['full40_manifest']['content']['execution_commit'] and m['mip_provenance']['git_commit'] == d['full40_manifest']['content']['mip_execution_commit'])
    for mode in ['cg', 'mip']:
        check(cid+' '+mode+' scheduler completion', acc[str(ids[mode])][2:4] == ['COMPLETED', '0:0'])
    check(cid+' separate proof and validation flags', not cg['certified_rc_optimal'] and not m['fleet_proven'] and m['physical_replay_validated'] and not m['duplicate_trip_removal_validated'] and not m['cross_route_charger_capacity_validated'])
    previous = info.get('previous_case')
    if previous in d['full40_artifacts']:
        check(cid+' true previous-k source', cg['inherited_event_pool_audit']['source_status_sha256'] == d['full40_artifacts'][previous]['cg.json']['sha256'])
    rows.append({'case_id': cid, 'chain': info['chain'], 'k': info['k'], 'trip_count': info['trip_count'], 'input_sha256': info['input_sha256'], 'cg_job': ids['cg'], 'mip_job': ids['mip'], 'cg_scheduler': acc[str(ids['cg'])][2], 'mip_scheduler': acc[str(ids['mip'])][2], 'cg_stop': cg['stop_reason'], 'pricing_certified': cg['certified_rc_optimal'], 'rmp_route_weight': cg['final']['route_weight'], 'rmp_weighted_objective': cg['final']['lp_obj'], 'pool_columns': m['pool_columns'], 'mip_status': m['status_name'], 'fleet_incumbent': m['buses'], 'finite_pool_fleet_only_bound': m['fleet_bound'], 'finite_pool_fleet_proven': m['fleet_proven'], 'own_target_matched': m['buses'] <= info['k'], 'individual_route_replay': m['physical_replay_validated'], 'overcovered_trips': m['overcovered_trips'], 'duplicate_trip_removal_validated': m['duplicate_trip_removal_validated'], 'shared_capacity_validated': m['cross_route_charger_capacity_validated'], 'cg_path': a['cg.json']['path'], 'cg_sha256': a['cg.json']['sha256'], 'pool_journal_sha256': gate['journal_sha256'], 'mip_path': a['mip_result.json']['path'], 'mip_sha256': a['mip_result.json']['sha256']})
csvfile('full40_cases.csv', rows)
csvfile('full40_k40.csv', [x for x in rows if x['k'] == 40])
dive_rows = []
for x in d['dive_artifacts']:
    if not x['path'].endswith('/execution.json'): continue
    e = x['content']; base = x['path'].rsplit('/', 1)[0]
    a = next(v for v in d['dive_artifacts'] if v['path'] == base+'/without_start/result.json')
    m = a['content']
    manifest = next(v for v in details if v['path'] == base+'/dive/manifest.json')
    mm = manifest['content']
    check(base+' endpoint hash', a['sha256'] == e['without_start']['result_sha256'] == e['output_sha256']['without_start/result.json'])
    check(base+' manifest hash', manifest['sha256'] == e['output_sha256']['dive/manifest.json'])
    check(base+' source journal bound', e['output_sha256']['dive/cg.json.columns.jsonl'] == m['source_journal_sha256'])
    check(base+' cap/stopping identity', mm['stop_policy'] == e['stop_policy'] and mm['fleet_cap'] == (8 if e['cap_arm'] == 'floor8' else 9))
    check(base+' source pins', mm['git_commit'] == d['dive_manifest']['content']['execution_commit'] and e['source_immutable'])
    log = next(v for v in details if v['path'] == base+'/without_start/gurobi.log')
    check(base+' native solver log identity', log['sha256'] == e['output_sha256']['without_start/gurobi.log'])
    check(base+' no incumbent transfer comparison', mm['incumbent_export'] is None and e['with_start']['skipped'] == 'no validated dive incumbent')
    dive_rows.append({'job_id': mm['slurm_job_id'], 'cap': e['cap_arm'], 'stop_policy': e['stop_policy'], 'dive_stop': e['dive_stop_reason'], 'dive_budget_s': e['dive_budget_s'], 'dive_wall_s': e['dive_wall_s'], 'dive_integer_buses': e['dive_integer_buses'], 'new_columns': e['columns_generated'], 'mip_budget_s': e['mip_budget_each_s'], 'mip_solver_runtime_s': m['runtime_s'], 'mip_fleet': m['buses'], 'finite_pool_fleet_only_bound': m['fleet_bound'], 'fleet_proven': m['fleet_proven'], 'mip_status': m['status_name'], 'with_start_skipped': e['with_start']['skipped'], 'individual_route_replay': m['physical_replay_validated'], 'overcovered_trips': m['overcovered_trips'], 'duplicate_trip_removal_validated': m['duplicate_trip_removal_validated'], 'shared_capacity_validated': m['cross_route_charger_capacity_validated'], 'execution_path': x['path'], 'execution_sha256': x['sha256'], 'mip_sha256': a['sha256']})
csvfile('dive_cap_cases.csv', sorted(dive_rows, key=lambda x: x['job_id']))
queue = [dict(zip(['job_id','name','partition','state','elapsed','limit','reason_or_node','dependencies'], x.split('|'))) for x in d['queue']['stdout'].splitlines()]
spatial = [x for x in queue if x['name'].startswith('st_')]
csvfile('active_spatial_jobs.csv', spatial)
scontrol = d['scontrol_all_user']['stdout'].splitlines()
for job in spatial:
    control = next(x for x in scontrol if x.startswith('JobId='+job['job_id']+' '))
    check(job['job_id']+' reserved compute node excluded', 'ExcNodeList=scaglione-compute-01 ' in control)
    if job['state'] == 'PENDING':
        pred = job['dependencies'].split(':')[1].split('(')[0]
        check(job['job_id']+' genuine running predecessor', any(x['job_id'] == pred and x['state'] == 'RUNNING' for x in spatial))
summary = {'captured_utc': d['captured_utc'], 'full40_cg_completed': len(rows), 'full40_mip_completed': len(rows), 'full40_pricing_certificates': sum(x['pricing_certified'] for x in rows), 'full40_finite_pool_fleet_proofs': sum(x['finite_pool_fleet_proven'] for x in rows), 'full40_own_target_matches': sum(x['own_target_matched'] for x in rows), 'full40_individual_replays_passed': sum(x['individual_route_replay'] for x in rows), 'full40_k40_fleets': [x['fleet_incumbent'] for x in rows if x['k'] == 40], 'spatial_running': sum(x['state'] == 'RUNNING' for x in spatial), 'spatial_running_cg': sum(x['state'] == 'RUNNING' and x['name'].startswith('st_cg') for x in spatial), 'spatial_running_cleanup': sum(x['state'] == 'RUNNING' and x['name'].startswith('st_cl') for x in spatial), 'spatial_pending_true_dependencies': sum(x['state'] == 'PENDING' for x in spatial), 'dive_cap_completed': len(dive_rows), 'dive_cap_integer_hits': sum(x['dive_integer_buses'] is not None for x in dive_rows), 'dive_cap_fleet8_mip_hits': sum(x['mip_fleet'] <= 8 for x in dive_rows), 'checks_passed': sum(x['passed'] for x in checks), 'checks_total': len(checks)}
(ROOT/'verification.json').write_text(json.dumps({'summary': summary, 'checks': checks}, indent=2)+'\n')
for section in ['queue','full40_sacct','dive_sacct','spatial_sacct','scontrol_all_user']:
    (ROOT/(section+'.txt')).write_text(d[section]['stdout'])
(ROOT/'SCAGLIONE_RESOURCE_POLICY.md').write_text(d['policy']['text'])
print(json.dumps(summary, indent=2))
assert all(x['passed'] for x in checks), [x for x in checks if not x['passed']]
