"""Run on Unicorn with Python; emits compact raw evidence, never reads column journals."""
from pathlib import Path
from collections import Counter, defaultdict
import json, datetime, subprocess, hashlib

home = Path.home() / 'ladder-lite'
roots = {
    'covering_complement75': home / 'covering_complement75_20260911_21fbecb',
    'warm_multichain_p1246': home / 'nested_warm_multichain_p1246_k2_10_20260910_ecb60c1',
    'warm_chain_p3_k2_10': home / 'nested_warm_chain_p3_k2_10_20260909_8830a34',
    'warm_chain_p5_k2_10': home / 'nested_warm_chain_p5_k2_10_20260910_ecb60c1',
    'covering_rerun9': home / 'covering_rerun9_20260909_21fbecb',
    'historical_cover_controls': home / 'historical_cover_controls_20260909_bead344',
    'nested_replication42': home / 'nested_replication_p7_20_fresh42_20260909_0209d3e',
    'overnight_cpu04_raw_controls': home / 'overnight_cpu04_raw_controls_20260909',
    'greedy_event4': home / 'greedy_event4_eaca565',
    'easy_k10_raw_split_replay': home / 'easy_k10_raw_split_replay_20260908_9665429',
    'matched_tariff_peak12_peak18': home / 'matched_tariff_peak12_peak18_a6e5059',
    'heavy_raw6': home / 'heavy_trip_ladder_raw6_20260908_9665429',
    'matched_tariff8': home / 'matched_tariff8_20260908_a9a9720',
    'legacy_selected10_current2': home / 'legacy_selected10_current2_20260908_bead344',
    'easy_raw6': home / 'easy_trip_ladder_raw6_20260908_aaf83f6',
    'nested84': home / 'nested_probability_k2_15_fresh84_20260908_21fbecb',
    'old70': home / 'threshold_9_15_event_20260904_9bdbb17/pool_mip8h_20260908_2d7a21a07e',
    'stage2_cap_saved_pool_reruns': home / 'stage2_cap_saved_pool_reruns_20260910_15e781a',
    'stage2_cap_license_recovery': home / 'stage2_cap_license_recovery_20260910_871d057',
    'terminal_energy_fair_mip_retry_5cdb813': home / 'terminal_energy_fair_mip_retry_5cdb813_20260910',
}
out = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'campaigns': {}}
for name, root in roots.items():
    rows = []
    for p in sorted(set(root.rglob('*mip8h.json')) | set(root.rglob('*mip_budgeted.json')) | set(root.glob('*/mip.json')) | set((root/'results').glob('*60m.json')) | set((root/'mip').glob('*1h2stage.json'))):
        raw = p.read_bytes()
        d = json.loads(raw)
        row = {k: d.get(k) for k in ['buses','fleet_bound','fleet_proven','status','status_name','mip_gap','optimal_scope','pool_columns','source_cg_iterations','source_cg_wall_s','runtime_s','gurobi_optimize_wall_s','partitioning','overcovered_trips','charging_cost','continuous_realized_charging_cost','two_stage','physical_pool_audit','physical_replay_validated','physical_replay_scope','duplicate_trip_removal_validated','cross_route_charger_capacity_validated']}
        row.update(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), file_mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat(), instance=d.get('instance'), selected_physical_statuses=dict(Counter(r.get('physical_realization',{}).get('status','missing') for r in d.get('selected_routes',[]))))
        if name == 'stage2_cap_license_recovery' and row['sha256'] == 'fa05bf422221abecfa37f3eb6eeebf07f586bfde643057fb073fe97be5cadf3a':
            row['authoritative_772009_completion'] = False
            row['provenance_warning'] = 'Shared path contains the 57-second INTERRUPTED artifact from canceled duplicate772031. It is not the authoritative772009 result. The validated772009 pending result was later recovered under a unique job-specific filename; see post-meeting license report.'
        if name == 'stage2_cap_license_recovery' and row['sha256'] == 'd9b7cdfa52eff7cf254a602bcb1d31f5bb1665e98eab460c70a8b3d32cc46996':
            row['authoritative_772009_result'] = True
            row['publication_recovered_without_optimization'] = True
            row['scheduler_failure_scope'] = 'Output-publication collision after solver termination and successful selected-route validation; scheduler FAILED is preserved.'
        row.update(physics=d.get('physics'), mip_start=d.get('mip_start'), column_pool_treatment=(d.get('pricer_provenance') or {}).get('column_pool_treatment'))
        rows.append(row)
    cg = []
    cg_paths=set((root/'cg').glob('*.json')) | set(root.glob('*/cg.json'))
    if cg_paths:
        for p in sorted(cg_paths):
            if p.stat().st_size > 100_000_000: continue
            try: d=json.loads(p.read_bytes())
            except Exception: continue
            row={k:v for k,v in d.items() if k not in ['routes','columns','selected_routes','iterations','history','iteration_log'] and not isinstance(v,list)}
            row['path']=str(p)
            cg.append(row)
    phases=[]
    phase_paths=set((root/'cg').glob('*.phase-telemetry.jsonl')) | set(root.glob('*/cg.phase-telemetry.jsonl'))
    for p in sorted(phase_paths):
        sums=defaultdict(float); counts=Counter(); last=None; partial=0
        for line in p.open():
            try: d=json.loads(line)
            except json.JSONDecodeError: partial+=1;continue
            if d.get('record_type')=='phase':
                sums[d['phase']]+=d.get('duration_s',0); counts[d['phase']]+=1
            last=d
        phases.append({'path':str(p),'duration_s_by_phase':dict(sums),'count_by_phase':dict(counts),'last_record':last,'partial_lines':partial})
    comparisons=[]
    for p in sorted(root.glob('*/comparison.json')):
        raw=p.read_bytes()
        comparisons.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'result':json.loads(raw)})
    rejected=[]
    for p in sorted((root/'mip').glob('*.rejected_physical_replay.json')):
        raw=p.read_bytes()
        rejected.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'result':json.loads(raw)})
    out['campaigns'][name]={'root':str(root),'mip':rows,'cg':cg,'phases':phases,'comparisons':comparisons,'rejected_mip_outputs':rejected}
    out['campaigns'][name]['workflow'] = {}
    for record_name in ['workflow_submission.json', 'mip_submission.json', 'mip_retry2_submission.json', 'submission.json', 'submission.mip.json', 'retry_manifest.json', 'repair_submission.json', 'publication_recovery_772009.json', 'manifests/submission_initial.json', 'manifests/submission_final.json', 'execution_plan.json']:
        record_path = root / record_name
        if record_path.exists():
            out['campaigns'][name]['workflow'][record_name] = json.loads(record_path.read_bytes())
    for record_name in ['cg_jobs.tsv', 'freeze_mip_jobs.tsv']:
        record_path = root / record_name
        if record_path.exists():
            out['campaigns'][name]['workflow'][record_name] = record_path.read_text()
# Controlled post-meeting reference CG: different schema from production CG.
pilot_root = home / 'capacity_speed_pilot_20260910_v2_7d38efd'
pilot_records = []
for phase, filename in [('cg', 'cg.json'), ('mip', 'mip_twostage.json')]:
    for p in sorted((pilot_root/'results').glob(f'*/{filename}')):
        try:
            raw = p.read_bytes()
            d = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        compact = {k:v for k,v in d.items() if k not in ['routes','selected_routes','iterations','history','columns','route_values','trip_duals','capacity_duals']}
        if isinstance(d.get('iterations'), list):
            compact['iteration_count'] = len(d['iterations'])
            compact['last_iteration'] = d['iterations'][-1] if d['iterations'] else None
        pilot_records.append({'phase':phase, 'path':str(p), 'sha256':hashlib.sha256(raw).hexdigest(), 'result':compact})
out['capacity_speed_pilot'] = {'root':str(pilot_root), 'records':pilot_records}
submission_path = pilot_root/'submission.json'
if submission_path.exists():
    out['capacity_speed_pilot']['submission'] = json.loads(submission_path.read_bytes())
corrected_submission_path = pilot_root/'submission_twostage.json'
if corrected_submission_path.exists():
    out['capacity_speed_pilot']['authoritative_mip_submission'] = json.loads(corrected_submission_path.read_bytes())
# Matched aggregate terminal-energy experiment. Retain proof scopes and energy
# totals, but do not return hundreds of route records in every monitor snapshot.
terminal_root = home / 'terminal_energy_fair_20260910_2424369'
terminal_records = []
def without_selected_routes(value):
    return {k:v for k,v in value.items() if k != 'selected_routes'}
for cell in ['peak08', 'peak12', 'peak18']:
    folder = terminal_root / cell
    for phase, filename, marker_name in [('frontier', 'frontier.json', 'FRONTIER_COMPLETE.json'), ('comparison', 'comparison.json', 'COMPLETE.json')]:
        p = folder / filename
        if not p.exists():
            continue
        try:
            raw = p.read_bytes()
            d = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        compact = {k:v for k,v in d.items() if k != 'frontiers'}
        for key in ['fixed_solution', 'fixed_duties_optimized', 'joint_pool_optimized']:
            if isinstance(compact.get(key), dict):
                compact[key] = without_selected_routes(compact[key])
        if 'frontiers' in d:
            compact['frontier_options_by_duty'] = {r['duty_id']:len(r['routes']) for r in d['frontiers']}
        marker_path = folder / marker_name
        marker = json.loads(marker_path.read_bytes()) if marker_path.exists() else None
        digest = hashlib.sha256(raw).hexdigest()
        marker_key = 'frontier_sha256' if phase == 'frontier' else 'comparison_sha256'
        terminal_records.append({'cell':cell, 'phase':phase, 'path':str(p), 'sha256':digest, 'completion_marker_matches':bool(marker and marker.get(marker_key) == digest), 'result':compact})
out['terminal_energy_fair'] = {'root':str(terminal_root), 'records':terminal_records, 'workflow':{}}
for filename in ['plan.json', 'submission.frontier.json', 'submission.mip.json']:
    p = terminal_root / filename
    if p.exists():
        out['terminal_energy_fair']['workflow'][filename] = json.loads(p.read_bytes())
out['terminal_energy_fair']['stderr_tails'] = {
    str(p):p.read_text(errors='replace').splitlines()[-20:]
    for p in (terminal_root/'logs').glob('*.err') if p.stat().st_size
}
# Early solver progress can precede final validated JSON by an hour.
recovery_root = home/'stage2_cap_license_recovery_20260910_871d057'
out['post_meeting_log_tails'] = {}
for p in sorted((recovery_root/'logs').glob('*.out')):
    lines = p.read_text(errors='replace').splitlines()
    milestones = [line for line in lines if any(s in line for s in ['preflight OK','stage 1:','stage 2:','Loaded user MIP','Traceback','Error:'])]
    out['post_meeting_log_tails'][str(p)] = {'milestones':milestones, 'tail':lines[-12:]}
# Isolated new-physics fleet-only prototype; schema differs from legacy MIPs.
for small_name, directory in [('giro_small_cg_newphysics', 'giro_small_cg_20260909'), ('giro_small_cg_capacity_duals', 'giro_small_cg_capacity_duals_20260909')]:
    small_root = home / directory
    small_rows = []
    for p in sorted((small_root / 'results').glob('*/result.json')):
        raw = p.read_bytes()
        d = json.loads(raw)
        small_rows.append({
            'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(),
            'instance': d.get('instance'), 'trip_count': d.get('trip_count'),
            'vehicle_profile': d.get('vehicle_profile'), 'seed_mode': d.get('seed_mode'),
            'shared_union_pool_size': d.get('shared_union_pool_size'),
            'reporting_scope': d.get('reporting_scope'),
            'cg_arms': [{k: v for k, v in arm.items() if k != 'iterations'} for arm in d.get('cg_arms', [])],
            'final_same_pool_comparisons': {
                cap: {sense: {
                    'lp': {k: v for k, v in result['lp'].items() if k not in ['trip_duals', 'route_values', 'capacity_duals_ignored_by_pricing']},
                    'mip': {k: v for k, v in result['mip'].items() if k != 'selected_indices'}
                } for sense, result in senses.items()}
                for cap, senses in d.get('final_same_pool_comparisons', {}).items()
            }
        })
    out[small_name] = {'root': str(small_root), 'results': small_rows}
# Lightweight evidence/status for targeted diagnosis jobs.
out['targeted_audits'] = {}
for name, directory in [('april_source_replay','april175_replay_20260909'), ('full_cache_witness','full_cache_witness_audit_20260909_1745'), ('legacy_full_cache_witness','full_cache_witness_legacy175_20260909_1755')]:
    root = home / directory
    paths = set(root.glob('submission.json')) | set((root/'results').glob('*.json'))
    rows = []
    for p in sorted(paths):
        if p.stat().st_size > 2_000_000: continue
        try:
            raw=p.read_bytes(); d=json.loads(raw)
            rows.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'data':d})
        except Exception: pass
    tails={str(p):'\n'.join(p.read_text(errors='replace').splitlines()[-12:]) for p in (root/'results').glob('*.log')}
    out['targeted_audits'][name]={'root':str(root),'records':rows,'log_tails':tails}

q=subprocess.run(['/usr/local/slurm/slurm-25.05.5/bin/squeue','--me','-r','-h','-o','%i|%T|%M|%R'],capture_output=True,text=True)
out['squeue']={'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr}
print(json.dumps(out))
