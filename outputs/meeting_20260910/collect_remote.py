"""Run on Unicorn with Python; emits compact raw evidence, never reads column journals."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, datetime, subprocess, hashlib, math


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _giro_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def giro_instance_trip_ids(path):
    """Read only the small instance CSV to verify selected-route coverage."""
    if not path:
        return None
    try:
        with Path(path).open(newline='') as stream:
            rows = csv.DictReader(stream)
            values = [int(row['count_trip_id']) for row in rows]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return values if values else None


def giro_selected_metrics(summary, fee, expected_trip_ids=None,
                          target_terminal_kwh=None):
    """Compact selected routes into auditable charging components."""
    if not isinstance(summary, dict):
        return None
    keep = [
        'fleet', 'expanded_grid_charging_cost', 'physical_charging_cost',
        'expanded_grid_terminal_energy_kwh', 'continuous_terminal_energy_kwh',
        'overcovered_trip_count', 'maximum_trip_multiplicity',
        'charge_start_count', 'charge_start_counts_by_route',
    ]
    compact = {key: summary[key] for key in keep if key in summary}
    routes = summary.get('selected_routes')
    metrics = {}
    if isinstance(routes, list):
        grid_starts = grid_kwh = continuous_starts = continuous_kwh = 0
        continuous_electricity = 0.0
        grid_known = continuous_known = continuous_price_known = True
        route_terminals_grid = []
        route_terminals_continuous = []
        route_statuses = []
        coverage = Counter()
        coverage_known = True
        for route in routes:
            if not isinstance(route, dict):
                grid_known = continuous_known = continuous_price_known = False
                route_statuses.append('missing')
                coverage_known = False
                continue
            physical = route.get('physical_realization')
            if not isinstance(physical, dict):
                physical = {}
            route_statuses.append(
                physical.get('status') or route.get('physical_replay_status')
                or 'missing'
            )
            trips = route.get('trips')
            if not isinstance(trips, list):
                coverage_known = False
            else:
                coverage.update(trips)
            realization = route.get('continuous_realization')
            if not isinstance(realization, dict):
                realization = {}
            grid_terminal = _giro_float(
                realization.get('expanded_grid_terminal_soc_kwh'))
            continuous_terminal = _giro_float(
                realization.get('continuous_terminal_soc_kwh'))
            if grid_terminal is None:
                route_terminals_grid = None
            elif route_terminals_grid is not None:
                route_terminals_grid.append(grid_terminal)
            if continuous_terminal is None:
                route_terminals_continuous = None
            elif route_terminals_continuous is not None:
                route_terminals_continuous.append(continuous_terminal)
            stops = route.get('expanded_grid_charging_stops')
            if not isinstance(stops, dict) or not isinstance(stops.get('stations'), list):
                grid_known = False
            else:
                station_count = len(stops['stations'])
                grid_starts += station_count
                if (not isinstance(stops.get('kwh'), list)
                        or len(stops['kwh']) != station_count
                        or any(_giro_float(value) is None for value in stops['kwh'])):
                    grid_known = False
                else:
                    grid_kwh += sum(_giro_float(value) for value in stops['kwh'])
            blocks = route.get('continuous_realized_charging_blocks')
            if not isinstance(blocks, list):
                continuous_known = False
            else:
                stop_ids = set()
                for block in blocks:
                    if (not isinstance(block, dict)
                            or block.get('stop_index') is None
                            or _giro_float(block.get('realized_kwh')) is None):
                        continuous_known = False
                        continuous_price_known = False
                        break
                    stop_ids.add(block['stop_index'])
                    realized_kwh = _giro_float(block['realized_kwh'])
                    continuous_kwh += realized_kwh
                    price = _giro_float(block.get('price_per_kwh'))
                    if price is None:
                        continuous_price_known = False
                    else:
                        continuous_electricity += price * realized_kwh
                continuous_starts += len(stop_ids)
        if grid_known:
            metrics.update(
                expanded_grid_charging_starts=grid_starts,
                expanded_grid_charging_kwh=grid_kwh,
            )
        if continuous_known:
            metrics.update(
                continuous_charging_starts=continuous_starts,
                continuous_charging_kwh=continuous_kwh,
            )
    # Grid route costs do not carry per-block prices, so their electricity
    # component is explicitly labeled as derived. Continuous blocks can
    # provide an independent price-times-energy sum; use it when complete.
    fee_value = _giro_float(fee)
    for model, total_key, starts_key in (
        ('expanded_grid', 'expanded_grid_charging_cost', 'expanded_grid_charging_starts'),
        ('continuous', 'physical_charging_cost', 'continuous_charging_starts'),
    ):
        total = _giro_float(compact.get(total_key))
        starts = metrics.get(starts_key)
        if total is not None and starts is not None and fee_value is not None:
            start_fees = fee_value * starts
            metrics[model + '_charge_start_fees'] = start_fees
            metrics[model + '_electricity_cost'] = total - start_fees
            metrics[model + '_electricity_cost_basis'] = (
                'derived_total_minus_charge_start_fees')
    if (metrics.get('continuous_charging_starts') is not None
            and continuous_price_known):
        metrics['continuous_electricity_cost'] = continuous_electricity
        metrics['continuous_electricity_cost_basis'] = (
            'sum_continuous_realized_charging_blocks_price_x_kwh')
    if isinstance(routes, list):
        replay_valid = bool(routes) and all(
            status in {'valid_event_time_realized', 'validated', 'valid',
                       'validated_all_routes'}
            for status in route_statuses
        )
        coverage_complete = None
        if expected_trip_ids is not None:
            coverage_complete = (
                coverage_known
                and set(coverage) >= set(expected_trip_ids)
                and all(coverage[trip] >= 1 for trip in expected_trip_ids)
            )
        summary_grid = _giro_float(
            compact.get('expanded_grid_terminal_energy_kwh'))
        summary_continuous = _giro_float(
            compact.get('continuous_terminal_energy_kwh'))
        route_grid = (
            sum(route_terminals_grid)
            if route_terminals_grid is not None else None
        )
        route_continuous = (
            sum(route_terminals_continuous)
            if route_terminals_continuous is not None else None
        )
        target = _giro_float(target_terminal_kwh)
        grid_match = (route_grid is not None and summary_grid is not None
                      and math.isclose(route_grid, summary_grid,
                                       abs_tol=1e-7, rel_tol=1e-10))
        continuous_match = (
            route_continuous is not None and summary_continuous is not None
            and math.isclose(route_continuous, summary_continuous,
                             abs_tol=1e-7, rel_tol=1e-10)
        )
        grid_meets = target is not None and route_grid is not None and route_grid + 1e-7 >= target
        continuous_meets = target is not None and route_continuous is not None and route_continuous + 1e-7 >= target
        metrics['physical_validation'] = {
            'selected_route_replay_statuses': dict(Counter(route_statuses)),
            'selected_route_replays_valid': replay_valid,
            'coverage_complete': coverage_complete,
            'coverage_trip_count': len(expected_trip_ids) if expected_trip_ids is not None else None,
            'grid_terminal_total_from_routes_kwh': route_grid,
            'continuous_terminal_total_from_routes_kwh': route_continuous,
            'grid_terminal_total_matches_summary': grid_match,
            'continuous_terminal_total_matches_summary': continuous_match,
            'grid_terminal_meets_target': grid_meets,
            'continuous_terminal_meets_target': continuous_meets,
            'validated': all((replay_valid, coverage_complete is True,
                              grid_match, continuous_match,
                              grid_meets, continuous_meets)),
        }
    if metrics:
        compact['charging_comparison_metrics'] = metrics
    return compact


def giro_original_metrics(original, fee):
    if not isinstance(original, dict):
        return None
    fields = [
        'fleet', 'charge_events', 'charge_start_fees', 'charging_cost_exact',
        'charging_cost_lower', 'charging_cost_upper', 'energy_cost_exact',
        'energy_cost_lower', 'energy_cost_upper', 'total_charged_kwh',
        'terminal_surplus_total_kwh', 'matched_physics_comparator_eligible',
        'comparator', 'cost_errors', 'charge_start_fee_scope',
    ]
    compact = {key: original[key] for key in fields if key in original}
    metrics = {}
    if original.get('charge_events') is not None:
        metrics['continuous_charging_starts'] = original['charge_events']
        metrics['continuous_charge_start_fees'] = float(fee) * original['charge_events']
    if original.get('total_charged_kwh') is not None:
        metrics['continuous_charging_kwh'] = original['total_charged_kwh']
    if original.get('energy_cost_exact') is not None:
        metrics['continuous_electricity_cost'] = original['energy_cost_exact']
    if metrics:
        compact['charging_comparison_metrics'] = metrics
    return compact

home = Path.home() / 'ladder-lite'
roots = {
    'graph_recovery_retry2_20260912': home / 'graph_recovery_retry2_20260912',
    'full_pool_recovery_20260912': home / 'full_pool_recovery_20260912',
    'graph_recovery_20260912': home / 'graph_recovery_20260912',
    'queue_recovery_20260912': home / 'queue_recovery_20260912',
    'w2_chain_recovery_retry2_20260912': home / 'w2_chain_recovery_retry2_20260912',
    'w2_chain_recovery_20260912': home / 'w2_chain_recovery_20260912',
    'w2_k14_import_fix_20260912': home / 'w2_k14_import_fix_20260912',
    'overnight_extension_20260912': home / 'overnight_extension_20260912',
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
    'capacity_deadline5_retry': home / 'capacity_deadline5_retry_20260911_253588e',
    'capacity_timeout6_rerun': home / 'capacity_speed_pilot_20260910_timeout6_rerun_7d38ef',
}
out = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'campaigns': {}}
for name, root in roots.items():
    rows = []
    for p in sorted(set(root.rglob('*mip8h.json')) | set(root.rglob('*mip_budgeted.json')) | set(root.glob('*/mip.json')) | set((root/'results').glob('*60m.json')) | set((root/'mip').glob('*1h2stage.json')) | set(root.glob('p*/mip/*__1h2stage.json')) | set((root/'mip_attempts').glob('**/result.json')) | {p for p in root.glob('cases/*/mip/*/result.json') if 'smoke' not in p.parts}):
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
    cg_paths=set((root/'cg').glob('*.json')) | set(root.glob('p*/cg/M__*.json')) | set(root.glob('*/cg.json')) | {p for p in root.glob('cases/*/cg.json') if 'smoke' not in p.parts}
    if cg_paths:
        for p in sorted(cg_paths):
            if p.stat().st_size > 100_000_000: continue
            try: d=json.loads(p.read_bytes())
            except Exception: continue
            row={k:v for k,v in d.items() if k not in ['routes','columns','selected_routes','iterations','history','iteration_log'] and not isinstance(v,list)}
            row['path']=str(p)
            cg.append(row)
    phases=[]
    phase_paths=set((root/'cg').glob('*.phase-telemetry.jsonl')) | set(root.glob('p*/cg/*.phase-telemetry.jsonl')) | set(root.glob('*/cg.phase-telemetry.jsonl')) | {p for p in root.glob('cases/*/*.phase-telemetry.jsonl') if 'smoke' not in p.parts}
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
    for p in sorted(set(root.glob('*/comparison.json')) | set(root.glob('cases/*/decomposed_solution.json'))):
        raw=p.read_bytes()
        comparisons.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'result':json.loads(raw)})
    rejected=[]
    for p in sorted((root/'mip').glob('*.rejected_physical_replay.json')):
        raw=p.read_bytes()
        rejected.append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'result':json.loads(raw)})
    out['campaigns'][name]={'root':str(root),'mip':rows,'cg':cg,'phases':phases,'comparisons':comparisons,'rejected_mip_outputs':rejected}
    out['campaigns'][name]['workflow'] = {}
    for record_name in ['ready_mips_manifest.json', 'ready_mips_submission.json', 'ready_mips_old_cancellations.json', 'obsolete_chain2_cancellations.json', 'retired_queue_entries.json', 'case_jobs.json', 'cache_jobs.json', 'cache_cli_validation.json', 'cache_compatibility.json', 'manifest.json', 'jobs.json', 'smoke_job.json', 'downstream/mip_concurrency_rebalance_20260911T0932Z.json', 'downstream/dependency_repair_20260911T0831Z.json', 'downstream/default_mip_migration_a01.json', 'downstream/default_mip_migration_a02.json', 'resource_override.json', 'workflow_submission.json', 'mip_submission.json', 'mip_retry2_submission.json', 'submission.json', 'submission.cg.json', 'submission.mip.json', 'retry_manifest.json', 'rerun_manifest.json', 'repair_submission.json', 'publication_recovery_772009.json', 'manifests/submission_initial.json', 'manifests/submission_final.json', 'execution_plan.json']:
        record_path = root / record_name
        if record_path.exists():
            out['campaigns'][name]['workflow'][record_name] = json.loads(record_path.read_bytes())
    for record_name in ['cg_jobs.tsv', 'freeze_mip_jobs.tsv']:
        record_path = root / record_name
        if record_path.exists():
            out['campaigns'][name]['workflow'][record_name] = record_path.read_text()
# Capacity retry uses the same nested results schema as the original pilot.
for retry_name in ['capacity_timeout6_rerun', 'capacity_deadline5_retry']:
    retry_root = roots[retry_name]
    retry_records = []
    for phase, filename in [('cg', 'cg.json'), ('mip', 'mip_twostage.json')]:
        for p in sorted((retry_root/'results').glob(f'*/{filename}')):
            raw = p.read_bytes()
            d = json.loads(raw)
            compact = {k:v for k,v in d.items() if k not in ['routes','selected_routes','iterations','history','columns','route_values','trip_duals','capacity_duals']}
            if isinstance(d.get('iterations'), list):
                compact['iteration_count'] = len(d['iterations'])
                compact['last_iteration'] = d['iterations'][-1] if d['iterations'] else None
            retry_records.append({'phase':phase, 'path':str(p), 'sha256':hashlib.sha256(raw).hexdigest(), 'result':compact})
    out['campaigns'][retry_name]['records'] = retry_records

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
study_script = home / 'mip_preemption_study_20260911' / 'collect_attempts.py'
if study_script.exists() and (study_script.parent / 'registry.json').exists():
    try:
        study_run = subprocess.run(['python3', str(study_script)], capture_output=True, text=True, timeout=45)
        out['mip_preemption_study'] = json.loads(study_run.stdout) if study_run.returncode == 0 else {'collection_error': study_run.stderr, 'returncode': study_run.returncode}
    except Exception as exc:
        out['mip_preemption_study'] = {'collection_error': str(exc)}
# Paired efficiency collections retain original failures and separate warm retries.
for efficiency_key, efficiency_directory in [('efficiency_validation', 'efficiency_validation_20260912'), ('efficiency_validation_warm_retry', 'efficiency_validation_warm_retry_20260912')]:
    efficiency_root = home / efficiency_directory
    efficiency_script = efficiency_root / 'code-baseline/scripts/efficiency_validation_20260912/campaign.py'
    if efficiency_script.exists() and (efficiency_root / 'manifest.json').exists():
        try:
            efficiency_run = subprocess.run(
                ['/home/nc437/evsp_env/bin/python', str(efficiency_script), 'collect', '--root', str(efficiency_root)],
                capture_output=True, text=True, timeout=60)
            if efficiency_run.returncode:
                out[efficiency_key] = {'collection_error': efficiency_run.stderr[-4000:], 'returncode': efficiency_run.returncode}
            else:
                efficiency_data = json.loads(efficiency_run.stdout)
                if efficiency_data.get('schema') != 'evsp-efficiency-collection-v2':
                    raise ValueError('Unexpected efficiency collection schema')
                out[efficiency_key] = efficiency_data
        except Exception as exc:
            out[efficiency_key] = {'collection_error': str(exc)}
# Controlled baseline comparisons use frozen parents and one-factor pairs.
controlled_root = home / 'controlled_comparison_20260913'
controlled_script = controlled_root / 'tooling/campaign.py'
if controlled_script.exists() and (controlled_root / 'manifest.json').exists():
    try:
        collected = subprocess.run(
            ['/home/nc437/evsp_env/bin/python', str(controlled_script), 'collect', '--root', str(controlled_root)],
            capture_output=True, text=True, timeout=60)
        if collected.returncode:
            raise RuntimeError(collected.stderr[-4000:])
        campaign = json.loads(collected.stdout)
        if campaign.get('schema') != 'evsp-controlled-comparison-v1':
            raise ValueError('Unexpected controlled-comparison schema')
        out['campaigns']['controlled_comparison_20260913'] = campaign
    except Exception as exc:
        out['campaigns']['controlled_comparison_20260913'] = {
            'root': str(controlled_root), 'collection_error': str(exc), 'cg': [], 'mip': []}

# Charge-start-fee treatments use separately pinned code and immutable prior pools.
controlled_root = home / 'zero_charge_start_fee_20260913'
controlled_script = controlled_root / 'tooling/campaign.py'
if controlled_script.exists() and (controlled_root / 'manifest.json').exists():
    try:
        collected = subprocess.run(
            ['/home/nc437/evsp_env/bin/python', str(controlled_script), 'collect', '--root', str(controlled_root)],
            capture_output=True, text=True, timeout=60)
        if collected.returncode:
            raise RuntimeError(collected.stderr[-4000:])
        campaign = json.loads(collected.stdout)
        if campaign.get('schema') != 'evsp-zero-charge-start-fee-v1':
            raise ValueError('Unexpected zero-charge-fee schema')
        out['campaigns']['zero_charge_start_fee_20260913'] = campaign
    except Exception as exc:
        out['campaigns']['zero_charge_start_fee_20260913'] = {
            'root': str(controlled_root), 'collection_error': str(exc), 'cg': [], 'mip': []}

# Fee-only GIRO comparison.  The campaign collector owns its compact six-cell
# status; this layer authenticates the actual comparison/MIP bytes and adds
# route-free cost, energy, and charging-activity summaries for the register.
giro_root = home / 'giro_zero_start_fee_20260913'
giro_plan_path = giro_root / 'plan.json'
giro_summary_path = giro_root / 'summary.json'
giro_campaign = {
    'schema': 'evsp-dr-terminal-energy-fee-comparison-collection-v1',
    'root': str(giro_root), 'cells': [], 'comparisons': [], 'workflow': {},
}
for record_name in ['plan.json', 'jobs.json']:
    record_path = giro_root / record_name
    if record_path.exists():
        try:
            giro_campaign['workflow'][record_name] = json.loads(record_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            giro_campaign.setdefault('workflow_read_errors', {})[record_name] = str(exc)
if giro_plan_path.exists():
    try:
        giro_plan = json.loads(giro_plan_path.read_bytes())
        if giro_plan.get('schema') != 'evsp-dr-terminal-energy-fee-comparison-v1':
            raise ValueError('Unexpected GIRO fee-comparison plan schema')
        giro_campaign.update(
            plan_path=str(giro_plan_path), plan_sha256=file_sha256(giro_plan_path),
            code_commit=giro_plan.get('commit'), execution_repo=giro_plan.get('execution_repo'),
            proof_scope=giro_plan.get('proof_scope'),
            target_physical_terminal_energy_kwh=giro_plan.get(
                'target_physical_terminal_energy_kwh'),
        )
        collector_script = Path(giro_plan.get('execution_repo') or home) / (
            'scripts/event_uniform_envelope/giro_zero_fee_campaign.py')
        if collector_script.exists():
            collected = subprocess.run(
                [giro_plan.get('python') or '/home/nc437/evsp_env/bin/python',
                 str(collector_script), 'collect', '--root', str(giro_root)],
                capture_output=True, text=True, timeout=60,
            )
            if collected.returncode:
                giro_campaign['collection_error'] = collected.stderr[-4000:]
                giro_campaign['collection_returncode'] = collected.returncode
        else:
            giro_campaign['collection_error'] = (
                'GIRO collector script is missing: ' + str(collector_script))
    except Exception as exc:
        giro_campaign['collection_error'] = str(exc)
if giro_summary_path.exists():
    try:
        giro_summary_raw = giro_summary_path.read_bytes()
        giro_summary = json.loads(giro_summary_raw)
        if giro_summary.get('schema') != 'evsp-dr-terminal-energy-fee-comparison-summary-v1':
            raise ValueError('Unexpected GIRO fee-comparison summary schema')
        giro_campaign['summary_path'] = str(giro_summary_path)
        giro_campaign['summary_sha256'] = hashlib.sha256(giro_summary_raw).hexdigest()
        giro_campaign['cells'] = giro_summary.get('cells') or []
        if giro_campaign.get('plan_sha256') is not None:
            giro_campaign['summary_plan_sha256_matches'] = (
                giro_summary.get('plan_sha256') == giro_campaign['plan_sha256'])
            if not giro_campaign['summary_plan_sha256_matches']:
                raise ValueError('GIRO summary does not match the current plan')
        for cell in giro_campaign['cells']:
            comparison_path = Path(cell.get('comparison_path', ''))
            mip_path = Path(cell.get('mip_path', ''))
            if not comparison_path.is_file():
                continue
            comparison_raw = comparison_path.read_bytes()
            comparison = json.loads(comparison_raw)
            marker_path = comparison_path.parent / 'COMPLETE.json'
            marker = json.loads(marker_path.read_bytes()) if marker_path.exists() else None
            comparison_sha = hashlib.sha256(comparison_raw).hexdigest()
            mip_sha = file_sha256(mip_path) if mip_path.is_file() else None
            fee = cell.get('destination_charge_start_fee')
            proof = comparison.get('proof_scope') or {}
            comparison_cell = comparison.get('cell') or {}
            expected_trip_ids = giro_instance_trip_ids(
                comparison_cell.get('instance'))
            target_terminal_kwh = comparison.get(
                'target_terminal_energy_kwh')
            compact = {
                'schema': comparison.get('schema'), 'cell': comparison.get('cell'),
                'target_terminal_energy_kwh': comparison.get('target_terminal_energy_kwh'),
                'terminal_constraint_semantics': comparison.get('terminal_constraint_semantics'),
                'original_giro': giro_original_metrics(comparison.get('original_giro'), fee),
                'fixed_duties_optimized': giro_selected_metrics(
                    comparison.get('fixed_duties_optimized'), fee,
                    expected_trip_ids, target_terminal_kwh),
                'joint_pool_optimized': giro_selected_metrics(
                    comparison.get('joint_pool_optimized'), fee,
                    expected_trip_ids, target_terminal_kwh),
                'joint_solver': comparison.get('joint_solver'),
                'saved_pool_records': comparison.get('saved_pool_records'),
                'pareto_augmented_pool_routes': comparison.get('pareto_augmented_pool_routes'),
                'all_pool_routes_replayed_before_optimization': comparison.get(
                    'all_pool_routes_replayed_before_optimization'),
                'pool_replay_s': comparison.get('pool_replay_s'),
                'proof_scope': proof, 'fee_provenance': comparison.get('fee_provenance'),
                'unchanged_conditions': comparison.get('unchanged_conditions'),
            }
            giro_campaign['comparisons'].append({
                'pair_id': cell.get('pair_id'), 'tariff': cell.get('tariff'),
                'instance_path': comparison_cell.get('instance'),
                'instance_sha256': comparison_cell.get('instance_sha256'),
                'tariff_path': comparison_cell.get('tariff_path'),
                'tariff_sha256': comparison_cell.get('tariff_sha256'),
                'destination_charge_start_fee': fee,
                'source_charge_start_fee': cell.get('source_charge_start_fee'),
                'path': str(comparison_path), 'sha256': comparison_sha,
                'completion_marker_path': str(marker_path),
                'completion_marker_matches': bool(
                    marker and marker.get('comparison_sha256') == comparison_sha),
                'mip_path': str(mip_path), 'mip_sha256': mip_sha,
                'mip_completion_marker_matches': bool(
                    marker and mip_sha and marker.get('mip_sha256') == mip_sha),
                'result': compact,
            })
    except Exception as exc:
        giro_campaign['summary_read_error'] = str(exc)
out['campaigns']['giro_zero_start_fee_20260913'] = giro_campaign

print(json.dumps(out))
