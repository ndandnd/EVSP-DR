"""Run on Unicorn; emit compact evidence and hash journals without decoding columns."""
from pathlib import Path
from collections import Counter, defaultdict
import csv, json, datetime, subprocess, hashlib, math

def collect_strict_capacity(root, kind):
    """Also support the usual SSH stdin invocation of this collector."""
    import importlib.util
    adapter = Path(__file__).resolve().with_name('strict_capacity_adapter.py')
    if not adapter.is_file():
        adapter = (Path.home() / 'ladder-lite/research-register/outputs'
                   / 'meeting_20260910/strict_capacity_adapter.py')
    spec = importlib.util.spec_from_file_location('strict_capacity_adapter', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.collect_strict_capacity(root, kind)


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def chain_extension_progress(root):
    """Operational attempt summaries; never promote checkpoints to results."""
    def latest_json(paths):
        for path in sorted(paths, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True):
            try:
                if path.stat().st_size > 100_000_000:
                    continue
                value = json.loads(path.read_bytes())
                if isinstance(value, dict):
                    return path, value
            except (OSError, ValueError):
                continue
        return None, None

    def scalars(value, keys):
        return {k: value[k] for k in keys if k in value
                and isinstance(value[k], (str, int, float, bool, type(None)))}

    result = {}
    for case in sorted((root/'cases').glob('w*_k*')):
        stages = {}
        for stage in ('cache', 'cg', 'mip'):
            path, value = latest_json((case/stage).glob('*/execution.json'))
            if path is not None:
                stages[stage] = {'path': str(path), 'authority': 'operational',
                    **scalars(value, ('case_id', 'mode', 'attempt', 'started_epoch',
                        'ended_epoch', 'execution_commit', 'manifest_sha256', 'input_sha256',
                        'status', 'returncode', 'watchdog_time_limit'))}
        entry = {'stages': stages}
        # Read only a bounded tail; tolerate a concurrent partial JSONL append.
        for path in sorted((case/'cache').glob('*/progress.jsonl'),
                           key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True):
            try:
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 65536))
                    lines = stream.read().splitlines()
                for line in reversed(lines):
                    try:
                        value = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if isinstance(value, dict):
                        entry['cache_progress'] = {'path': str(path), 'attempt': path.parent.name,
                            'authority': 'operational', **scalars(value, ('phase', 'event',
                                'elapsed_s', 'peak_rss_kib', 'pid', 'finished_sources',
                                'total_sources', 'source', 'stored_arcs', 'duration_s', 'error_type'))}
                        break
            except OSError:
                continue
            if 'cache_progress' in entry:
                break
        if not (case/'cg.json').exists():
            path, value = latest_json((case/'cg').glob('*/cg.json'))
            if path is not None:
                entry['cg_checkpoint'] = {'path': str(path), 'attempt': path.parent.name,
                    'authority': 'provisional', **scalars(value, ('csv', 'wall_s',
                        'certified_rc_optimal', 'stop_reason', 'iterations')),
                    'final': scalars(value.get('final') or {}, ('iter', 'artificials',
                        'obj', 'lp_obj', 'min_rc', 'pool_size', 'columns', 'routes',
                        'route_weight', 'route_weight_sum', 'fleet_weight', 'wall_s'))}
        if stages or len(entry) > 1:
            result[case.name] = entry
    return result


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
roots = {'compact_large_seed_20260914': home / 'compact_large_seed_20260914', 'lp_support_pool_diagnostic_20260914': home / 'lp_support_pool_diagnostic_20260914'}
out = {'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'campaigns': {}}
for name, root in roots.items():
    rows = []
    diagnostic = name in ('overnight_diagnostics_20260914', 'mip_repeatability_20260914', 'parallel_pool_followup_20260914', 'parallel_pool_unions_20260914', 'overnight_parallel_20260914', 'compact_seed_support_20260914', 'compact_large_seed_20260914', 'lp_support_pool_diagnostic_20260914', 'remaining_chain_gaps_20260914', 'retrospective_prefix_controls_20260914', 'decomposition_pool_union_20260914', 'decomposition_lp_support_union_20260914')
    diagnostic_manifest_sha = hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest() if diagnostic and (root/'manifest.json').exists() else None
    published_mips = set(root.glob('cases/*/mip_result.json')) if diagnostic else set()
    for p in sorted(set(root.rglob('*mip8h.json')) | set(root.rglob('*mip_budgeted.json')) | set(root.glob('*/mip.json')) | set((root/'results').glob('*60m.json')) | set((root/'mip').glob('*1h2stage.json')) | set(root.glob('p*/mip/*__1h2stage.json')) | set((root/'mip_attempts').glob('**/result.json')) | {p for p in root.glob('cases/*/mip/*/result.json') if 'smoke' not in p.parts} | published_mips):
        if name in ('chain_extension_20260913', 'chain_extension_20260914') and 'validation' in p.relative_to(root).parts:
            continue
        raw = p.read_bytes()
        d = json.loads(raw)
        row = {k: d.get(k) for k in ['buses','fleet_bound','fleet_proven','status','status_name','mip_gap','optimal_scope','pool_columns','source_cg_iterations','source_cg_wall_s','runtime_s','gurobi_optimize_wall_s','partitioning','overcovered_trips','charging_cost','continuous_realized_charging_cost','two_stage','physical_pool_audit','physical_replay_validated','physical_replay_scope','duplicate_trip_removal_validated','cross_route_charger_capacity_validated']}
        row.update(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), file_mtime_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime, datetime.timezone.utc).isoformat(), instance=d.get('instance'), selected_physical_statuses=dict(Counter(r.get('physical_realization',{}).get('status','missing') for r in d.get('selected_routes',[]))))
        if diagnostic:
            marker = json.loads((p.parent/'completion.json').read_bytes())
            if not (marker.get('usable') is True and marker.get('kind') == 'mip'
                    and marker.get('case_id') == p.parent.name
                    and marker.get('manifest_sha256') == diagnostic_manifest_sha
                    and Path(marker['result_path']).resolve() == p.resolve()
                    and d.get('source_result_sha256') == marker.get('source_status_sha256')
                    and d.get('source_journal_sha256') == marker.get('source_journal_sha256')
                    and marker['result_sha256'] == row['sha256']):
                raise ValueError(f'unverified diagnostic MIP publication: {p}')
            row.update(completion_marker_matches=True, authority='published',
                       resolved_source_path=str(p.resolve()),
                       publication_manifest_sha256=diagnostic_manifest_sha,
                       source_status_sha256=marker.get('source_status_sha256'),
                       source_journal_sha256=marker.get('source_journal_sha256'))
            for key in ('source_result_sha256', 'mip_provenance'):
                row[key] = d.get(key)
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
            try:
                raw = p.read_bytes()
                d=json.loads(raw)
            except Exception: continue
            row={k:v for k,v in d.items() if k not in ['routes','columns','selected_routes','iterations','history','iteration_log'] and not isinstance(v,list)}
            row['path']=str(p)
            if name in ('chain_extension_20260913', 'chain_extension_20260914') or diagnostic:
                for key in ('route_values', 'trip_duals', 'capacity_duals'):
                    row.pop(key, None)
                row['resolved_source_path'] = str(p.resolve())
                row['authority'] = 'published'
            if diagnostic:
                row['sha256'] = hashlib.sha256(raw).hexdigest()
                marker = json.loads((p.parent/'completion.json').read_bytes())
                if not (marker.get('usable') is True and marker.get('kind') == 'cg'
                        and marker.get('case_id') == p.parent.name
                        and marker.get('manifest_sha256') == diagnostic_manifest_sha
                        and Path(marker['result_path']).resolve() == p.resolve()
                        and marker['result_sha256'] == row['sha256']):
                    raise ValueError(f'unverified diagnostic CG publication: {p}')
                journal = Path(marker['journal_path'])
                journal_hasher = hashlib.sha256()
                with journal.open('rb') as stream:
                    for block in iter(lambda: stream.read(1048576), b''):
                        journal_hasher.update(block)
                if journal_hasher.hexdigest() != marker['journal_sha256']:
                    raise ValueError(f'diagnostic CG journal changed after publication: {p}')
                row['completion_marker_matches'] = True
                row['publication_manifest_sha256'] = diagnostic_manifest_sha
                row['columns_journal_sha256'] = marker['journal_sha256']
                if isinstance(row.get('final_lp'), dict):
                    row['final_lp'] = {k:v for k,v in row['final_lp'].items()
                                       if not isinstance(v, (list, dict))}
            cg.append(row)
    phases=[]
    phase_paths=set((root/'cg').glob('*.phase-telemetry.jsonl')) | set(root.glob('p*/cg/*.phase-telemetry.jsonl')) | set(root.glob('*/cg.phase-telemetry.jsonl')) | {p for p in root.glob('cases/*/*.phase-telemetry.jsonl') if 'smoke' not in p.parts}
    if name in ('chain_extension_20260913', 'chain_extension_20260914'):
        phase_paths.update(root.glob('cases/*/cg/*/*.phase-telemetry.jsonl'))
    if diagnostic:
        phase_paths.update(root.glob('cases/*/attempts/*/*.phase*.jsonl'))
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
    if name == 'graph_timeout_gates_v2_20260914' and (root/'manifest.json').exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location('graph_gate_metadata', root/'collect_adapter.py')
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        # Metadata only: graph construction is not a CG or integer-solve result.
        out['campaigns'][name]['workflow']['operational_graph_recovery'] = adapter.collect_graph_timeout_gates(root)
    if name in ('chain_extension_20260913', 'chain_extension_20260914'):
        out['campaigns'][name]['workflow']['attempt_progress'] = chain_extension_progress(root)
    if diagnostic:
        progress = []
        for p in sorted(root.glob('cases/*/attempts/*/state.json')):
            record = json.loads(p.read_bytes())
            progress.append({'case_id': p.parents[2].name, 'attempt': p.parent.name,
                             'path': str(p), 'state': record})
        out['campaigns'][name]['workflow']['attempt_progress'] = progress
        construction_kinds = {
            'parallel_pool_unions_20260914': 'finite_pool_union',
            'retrospective_prefix_controls_20260914': 'retrospective_iteration_prefix',
            'decomposition_pool_union_20260914': 'parent_mapped_partition_pool',
            'decomposition_lp_support_union_20260914': 'parent_mapped_partition_pool',
        }
        if name in construction_kinds:
            constructions = {}
            for marker_path in sorted(root.glob('cases/*/completion.json')):
                marker = json.loads(marker_path.read_bytes())
                if marker.get('kind') != 'pool_construction':
                    continue
                if not (marker.get('manifest_sha256') == diagnostic_manifest_sha
                        and marker.get('case_id') == marker_path.parent.name
                        and marker.get('usable') is True
                        and marker.get('optimization_run') is False):
                    raise ValueError(f'unverified pool construction: {marker_path}')
                artifact = Path(marker['result_path'])
                detail = Path(marker['construction_path'])
                if (file_sha256(artifact) != marker['result_sha256']
                        or file_sha256(detail) != marker['construction_sha256']):
                    raise ValueError(f'changed pool construction: {marker_path}')
                value = json.loads(artifact.read_bytes())
                if not (value.get('artifact_kind') == construction_kinds[name]
                        and value.get('certified_rc_optimal') is False):
                    raise ValueError(f'pool construction claims CG: {artifact}')
                audit = json.loads(detail.read_bytes())
                constructions[marker['case_id']] = {
                    **{key: marker.get(key) for key in ('kind', 'result_path',
                        'result_sha256', 'journal_path', 'journal_sha256',
                        'construction_path', 'construction_sha256', 'execution',
                        'manifest_sha256')},
                    'optimization_run': False, 'full_model_lp_certified': False,
                    'construction_summary': {key: audit.get(key) for key in
                        ('sources', 'source_order', 'union_columns', 'source',
                         'cutoff_found_iter_exclusive', 'historical_log_elapsed_s',
                         'unique_pool_columns', 'semantics', 'columns',
                         'component_audits', 'source_audits', 'parent_graph_constructed')},
                    'journal_hash_scope': 'Verified by construction worker; not rehashed by this collector.'}
            out['campaigns'][name]['workflow']['pool_constructions'] = constructions
        for record_name in ['validation.json', 'scheduler_verification.json', 'selection.json']:
            p = root/record_name
            if p.exists():
                out['campaigns'][name]['workflow'][record_name] = json.loads(p.read_bytes())
        metadata_path = root/'case_metadata.json'
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_bytes())
            if metadata.get('campaign_manifest_sha256') != diagnostic_manifest_sha:
                raise ValueError(f'case metadata differs from frozen campaign: {metadata_path}')
            manifest_cases = json.loads((root/'manifest.json').read_bytes())['cases']
            parent = metadata['parent']
            benchmark_path = Path(parent['benchmark_source_manifest'])
            if file_sha256(benchmark_path) != parent['benchmark_source_manifest_sha256']:
                raise ValueError(f'parent benchmark manifest changed: {benchmark_path}')
            benchmarks = [case for case in json.loads(benchmark_path.read_bytes())['cases'].values()
                          if case.get('kind') == 'decomposition_recombine_and_cross_group_cg'
                          and case.get('input_sha256') == parent['input_sha256']]
            if (not benchmarks or {case['target_duties'] for case in benchmarks} != {parent['target_duties']}
                    or {case['trip_count'] for case in benchmarks} != {parent['trip_count']}
                    or file_sha256(Path(parent['input_path'])) != parent['input_sha256']):
                raise ValueError(f'parent benchmark target or input differs: {metadata_path}')
            for cid, values in metadata['case_metadata'].items():
                case = manifest_cases[cid]
                if (values['input_sha256'] != case.get('parent_input_sha256', case.get('input_sha256'))
                        or values['kind'] != case['kind']
                        or values['input_sha256'] != parent['input_sha256']
                        or values['input_path'] != parent['input_path']
                        or values['target_duties'] != parent['target_duties']
                        or {values['csv']} != {item['csv'] for item in benchmarks}
                        or values.get('is_validation', False) != case.get('is_validation', False)):
                    raise ValueError(f'case metadata changes input or stage: {cid}')
            out['campaigns'][name]['workflow']['case_metadata.json'] = metadata
            out['campaigns'][name]['workflow']['case_metadata_verification'] = {
                'path': str(metadata_path), 'sha256': file_sha256(metadata_path),
                'manifest_sha256': diagnostic_manifest_sha,
                'benchmark_manifest_sha256': parent['benchmark_source_manifest_sha256'], 'verified': True}
        if diagnostic_manifest_sha:
            manifest_cases = json.loads((root/'manifest.json').read_bytes())['cases']
            validation_ids = {cid for cid, case in manifest_cases.items() if case.get('is_validation')}
            validation_mips = [row for row in rows if Path(row['path']).parent.name in validation_ids]
            if validation_mips:
                out['campaigns'][name]['workflow']['validation_mip_artifacts'] = validation_mips
                out['campaigns'][name]['mip'] = [row for row in rows if Path(row['path']).parent.name not in validation_ids]
    for record_name in ['ready_mips_manifest.json', 'ready_mips_submission.json', 'ready_mips_old_cancellations.json', 'obsolete_chain2_cancellations.json', 'retired_queue_entries.json', 'case_jobs.json', 'cache_jobs.json', 'cache_cli_validation.json', 'cache_compatibility.json', 'manifest.json', 'jobs.json', 'smoke_job.json', 'downstream/mip_concurrency_rebalance_20260911T0932Z.json', 'downstream/dependency_repair_20260911T0831Z.json', 'downstream/default_mip_migration_a01.json', 'downstream/default_mip_migration_a02.json', 'resource_override.json', 'workflow_submission.json', 'mip_submission.json', 'mip_retry2_submission.json', 'submission.json', 'submission.cg.json', 'submission.mip.json', 'retry_manifest.json', 'rerun_manifest.json', 'repair_submission.json', 'publication_recovery_772009.json', 'manifests/submission_initial.json', 'manifests/submission_final.json', 'execution_plan.json']:
        record_path = root / record_name
        if record_path.exists():
            out['campaigns'][name]['workflow'][record_name] = json.loads(record_path.read_bytes())
    for record_name in ['cg_jobs.tsv', 'freeze_mip_jobs.tsv']:
        record_path = root / record_name
        if record_path.exists():
            out['campaigns'][name]['workflow'][record_name] = record_path.read_text()

strict_name='reserve_feasibility_screen_20260914'
strict_root=home/strict_name
if (strict_root/'manifest.json').exists():
    out['campaigns'][strict_name]=collect_strict_capacity(strict_root,'pilot')
q=subprocess.run(['/usr/local/slurm/slurm-25.05.5/bin/squeue','--me','--json'],capture_output=True,text=True,check=True)
out['queue']=json.loads(q.stdout)
out['collected_until_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(out,allow_nan=False))
