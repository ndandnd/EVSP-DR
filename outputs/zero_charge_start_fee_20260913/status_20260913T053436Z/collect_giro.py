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

home=Path.home()/'ladder-lite'
out={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaigns':{},'collection_scope':'GIRO fee campaign only'}
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
