"""Replay a matched v15 schedule from the source data; emit a fail-closed publication receipt."""
import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from run_capacity_terminal_cg import DEFAULT_PLUGS, problem_kwargs
from audit_giro_known_columns import HORIZON_MIN, build_problem
from config import BUS_COST_KX, CHARGING_STATIONS
from expanded_path_realization import (normalize_event_station_prices, realize_expanded_path, realized_costs,
                                       validate_continuous_charging_blocks)
from run_exact_pool_mip import validate_injected_route
from run_fixed_duty_terminal import giro_sequences
from run_terminal_energy_cg import sha
from station_power import StationPower
from utils_v2 import base_station_name, load_station_hourly_prices

REQUIRED_PHYSICS = dict(battery_kwh=240, initial_kwh=240, charge_kw=240, station_kw={'PARX': 60},
                        reserve_kwh=0, soc_step=2.5, time_step=5, shared_capacity=True, charge_start_fee=5,
                        coverage='cover', non_strict_tariff=True, bus_cost_kx=100000,
                        max_trip2trip_min=57, max_station_to_trip_wait_min=1560)
COMMON_HASHES = {'hourly_prices_flat.csv': '1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200',
                 'Ref_dict.csv': '7bda0e1f439dc8bf5081499566eb2c6a0314190ef27294707f1403fd2c13e3a0',
                 'par_ref_dhd.csv': '5993e922c671f053611635578b32a1be13bab87b3b5fd8c02b699b81fe0eb66c'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b):
    return math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-12)


def audit_treatment_pool(summary_path, summary, selected, receipt):
    """Verify the exact LP prefix and that selected routes belong to the authenticated blind pool."""
    ip = summary['integer_pricing']
    receipt['integer_pricing'] = {k: ip.get(k) for k in
        ('seconds','budget_seconds','pricing_calls','columns_new','neighbourhoods','rep_solves','error')}
    # The owner writes each unmodified record with json.dumps followed by a newline.
    selected_lines = {(json.dumps(r) + chr(10)).encode(): None for r in selected}
    require(len(selected_lines) == len(selected), 'duplicate selected record')
    root = hashlib.sha256()
    full = hashlib.sha256()
    count = 0
    with (summary_path.parent / 'columns.jsonl').open('rb') as stream:
        for count, line in enumerate(stream, start=1):
            full.update(line)
            if count <= summary['columns']:
                root.update(line)
            if line in selected_lines:
                selected_lines[line] = count
    require(root.hexdigest() == summary['columns_sha256'], 'root LP prefix hash differs')
    require(full.hexdigest() == ip['columns_sha256_after'] and count == ip['columns_after'],
            'post-pricing pool hash or size differs')
    require(all(i is not None for i in selected_lines.values()), 'selected columns absent from final blind pool')
    receipt.update(root_columns=summary['columns'], root_sha256=root.hexdigest(), final_columns=count,
                   final_pool_sha256=full.hexdigest(),
                   selected_new_columns=sum(i > summary['columns'] for i in selected_lines.values()),
                   pricing_executed=not ip.get('error') and ip.get('pricing_calls', 0) > 0)


def validate(summary_path, selected_path, arm, receipt):
    summary, selected = (json.loads(path.read_text()) for path in (summary_path, selected_path))
    require(selected, 'selection is empty')
    receipt.update(selected_sha256=sha(selected_path), summary_sha256=sha(summary_path))
    for key, value in REQUIRED_PHYSICS.items():
        require(summary['physics'].get(key) == value, f'wrong physics for {key}')
    require(summary['plugs'] == DEFAULT_PLUGS and summary['terminal_target_kwh'] == 0, 'wrong plugs or target')
    if arm == 'blind':
        require(summary['seed_routes'] == [] and summary['integer_pricing_settings']['seconds'] == 5400 and
                summary['dive_settings']['seconds'] == 0, 'unexpected treatment')
    info = summary['inputs']
    data = Path(info['data_dir'])
    tariff, instance = Path(info['tariff']), Path(info['instance'])
    for path in (tariff, instance, data / 'Ref_dict.csv', data / 'par_ref_dhd.csv'):
        digest = sha(path)
        require(summary['input_hashes'][str(path)] == digest and
                COMMON_HASHES.get(path.name, digest) == digest, f'input hash mismatch for {path.name}')
    receipt.update(instance_sha256=sha(instance), inputs_sha256=summary['input_hashes'])
    problem = build_problem(data, str(instance), **problem_kwargs())
    prices = normalize_event_station_prices(load_station_hourly_prices(tariff, CHARGING_STATIONS),
                                             horizon_min=HORIZON_MIN, strict_tariff_coverage=False)
    power = StationPower(240, {'PARX': 60})
    occupancy, coverage = Counter(), Counter()
    grid_cost = actual_cost = terminal = 0
    for route in selected:
        realized, detail = realize_expanded_path(problem, route, g_kwh=240, charge_kw=power, reserve_kwh=0,
                                                 soc_step=2.5, block_min=5, time_model='event')
        require(realized is not None, f"route replay failed: {detail.get('reason')}")
        # Validate the saved schedule itself as well as the independently regenerated realization.
        reason = validate_injected_route(problem, route, 240, power, 0, HORIZON_MIN, arrival_grace_min=0)
        require(reason is None, f'continuous replay failed: {reason}')
        recomputed = realized_costs(realized, detail['mapping'], station_prices=prices)
        stored = validate_continuous_charging_blocks(route, route['continuous_realized_charging_blocks'],
                                                    station_prices=prices, charge_kw=power)
        require(stored['block_schedule_sha256'] == recomputed['continuous_realized_charging_blocks_sha256'],
                'saved charge blocks differ from replay')
        require(close(route['cost'], recomputed['recomputed_expanded_grid_cost'])
                and close(route['continuous_realized_cost'], recomputed['continuous_realized_cost']),
                'stored route cost differs from independently repriced schedule')
        end = detail['mapping']['expanded_grid_terminal_soc_kwh']
        require(close(end, route['terminal_energy_kwh']), 'terminal energy mismatch')
        terminal += end
        grid_cost += recomputed['recomputed_expanded_grid_cost'] - BUS_COST_KX
        actual_cost += recomputed['continuous_realized_cost'] - BUS_COST_KX
        coverage.update(route['trips'])
        require(len(set(route['trips'])) == len(route['trips']), 'repeated trip inside a route')
        stops = route['expanded_grid_charging_stops']
        for s, a, b in zip(stops['stations'], stops['cst'], stops['cet']):
            station = base_station_name(s)
            require(station in DEFAULT_PLUGS, 'unexpected station')
            for minute in range(math.floor(a+1e-9), math.ceil(b-1e-9)):
                occupancy[station, minute] += 1
    violations = sum(DEFAULT_PLUGS[s] is not None and n > DEFAULT_PLUGS[s] for (s, _), n in occupancy.items())
    exact_once = coverage == Counter({t: 1 for t in problem.trips})
    cover_valid = set(coverage) == set(problem.trips)
    demand = Counter()
    for t in problem.trips:
        demand[problem.start_min[t]] += 1
        demand[problem.end_min[t]] -= 1
    peak = active = 0
    for _, delta in sorted(demand.items()):
        active += delta
        peak = max(peak, active)
    receipt.update(fleet=len(selected), trip_count=len(problem.trips), exact_once=exact_once,
                   cover_valid=cover_valid, duplicate_service=sum(max(0,n-1) for n in coverage.values()),
                   peak_simultaneous_trips=peak, shared_fleet_cap=summary['fleet_cap'],
                   grid_plug_overload_minutes=violations, objective_nonfleet=grid_cost,
                   continuous_nonfleet=actual_cost, grid_terminal_kwh=terminal,
                   physics=summary['physics'], plugs=DEFAULT_PLUGS)
    require(cover_valid and not violations and terminal >= -1e-6 and len(selected) <= summary['fleet_cap'],
            'failed fleet, cover, terminal-energy or shared-minute-occupancy constraints')
    if arm == 'fixed_giro_diagnostic':
        duties = giro_sequences(data, str(instance), problem)
        require(Counter(tuple(r['trips']) for r in selected) == Counter(tuple(d['trips']) for d in duties),
                'selected routes differ from independently reconstructed historical duty sequences')
        receipt['duty_source_sha256'] = sha(data / 'Par_VehicleDetails_Updated.csv')
    if arm in ('blind','baseline'):
        require(summary['charging_stage']['fleet'] == len(selected)
                and close(summary['charging_stage']['cost'], grid_cost), 'selection differs from pool MIP')
    if arm == 'blind':
        audit_treatment_pool(summary_path, summary, selected, receipt)
    require(exact_once, 'cover is feasible but has duplicate service')
    receipt['validated'] = True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--selected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--arm', required=True, choices=['fixed_giro_diagnostic','blind','augmented','baseline'])
    p.add_argument('--start-fee', type=float, choices=[5], required=True)
    a=p.parse_args()
    receipt = dict(validated=False, arm=a.arm, schema='partille-selection-replay-v1')
    try:
        validate(a.summary, a.selected, a.arm, receipt)
    except Exception as exc:
        receipt['reason'] = str(exc)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    os.replace(tmp,a.out)
    print(json.dumps(receipt, sort_keys=True))
    raise SystemExit(0 if receipt['validated'] else 3)


if __name__ == '__main__':
    main()
