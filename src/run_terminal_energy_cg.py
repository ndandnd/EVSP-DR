"""Isolated zero-fee full-CG experiment; does not reuse old priced pools.

Every process starts afresh. Slurm retries must use distinct output directories.
Two-stage MIP proofs concern this finite pool, never branch-and-price.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import config
config.CHARGE_START_COST = 0.0  # Set before importing pricing/replay modules.

from audit_giro_known_columns import build_problem, HORIZON_MIN
from config import BUS_COST_KX, CHARGING_STATIONS
from expanded_path_realization import realize_expanded_path
from terminal_energy_pricer import TerminalEnergyNetwork
from utils_v2 import load_station_hourly_prices


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_energy(net, record):
    realized, detail = realize_expanded_path(
        net.problem, record, g_kwh=net.g, charge_kw=net.charge_kw,
        reserve_kwh=net.reserve, soc_step=net.soc_step,
        block_min=net.block_min, time_model='event')
    if realized is None:
        raise RuntimeError(f'Physical replay failed: {detail}')
    energy = detail['mapping']['expanded_grid_terminal_soc_kwh']
    if 'terminal_energy_kwh' in record and not math.isclose(
            energy, record['terminal_energy_kwh'], abs_tol=1e-6):
        raise RuntimeError('Pricing/master return-energy mismatch')
    record['terminal_energy_kwh'] = energy
    record['continuous_terminal_energy_kwh'] = detail['mapping']['continuous_terminal_soc_kwh']
    return record


def run(net, out, *, target, fleet_cap, cg_seconds, mip_seconds,
        batch_size=30, seeds=(), max_iters=100000, rc_eps=1e-6):
    import gurobipy as gp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    m = gp.Model('zero_fee_terminal_full_cg')
    m.Params.Threads = 1
    m.Params.Method = 1
    m.Params.LogFile = str(out / 'cg.gurobi.log')
    rows = {t: m.addConstr(gp.LinExpr() >= 1, name=f'trip_{t}') for t in net.problem.trips}
    energy_row = m.addConstr(gp.LinExpr() >= target, name='aggregate_return_energy_kwh')
    fleet_row = m.addConstr(gp.LinExpr() <= fleet_cap, name='fleet_cap')
    artificial = [m.addVar(obj=1, column=gp.Column([1], [row])) for row in rows.values()]
    artificial.append(m.addVar(obj=1 / net.g, column=gp.Column([1], [energy_row])))
    records, variables, seen = [], [], set()
    phase = 'artificial-elimination'
    history = []
    certified = False
    stop = 'iteration_limit'
    pricing_s = lp_s = 0.0

    def add(record):
        record = audit_energy(net, dict(record))
        if not record['trips'] or len(set(record['trips'])) != len(record['trips']):
            raise RuntimeError('Empty or repeated-trip route')
        key = (frozenset(record['trips']), record['terminal_energy_kwh'], record['cost'])
        if key in seen:
            return False
        seen.add(key)  # No trip-only dominance: preserve energy alternatives.
        coefficients = [1.0] * len(record['trips']) + [record['terminal_energy_kwh'], 1.0]
        constraints = [rows[t] for t in record['trips']] + [energy_row, fleet_row]
        variables.append(m.addVar(obj=0 if phase == 'artificial-elimination' else record['cost'],
                                  column=gp.Column(coefficients, constraints)))
        records.append(record)
        with (out / 'columns.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return True

    for record in seeds:
        add(record)
    if not seeds:
        # Only direct depot-trip-depot singletons; Phase I covers any missing
        # trips/energy while respecting the real-column fleet cap.
        for node, cost, trip, source_action in net._source_candidates():
            for sink_cost, energy, action in net.terminal_options.get(node, []):
                if action['kind'] == 'direct':
                    record = net._record([source_action, action])
                    record['terminal_energy_kwh'] = energy
                    add(record)
    for iteration in range(max_iters):
        remaining = cg_seconds - (time.monotonic() - started)
        if remaining <= 0:
            stop = 'cg_time_limit'
            break
        m.Params.TimeLimit = remaining
        tick = time.monotonic()
        m.optimize()
        lp_s += time.monotonic() - tick
        if m.Status != gp.GRB.OPTIMAL:
            stop = 'restricted_lp_not_optimal'
            break
        if phase == 'artificial-elimination' and m.ObjVal <= 1e-8:
            for v in artificial:
                v.UB = 0
                v.Obj = 0
            phase = 'combined-cost'
            for v, record in zip(variables, records):
                v.Obj = record['cost']
            continue
        tick = time.monotonic()
        batch = net.terminal_batch({t: r.Pi for t, r in rows.items()},
                                  terminal_dual=energy_row.Pi, route_dual=fleet_row.Pi,
                                  objective=phase, limit=batch_size)
        pricing_s += time.monotonic() - tick
        min_rc = batch[0]['rc'] if batch else None
        history.append(dict(iteration=iteration, phase=phase, elapsed_s=time.monotonic()-started,
                            rmp_objective=m.ObjVal, route_weight=sum(v.X for v in variables),
                            terminal_dual=energy_row.Pi, fleet_dual=fleet_row.Pi,
                            min_reduced_cost=min_rc, columns=len(records),
                            lp_seconds=lp_s, pricing_seconds=pricing_s))
        save(out / 'iterations.json', history)
        if min_rc is None or min_rc >= -rc_eps:
            certified = bool(batch) and phase == 'combined-cost'
            stop = ('pricing_certified' if certified else 'phase_one_infeasible'
                    if batch else 'no_route')
            break
        added = sum(add(item['record']) for item in batch if item['rc'] < -rc_eps)
        if not added:
            stop = 'negative_reduced_cost_duplicate_requires_audit'
            break
    summary = dict(cg_stop=stop, cg_pricing_certified=certified, cg_seconds=time.monotonic()-started,
                   terminal_target_kwh=target, fleet_cap=fleet_cap, columns=len(records),
                   last_iteration=history[-1] if history else None,
                   proof_scope='expanded_event_graph_weighted_LP_if_certified; MIP_finite_pool_only',
                   lp_seconds=lp_s, pricing_seconds=pricing_s,
                   columns_sha256=sha(out / 'columns.jsonl') if records else None)
    if certified:
        summary['weighted_full_graph_lp_lower_bound'] = (
            history[-1]['rmp_objective']
            + fleet_cap * min(0.0, history[-1]['min_reduced_cost']))
        summary['certificate_tolerance'] = rc_eps
    save(out / 'summary.json', summary)
    # Artificial columns are never available to the physical MIP.
    for v in artificial:
        v.UB = 0
    for v in variables:
        v.VType = gp.GRB.BINARY
        v.Obj = 1
    m.Params.Threads = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))
    m.Params.MIPGap = 0
    m.Params.TimeLimit = mip_seconds / 2
    m.Params.LogFile = str(out / 'fleet.gurobi.log')
    tick = time.monotonic()
    m.optimize()
    summary['fleet_stage'] = dict(status=m.Status, seconds=time.monotonic()-tick,
                                  bound=m.ObjBound if m.Status != gp.GRB.INFEASIBLE else None,
                                  buses=round(m.ObjVal) if m.SolCount else None)
    if m.SolCount:
        cap = round(m.ObjVal)
        starts = [v.X for v in variables]
        m.addConstr(gp.quicksum(variables) <= cap, name='stage_one_incumbent_fleet_cap')
        for v, record, start in zip(variables, records, starts):
            v.Obj = record['cost'] - BUS_COST_KX
            v.Start = start
        m.Params.TimeLimit = max(1, mip_seconds - (time.monotonic()-tick))
        m.Params.LogFile = str(out / 'charging.gurobi.log')
        m.optimize()
        selected = [r for r, v in zip(records, variables) if m.SolCount and v.X > .5]
        summary['charging_stage'] = dict(status=m.Status, cost=m.ObjVal if m.SolCount else None,
                                         bound=m.ObjBound, fleet=len(selected) if selected else None)
        if selected:
            summary['selected_terminal_kwh'] = sum(r['terminal_energy_kwh'] for r in selected)
            summary['selected_continuous_charging_cost'] = sum(r['continuous_realized_cost']-BUS_COST_KX for r in selected)
            save(out / 'selected_routes.json', selected)
    save(out / 'summary.json', summary)
    m.dispose()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--prices', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--cg-seconds', type=float, default=14400)
    p.add_argument('--mip-seconds', type=float, default=3600)
    p.add_argument('--target', type=float, default=280.7833253)
    p.add_argument('--fleet-cap', type=int, default=5)
    args = p.parse_args()
    from master_lp_gurobi import gurobi_preflight
    gurobi_preflight()
    data = Path(args.data_dir)
    problem = build_problem(data, args.csv, max_station_to_trip_wait_min=HORIZON_MIN)
    prices = load_station_hourly_prices(Path(args.prices), CHARGING_STATIONS)
    tick = time.monotonic()
    net = TerminalEnergyNetwork(problem, prices, soc_step=2.5, block_min=5,
                                g_kwh=240, charge_kw=350, reserve_kwh=0,
                                strict_tariff_coverage=True, arc_mode='lazy')
    graph_seconds = time.monotonic()-tick
    result = run(net, args.out, target=args.target, fleet_cap=args.fleet_cap,
                 cg_seconds=args.cg_seconds, mip_seconds=args.mip_seconds)
    result['graph_seconds'] = graph_seconds
    result['input_hashes'] = {str(path): sha(path) for path in
                              [data/args.csv, Path(args.prices), data/'Ref_dict.csv', data/'par_ref_dhd.csv']}
    result['physics'] = dict(battery_kwh=240, initial_kwh=240, charge_kw=350,
                            reserve_kwh=0, soc_step=2.5, time_step=5,
                            shared_capacity=False, charge_start_fee=0, coverage='cover')
    save(Path(args.out)/'summary.json', result)


if __name__ == '__main__':
    main()
