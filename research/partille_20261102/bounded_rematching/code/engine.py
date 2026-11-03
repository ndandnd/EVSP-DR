"""Frozen, one-order trip-connection recombination. No fleet MIP or GIRO duty assignments.

The manifest authenticates the historical owner summary, native provenance, source tree
and four physical inputs. Only the authenticated root prefix supplies trip links; its pricing certificate is recorded separately. All raw
trip identifiers, candidate schedules and certificates stay in the private output.
"""
from __future__ import annotations
import argparse, collections, csv, gzip, hashlib, itertools, json, math, os, pathlib, signal, subprocess, sys, time, uuid
PHYSICS = dict(battery_kwh=240, initial_kwh=240, charge_kw=240, station_kw={'PARX': 60}, reserve_kwh=0, soc_step=2.5, time_step=5, shared_capacity=True, charge_start_fee=5, coverage='cover', non_strict_tariff=True, bus_cost_kx=100000, max_trip2trip_min=57, max_station_to_trip_wait_min=1560)
PLUGS = {'2190L': 1, '4808': 1, '7880C': 1, 'JON_A': 1, '3127L': 2, 'PARX': None}

class CheckFailure(Exception):
    pass

def require(ok, message):
    if not ok:
        raise CheckFailure(message)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def dump(p, x):
    pathlib.Path(p).write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False) + '\n')

def minute(x):
    h, m = map(int, x.split(':'))
    return h * 60 + m

def pointer(obj, key):
    for part in key.strip('/').split('/'):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part.replace('~1', '/').replace('~0', '~')]
    return obj

def authenticate(m):
    """Authenticate frozen provenance and return summary, solver, data, physical inputs."""
    summary_path = pathlib.Path(m['summary'])
    require(sha(summary_path) == m['summary_sha256'], 'owner summary hash mismatch')
    s = json.loads(summary_path.read_text())
    require(s.get('cg_pricing_certified') is m['expected_cg_pricing_certified'], 'unexpected root pricing certificate status')
    require(s.get('seed_routes') == [], 'owner root is not unseeded')
    require(all((s['physics'].get(k) == v for k, v in PHYSICS.items())), 'owner physics/master mismatch')
    require(s['plugs'] == PLUGS and s['terminal_target_kwh'] == 0, 'plug/terminal mismatch')
    require(s['columns'] == m['root_columns'] and s['columns_sha256'] == m['root_prefix_sha256'], 'manifest root prefix differs from owner')
    require(m.get('native_evidence'), 'native provenance receipt required')
    for e in m['native_evidence']:
        require(sha(e['path']) == e['sha256'], 'native receipt hash mismatch')
        native = json.loads(pathlib.Path(e['path']).read_text())
        for k, v in e.get('assertions', {}).items():
            require(pointer(native, k) == v, 'native assertion mismatch: ' + k)
    solver = pathlib.Path(m['solver_root']).resolve()
    pin = subprocess.check_output(['git', '-C', str(solver), 'rev-parse', 'HEAD'], text=True).strip()
    require(pin == m['solver_commit'], 'solver commit mismatch')
    expected = m['solver_source_sha256']
    actual = {str(p.relative_to(solver)): sha(p) for p in sorted((solver / 'src').rglob('*.py'))}
    require(actual == expected, 'solver source manifest mismatch (including added/removed files)')
    data = pathlib.Path(m['data_root']).resolve()
    inputs = {k: pathlib.Path(v) for k, v in m['inputs'].items()}
    require(set(inputs) == {'instance', 'prices', 'locations', 'deadheads'}, 'exactly four non-duty physical inputs required')
    require(inputs['locations'].name == 'Ref_dict.csv' and inputs['deadheads'].name == 'par_ref_dhd.csv' and (inputs['prices'].name == 'hourly_prices_flat.csv'), 'unexpected physical input names')
    require(inputs['locations'].resolve() == data / 'Ref_dict.csv' and inputs['deadheads'].resolve() == data / 'par_ref_dhd.csv', 'reference inputs outside supplied data root')
    owner_roles = {'instance': s['inputs']['instance'], 'prices': s['inputs']['tariff'], 'locations': str(pathlib.PurePosixPath(s['inputs']['data_dir']) / 'Ref_dict.csv'), 'deadheads': str(pathlib.PurePosixPath(s['inputs']['data_dir']) / 'par_ref_dhd.csv')}
    for key, p in inputs.items():
        require(sha(p) == m['input_sha256'][key] == s['input_hashes'][owner_roles[key]], 'physical input hash mismatch: ' + key)
    require(len(s['input_hashes']) == 4, 'unexpected owner physical inputs')
    binding = m['native_binding']
    native = json.loads(pathlib.Path(m['native_evidence'][binding['evidence_index']]['path']).read_text())
    case = pointer(native, binding['case_pointer'])
    native_solver = pointer(native, binding['solver_pointer'])
    require(case['summary_sha256'] == m['summary_sha256'] and case['root_columns'] == m['root_columns'] and (case['root_actual_sha256'] == m['root_prefix_sha256']) and (case['root_authenticates'] is True) and (case['inputs_authenticate'] is True), 'native owner/prefix/input binding mismatch')
    require(native_solver['revision'] == m['solver_commit'] and native_solver['source_sha256'] == expected, 'native solver pin/source binding mismatch')
    return (s, solver, data, inputs)

def passenger_concurrency_floor(starts, ends):
    """Independent fleet floor from half-open [start, end) passenger intervals."""
    events = collections.Counter()
    for start, end in zip(starts, ends):
        events[start] += 1
        events[end] -= 1
    active = peak = 0
    for _, delta in sorted(events.items()):
        active += delta
        peak = max(peak, active)
    return peak


def connect(pool, count, prefix_sha, instance):
    """Extract passenger connections from exactly the authenticated root prefix."""
    rows = list(csv.DictReader(open(instance)))
    n = len(rows)
    starts = [minute(r['Start1']) for r in rows]
    ends = [minute(r['End1']) for r in rows]
    require(all((b > a for a, b in zip(starts, ends))), 'nonpositive passenger interval')
    arcs = collections.Counter()
    sequences = set()
    h = hashlib.sha256()
    seen = 0
    opener = gzip.open if str(pool).endswith('.gz') else open
    with opener(pool, 'rb') as f:
        for line in itertools.islice(f, count):
            h.update(line)
            seen += 1
            r = json.loads(line)
            seq = tuple(r['trips'])
            require(seq and len(seq) == len(set(seq)) and (set(seq) <= set(range(n))), 'invalid source trip sequence')
            require(r.get('physical_realization', {}).get('status') == 'valid_event_time_realized', 'source physical realization missing')
            mapping = r.get('continuous_realization', {})
            require(mapping.get('charge_kw') == 240 and mapping.get('station_kw') == {'PARX': 60}, 'source route power mismatch')
            sequences.add(seq)
            for u, v in zip(seq, seq[1:]):
                require(ends[u] <= starts[v], 'source passenger adjacency not a DAG')
                arcs[u, v] += 1
    require(seen == count and h.hexdigest() == prefix_sha, 'root prefix hash/count mismatch')
    return (rows, starts, ends, arcs, sequences)

def match(starts, arcs):
    """Maximum bipartite matching plus equal-size vertex-cover certificate."""
    n = len(starts)
    adj = {u: [] for u in range(n)}
    for u, v in arcs:
        adj[u].append(v)
    for u in adj:
        adj[u].sort(key=lambda v: (-arcs[u, v], v))
    right = {}

    def augment(u, seen):
        for v in adj[u]:
            if v in seen:
                continue
            seen.add(v)
            if v not in right or augment(right[v], seen):
                right[v] = u
                return True
        return False
    for u in sorted(range(n), key=lambda u: (starts[u], u)):
        augment(u, set())
    left = {u: v for v, u in right.items()}
    zl = set(range(n)) - set(left)
    zr = set()
    todo = list(zl)
    while todo:
        u = todo.pop()
        for v in adj[u]:
            if left.get(u) == v or v in zr:
                continue
            zr.add(v)
            if v in right and right[v] not in zl:
                zl.add(right[v])
                todo.append(right[v])
    cl = set(range(n)) - zl
    require(len(cl) + len(zr) == len(right) and all((u in cl or v in zr for u, v in arcs)), 'matching certificate failed')
    paths = []
    for start in sorted(set(range(n)) - set(right), key=lambda u: (starts[u], u)):
        p = [start]
        while p[-1] in left:
            p.append(left[p[-1]])
        paths.append(p)
    require(sorted((t for p in paths for t in p)) == list(range(n)), 'path cover mismatch')
    cert = {'edges': [[u, v, c] for (u, v), c in sorted(arcs.items())], 'matching': sorted(map(list, left.items())), 'min_vertex_cover_left': sorted(cl), 'min_vertex_cover_right': sorted(zr)}
    return (paths, cert)

def retime(paths, solver, data, inputs, out, result):
    """Use only the first deterministic ordering; stop at its first failed path."""
    sys.path.insert(0, str(solver / 'src'))
    sys.dont_write_bytecode = True
    from run_capacity_terminal_cg import CapacityTerminalNetwork, DEFAULT_PLUGS, audit_energy, problem_kwargs
    from audit_giro_known_columns import build_problem, HORIZON_MIN
    from config import CHARGING_STATIONS
    from capacity_event_pricer import record_occupancy
    from utils_v2 import load_station_hourly_prices
    import config, capacity_event_pricer, event_pricer_network, expanded_path_realization
    from dataclasses import replace
    require(DEFAULT_PLUGS == PLUGS, 'imported plug settings differ')
    require(all((getattr(m, 'CHARGE_START_COST', None) == 5 for m in [config, capacity_event_pricer, event_pricer_network, expanded_path_realization])), 'captured start fee differs')

    class FixedSequence(CapacityTerminalNetwork):

        def __init__(self, problem, prices, trips):
            self.following = dict(zip(trips, [*trips[1:], None]))
            self.first = trips[0]
            super().__init__(replace(problem, trips=tuple(trips)), prices, plugs=PLUGS, soc_step=2.5, block_min=5, g_kwh=240, charge_kw=240, station_kw={'PARX': 60}, reserve_kwh=0, strict_tariff_coverage=False)

        def _source_candidates(self):
            for edge in super()._source_candidates():
                if edge[2] == self.first:
                    yield edge

        def _direct_candidates(self, trip, soc_exit, depart):
            for edge in super()._direct_candidates(trip, soc_exit, depart):
                if edge[2] == self.following[trip]:
                    yield edge

        def _charge_candidates(self, trip, soc_exit, depart):
            successor = self.following[trip]
            outgoing, depot = (self.station_trip, self.station_depot)
            self.station_trip = {station: {successor: trips[successor]} for station, trips in outgoing.items() if successor in trips}
            self.station_depot = depot if successor is None else {}
            try:
                yield from super()._charge_candidates(trip, soc_exit, depart)
            finally:
                self.station_trip, self.station_depot = (outgoing, depot)
    problem = build_problem(data, str(inputs['instance'].resolve()), **problem_kwargs())
    prices = load_station_hourly_prices(inputs['prices'], CHARGING_STATIONS)
    occupancy = collections.Counter()
    selected = []
    for i, path in enumerate(paths):
        t = time.monotonic()
        entry = {'path_ordinal': i, 'trip_count': len(path)}
        result['retiming_steps'].append(entry)
        net = FixedSequence(problem, prices, path)
        entry['graph_seconds'] = time.monotonic() - t
        t = time.monotonic()
        penalty = {k: 10000.0 for k, c in occupancy.items() if PLUGS[k[0]] is not None and c >= PLUGS[k[0]]}
        candidates = net.capacity_terminal_batch({}, penalty, limit=1)
        entry['pricing_seconds'] = time.monotonic() - t
        require(candidates, 'fixed_sequence_unreachable')
        route = audit_energy(net, candidates[0]['record'])
        require(route['trips'] == path, 'retimer changed passenger sequence')
        occupancy.update(record_occupancy(route))
        selected.append(route)
        overload = sum((PLUGS[s] is not None and c > PLUGS[s] for (s, _), c in occupancy.items()))
        entry['overload_minutes'] = overload
        dump(out / 'routes.json', selected)
        require(overload == 0, 'first_order_capacity_conflict')
        del net
    return (problem, selected)

def independent_replay(problem, selected, paths, cert, inputs):
    """Recheck exact service, charging blocks, physical replay and shared plugs."""
    from audit_giro_known_columns import HORIZON_MIN
    from config import CHARGING_STATIONS
    from expanded_path_realization import normalize_event_station_prices, realize_expanded_path, realized_costs, validate_continuous_charging_blocks
    from run_exact_pool_mip import validate_injected_route
    from capacity_event_pricer import record_occupancy
    from station_power import StationPower
    from utils_v2 import load_station_hourly_prices
    n = len(problem.trips)
    edges = {(u, v) for u, v, _ in cert['edges']}
    matching = cert['matching']
    cl = set(cert['min_vertex_cover_left'])
    cr = set(cert['min_vertex_cover_right'])
    require(len({u for u, v in matching}) == len(matching) == len({v for u, v in matching}) and all((tuple(e) in edges for e in matching)), 'independent matching failed')
    require(len(cl) + len(cr) == len(matching) and all((u in cl or v in cr for u, v in edges)), 'independent vertex cover failed')
    require(all((problem.end_min[u] <= problem.start_min[v] for u, v in edges)), 'independent DAG failed')
    require([r['trips'] for r in selected] == paths, 'independent passenger path mismatch')
    prices = normalize_event_station_prices(load_station_hourly_prices(inputs['prices'], CHARGING_STATIONS), horizon_min=HORIZON_MIN, strict_tariff_coverage=False)
    power = StationPower(240, {'PARX': 60})
    occ = collections.Counter()
    coverage = collections.Counter()
    grid_terminal = actual_terminal = grid_cost = actual_cost = 0
    sessions = 0
    for route in selected:
        realized, detail = realize_expanded_path(problem, route, g_kwh=240, charge_kw=power, reserve_kwh=0, soc_step=2.5, block_min=5, time_model='event')
        require(realized is not None, 'independent energy replay failed')
        reason = validate_injected_route(problem, route, 240, power, 0, HORIZON_MIN, arrival_grace_min=0)
        require(reason is None, 'independent route replay: ' + str(reason))
        costs = realized_costs(realized, detail['mapping'], station_prices=prices)
        stored = validate_continuous_charging_blocks(route, route['continuous_realized_charging_blocks'], station_prices=prices, charge_kw=power)
        require(stored['block_schedule_sha256'] == costs['continuous_realized_charging_blocks_sha256'], 'independent charging block mismatch')
        require(math.isclose(route['cost'], costs['recomputed_expanded_grid_cost'], abs_tol=1e-06) and math.isclose(route['continuous_realized_cost'], costs['continuous_realized_cost'], abs_tol=1e-06), 'independent cost mismatch')
        require(math.isclose(route['terminal_energy_kwh'], detail['mapping']['expanded_grid_terminal_soc_kwh'], abs_tol=1e-06), 'independent terminal mismatch')
        coverage.update(route['trips'])
        occ.update(record_occupancy(route))
        grid_terminal += detail['mapping']['expanded_grid_terminal_soc_kwh']
        actual_terminal += detail['mapping']['continuous_terminal_soc_kwh']
        grid_cost += costs['recomputed_expanded_grid_cost'] - 100000
        actual_cost += costs['continuous_realized_cost'] - 100000
        sessions += len(route['charging_stops']['stations'])
    overload = sum((PLUGS[s] is not None and c > PLUGS[s] for (s, _), c in occ.items()))
    require(coverage == collections.Counter({i: 1 for i in problem.trips}) and overload == 0 and (grid_terminal >= -1e-06), 'independent coverage/capacity/terminal failure')
    return dict(validated=True, fleet=len(selected), trip_count=n, exact_once=True, grid_plug_overload_minutes=overload, grid_terminal_kwh=grid_terminal, continuous_terminal_kwh=actual_terminal, grid_nonfleet_cost=grid_cost, continuous_nonfleet_cost=actual_cost, charging_sessions=sessions)

def upstream_telemetry(summary):
    result = {k: summary.get(k) for k in ['graph_seconds', 'cg_seconds', 'lp_seconds', 'pricing_seconds']}
    allowed = {'fleet_stage': ['status', 'seconds', 'bound', 'buses'], 'charging_stage': ['status', 'seconds', 'cost', 'bound', 'fleet'], 'dive_settings': ['seconds', 'theta'], 'dive': ['state', 'steps', 'fixed', 'columns_after_dive', 'theta']}
    for section, keys in allowed.items():
        value = summary.get(section, {})
        result[section] = {key: value.get(key) for key in keys if key in value}
    return result

def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--manifest', type=pathlib.Path, required=True)
    a.add_argument('--output', type=pathlib.Path, required=True)
    a.add_argument('--retime-seconds', type=int, choices=[180], default=180, help='registered 180-second retiming-only cap')
    a.add_argument('--start-fee', type=float, choices=[5], required=True)
    z = a.parse_args()
    z.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    m = json.loads(z.manifest.read_text())
    phase = 'authentication'
    r = dict(schema='frozen-root-trip-link-run-v1', case=m['case'], attempt_id=str(uuid.uuid4()), validated=False, ordering_attempts=1, ordering='left/path start minute then canonical row id; successor frequency descending then canonical row id', row_order_caveat='Canonical row IDs resolve ties; consistent row permutations preserve matching cardinality but may alter matching and first-order retiming success; no permutations tested', retiming_limit_seconds=z.retime_seconds, phase_seconds={}, retiming_steps=[], physics=PHYSICS, plugs=PLUGS, manifest_sha256=sha(z.manifest), runner_sha256=sha(__file__), no_witness_inputs=True, scope='archived root CG only; cold-start performance and full-model charging optimality are untested')

    def timeout(*_):
        raise TimeoutError('retiming-only wall time cap')
    signal.signal(signal.SIGALRM, timeout)
    phase_start = start
    try:
        s, solver, data, inputs = authenticate(m)
        r['phase_seconds'][phase] = time.monotonic() - phase_start
        r['upstream_timings'] = upstream_telemetry(s)
        r['root_pricing_certified'] = s['cg_pricing_certified']
        r['upstream_timing_note'] = 'owner-native fields retained separately; root-only repair excludes dive columns and does not imply upstream cost is zero; components may overlap'
        r.update(source_root_columns=s['columns'], source_root_prefix_sha256=s['columns_sha256'], instance_sha256=m['input_sha256']['instance'], solver_commit=m['solver_commit'], solver_source_manifest_sha256=hashlib.sha256(json.dumps(m['solver_source_sha256'], sort_keys=True).encode()).hexdigest())
        phase = 'connections'
        phase_start = time.monotonic()
        rows, starts, ends, arcs, sequences = connect(m['pool'], s['columns'], s['columns_sha256'], inputs['instance'])
        r['phase_seconds'][phase] = time.monotonic() - phase_start
        phase = 'matching'
        phase_start = time.monotonic()
        paths, cert = match(starts, arcs)
        dump(z.output / 'paths.json', paths)
        dump(z.output / 'graph_certificate.json', cert)
        r.update(physical_trip_concurrency_floor=passenger_concurrency_floor(starts, ends), target_fleet=m.get('target_fleet', 8), trip_count=len(rows), adjacency_edges=len(arcs), unique_pool_sequences=len(sequences), matching_size=len(cert['matching']), minimum_paths=len(paths), new_sequences=sum((tuple(p) not in sequences for p in paths)))
        r['phase_seconds'][phase] = time.monotonic() - phase_start
        require(len(paths) <= m.get('target_fleet', 8), 'adjacency_target_not_reached')
        phase = 'retiming'
        phase_start = time.monotonic()
        signal.alarm(z.retime_seconds)
        try:
            problem, selected = retime(paths, solver, data, inputs, z.output, r)
        finally:
            signal.alarm(0)
        r['phase_seconds'][phase] = time.monotonic() - phase_start
        phase = 'independent_replay'
        phase_start = time.monotonic()
        r.update(independent_replay(problem, selected, paths, cert, inputs))
        r['target_hit'] = r['fleet'] == r['target_fleet'] and r['physical_trip_concurrency_floor'] == r['target_fleet']
        r['fleet_minimum_proved_by_service_overlap'] = r['fleet'] == r['physical_trip_concurrency_floor']
        r['subtarget_feasible'] = r['fleet'] < r['target_fleet']
        r['phase_seconds'][phase] = time.monotonic() - phase_start
        r['outcome'] = 'validated_candidate'
        r['failure_class'] = None
    except Exception as e:
        r['phase_seconds'][phase] = time.monotonic() - phase_start
        r.update(outcome='failed_closed', failure_phase=phase, failure_class='wall_timeout' if isinstance(e, TimeoutError) else str(e) if str(e) in {'fixed_sequence_unreachable', 'first_order_capacity_conflict', 'adjacency_target_not_reached'} else 'authentication_error' if phase == 'authentication' else 'invalid_input_or_validation_error', private_error_type=type(e).__name__)
        dump(z.output / 'private_error.json', {'type': type(e).__name__, 'message': str(e)})
    finally:
        signal.alarm(0)
    r['total_elapsed_seconds'] = time.monotonic() - start
    r['timing_scope'] = 'monotonic total includes authentication/imports/scan/matching/one retiming order/independent replay; excludes interpreter startup, manifest preparation and upstream owner timings'
    r['artifact_sha256'] = {p.name: sha(p) for p in z.output.iterdir() if p.is_file() and p.name not in {'receipt.json', 'SHA256SUMS', 'private_error.json'}}
    dump(z.output / 'receipt.json', r)
    (z.output / 'SHA256SUMS').write_text(''.join((sha(p) + '  ' + p.name + '\n' for p in sorted(z.output.iterdir()) if p.is_file() and p.name != 'SHA256SUMS')))
    print(json.dumps(r, sort_keys=True))
    return 0 if r['validated'] else 3
if __name__ == '__main__':
    raise SystemExit(main())
