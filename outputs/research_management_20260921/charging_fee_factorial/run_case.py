"""Execute and physically validate one frozen fee-factorial cell."""
import argparse
import collections
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
B = P / 'bundle'
SRC = B / 'outputs/week_20260921/cleanup_physics'
sys.path.insert(0, str(SRC))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_charge_integral(profile, station, start, end, initial_soc, prices):
    """Integrate documented power bands and tariff boundaries, not model PWL helpers."""
    now, soc, cost = start, initial_soc, 0.0
    for _ in range(1000):
        if end-now < 1e-9:
            return soc, cost
        if station == 'PARX':
            rate, boundary = 60.0, profile.usable_capacity_kwh
        else:
            band = next(b for b in profile.opportunity_curve
                        if b.maximum_soc_fraction*profile.usable_capacity_kwh > soc+1e-8)
            rate = band.power_kw
            boundary = band.maximum_soc_fraction*profile.usable_capacity_kwh
        hour = math.floor(now+1e-9)
        hour //= 60
        dt = min(end-now, (boundary-soc)*60/rate, (hour+1)*60-now)
        assert dt > 0
        energy = rate*dt/60
        soc += energy
        cost += energy*prices[hour]
        now += dt
    raise AssertionError('Charging integral did not terminate')


def validate(result, j):
    pr = j.pr
    idle = pr.idle_kw / 60
    assert len(result['routes']) == 5
    covered = collections.Counter()
    site_events = {}
    route_checks = []
    sources = j.REPLAY[result['arm']]['routes']
    assignment = max(itertools.permutations(range(5)),
        key=lambda perm:sum(len(set(sources[i]['trips']) & set(j.ORIG[k]['trips'])) for i,k in enumerate(perm)))
    expected_duties = [j.ORIG[k]['duty_id'] for k in assignment]
    assert result['original_terminal_match'] == expected_duties
    boundary_tolerance_min = 1e-3  # big-M feasibility tolerance; preserve actual clocks/cost integral
    for index, (route, source) in enumerate(zip(result['routes'], sources)):
        assert route['source_index'] == index
        assert route['trips'] == source['trips']
        assert [a['from_trip'] for a in source['actions'][1:]] == source['trips']
        assert source['actions'][0]['next_trip'] == source['trips'][0]
        expected_target = j.f.TARGET[expected_duties[index]]
        assert abs(route['target_kwh']-expected_target) < 1e-8
        soc = pr.usable_capacity_kwh - source['actions'][0]['deadhead_kwh']
        minimum = soc
        consumed_charges = set()
        for action in source['actions'][1:]:
            soc -= j.TR[action['from_trip']]['energy']
            minimum = min(minimum, soc)
            if action['kind'] == 'direct':
                soc -= action['deadhead_kwh'] + action['idle_kwh']
            else:
                arrival, latest = action['arrival_min'], action['latest_departure_min']
                soc -= action['inbound_kwh']
                minimum = min(minimum, soc)
                matches = [(i, c) for i, c in enumerate(route['charges']) if c['station'] == action['station'] and c['start'] >= arrival - 1e-5 and c['end'] <= latest + 1e-5]
                assert len(matches) <= 1
                if matches:
                    i, c = matches[0]
                    assert i not in consumed_charges
                    consumed_charges.add(i)
                    soc -= (c['start'] - arrival) * idle
                    minimum = min(minimum, soc)
                    duration = c['end'] - c['start']
                    assert duration >= 3 - 1e-5
                    assert any(c['start'] >= h*60-boundary_tolerance_min and
                               c['end'] <= (h+1)*60+boundary_tolerance_min
                               for h in range(int(c['start']//60)-1,int(c['end']//60)+1))
                    after = j.charge_soc_after_minutes(pr, c['station'], soc, duration)
                    assert abs(after - soc - c['kwh']) < 1e-4
                    assert after <= pr.usable_capacity_kwh + 1e-5
                    integrated_soc, cost = independent_charge_integral(pr, c['station'], c['start'], c['end'], soc, j.f.prices)
                    assert abs(integrated_soc-after) < 1e-4
                    assert abs(cost - c['cost']) < 1e-4
                    soc = after - idle * (latest - c['end'])
                else:
                    soc -= idle * (latest - arrival)
                soc -= action['outbound_kwh']
            minimum = min(minimum, soc)
        assert len(consumed_charges) == len(route['charges'])
        assert minimum >= pr.reserve_kwh - 1e-5
        assert abs(soc - route['terminal_kwh']) < 1e-4
        assert soc >= expected_target - 1e-5
        assert route['starts'] == len(route['charges'])
        assert abs(route['electricity_cost'] - sum(c['cost'] for c in route['charges'])) < 1e-4
        covered.update(route['trips'])
        for c in route['charges']:
            site_events.setdefault(c['station'], []).extend([(c['start'], 1), (c['end'], -1)])
        route_checks.append(dict(minimum_soc_kwh=minimum, terminal_kwh=soc, target_kwh=route['target_kwh']))
    assert covered == collections.Counter({t: 1 for t in j.TR})
    peaks = {}
    for site, events in site_events.items():
        count = peak = 0
        for _, change in sorted(events, key=lambda v: (round(v[0], 5), v[1])):
            count += change
            peak = max(peak, count)
        assert site == 'PARX' or peak <= 1
        assert count == 0
        peaks[site] = peak
    cost = sum(r['electricity_cost'] for r in result['routes'])
    starts = sum(r['starts'] for r in result['routes'])
    assert abs(cost + result['fee'] * starts - result['objective']) < 1e-4
    return dict(valid=True, fleet=5, exactly_once_trips=len(covered), electricity=cost, starts=starts,
                charged_kwh=sum(c['kwh'] for r in result['routes'] for c in r['charges']),
                terminal_kwh=sum(r['terminal_kwh'] for r in result['routes']), station_peaks=peaks, routes=route_checks,
                independent_power_and_tariff_integral=True, single_hour_boundary_tolerance_min=boundary_tolerance_min,
                matched_original_duties=expected_duties)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('index', type=int)
    ap.add_argument('--seconds', type=float)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((P / 'manifest.json').read_text())
    for name, row in manifest['sources'].items():
        assert sha(B / name) == row.get('executed_sha256', row['sha256']), name
    assert 0 <= args.index < len(manifest['cases'])
    case = manifest['cases'][args.index]
    if args.seconds is not None:
        case = dict(case, seconds=args.seconds)
    attempt = f"{os.environ.get('SLURM_JOB_ID', 'local')}_r{os.environ.get('SLURM_RESTART_COUNT', '0')}_{time.time_ns()}"
    out = P / ('smoke' if args.smoke else 'results') / case['case_id'] / attempt
    out.mkdir(parents=True)
    started = time.time()
    import joint_sequence_charging as j
    tariff = B / f"tariffs/peak{case['peak']:02}.csv"
    j.f.prices = {int(r['time_block']): float(r['cost']) for r in csv.DictReader(tariff.open())}
    j.P = out
    original = j.f.baseline()
    assert all(not r['violations'] for r in original)
    receipt = dict(case=case, manifest_sha256=sha(P/'manifest.json'), started_epoch=started,
                   source_hashes={name: sha(B/name) for name in manifest['sources']},
                   source_execution_commit=json.loads((P/'code_receipt.json').read_text())['commit'] if (P/'code_receipt.json').exists() else 'uncommitted-smoke',
                   input_trip_sequence_sha256=hashlib.sha256(json.dumps(j.REPLAY[case['arm']]['routes'], sort_keys=True).encode()).hexdigest(),
                   tariff_sha256=sha(tariff), solver_version=j.gp.gurobi.version(), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                   original_repriced_electricity=sum(r['electricity_cost'] for r in original), original_charging_starts=sum(r['starts'] for r in original))
    pair_identity = dict(arm=case['arm'], tariff_sha256=sha(tariff),
        fixed_paths_sha256=receipt['input_trip_sequence_sha256'], physics=manifest['physics'],
        source_hashes=receipt['source_hashes'], terminal_targets=j.f.TARGET,
        seconds=case['seconds'],threads=case['threads'],seed=case['seed'])
    receipt['fee_pair_identity_sha256'] = hashlib.sha256(json.dumps(pair_identity,sort_keys=True).encode()).hexdigest()
    receipt['fee_pair_identity'] = pair_identity
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2))
    result = j.solve(case['arm'], case['fee'], case['seconds'], case['threads'], case['seed'])
    if result['solutions']:
        assert result['arm'] == case['arm'] and result['fee'] == case['fee']
        check = validate(result, j)
        if args.smoke:
            bad = copy.deepcopy(result)
            bad['routes'][0]['terminal_kwh'] += 2
            try:
                validate(bad, j)
            except AssertionError:
                check['corrupt_terminal_rejected'] = True
            else:
                raise AssertionError('Validator accepted corrupt terminal energy')
            bad = copy.deepcopy(result)
            bad['routes'][0]['charges'][0]['kwh'] += 2
            try:
                validate(bad, j)
            except AssertionError:
                check['corrupt_charge_rejected'] = True
            else:
                raise AssertionError('Validator accepted corrupt charging energy')
            bad = copy.deepcopy(result)
            bad['routes'][0]['target_kwh'] -= 2
            try:
                validate(bad, j)
            except AssertionError:
                check['corrupt_terminal_floor_rejected'] = True
            else:
                raise AssertionError('Validator accepted altered terminal target')
            bad = copy.deepcopy(result)
            bad['routes'][0]['charges'][0]['cost'] += 2
            try:
                validate(bad, j)
            except AssertionError:
                check['corrupt_cost_rejected'] = True
            else:
                raise AssertionError('Validator accepted corrupt tariff cost')
    else:
        check = {'valid': False, 'reason': 'no incumbent; not a physical infeasibility certificate'}
    (out/'validation.json').write_text(json.dumps(check, indent=2))
    receipt.update(ended_epoch=time.time(), physical_validation=check['valid'], status=result['status'],
                   result_sha256=sha(out/f"{case['arm']}.json"), validation_sha256=sha(out/'validation.json'))
    receipt['artifact_sha256'] = {q.name:sha(q) for q in out.iterdir() if q.is_file() and q.name != 'receipt.json'}
    (out/'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(dict(path=str(out), validation=check, objective=result['objective'], bound=result['bound'])))


if __name__ == '__main__':
    main()
