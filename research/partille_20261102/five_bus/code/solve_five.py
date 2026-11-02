"""Reprice the recorded cohort and rerun the two archived fixed paths with literal connector arcs."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys


def read_rows(path):
    with path.open(newline='') as source:
        return list(csv.DictReader(source))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def clock_capped_baseline(j):
    """Honor both the recorded energy target and the physical power/time limit."""
    result = []
    for duty, rows in j.f.rows.items():
        p = j.f.profile_for_duty(duty)
        soc = p.usable_capacity_kwh
        charges = []
        prev = rows[0]['start']
        for r in rows:
            soc -= (r['start']-prev)*p.idle_kw/60
            if r['Identifier'] == 'Recharge':
                site, start, end = r['From1'], r['start'], r['end']
                q = float(r['Recharge kWh'])
                peak = min(soc+q, j.charge_soc_after_minutes(p,site,soc,end-start))
                active = j.f.h(p,site,peak)-j.f.h(p,site,soc)
                cost, _ = j.f.cost_profile(p,site,start,soc,peak)
                charges.append(dict(station=site,start=start,end=end,kwh=peak-soc,reported_kwh=q,
                                    active_minutes=active,cost=cost))
                soc = peak - max(0,end-start-active)*p.idle_kw/60
            else:
                soc -= float(r['Usage kWh'] or 0)
            prev = r['end']
        result.append(dict(duty=duty,charges=charges,electricity_cost=sum(c['cost'] for c in charges),
                           terminal_kwh=soc,starts=len(charges),violations=[]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('arm', choices=['recorded', 'original', 'saved_joint_fee0'])
    parser.add_argument('--across-hours', action='store_true')
    parser.add_argument('--clock-capped-targets', action='store_true')
    args = parser.parse_args()
    native = args.root.resolve() / 'outputs/research_management_20260921/charging_fee_factorial/native'
    bundle = native / 'bundle'
    sources = json.loads((native / 'code_receipt.json').read_text())
    for name, expected in sources['files'].items():
        assert digest(native / name) == expected, name
    out = args.out.resolve() / args.arm
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(bundle / 'outputs/week_20260921/cleanup_physics'))
    import joint_sequence_charging as joint
    raw = read_rows(bundle / 'data/Par_VehicleDetails_Updated.csv')
    def transfer_record(row):
        return (joint.f.minutes(row['End1'])-joint.f.minutes(row['Start1']), float(row['Usage kWh']))
    pair = ('2190', '2190L')
    transfers = {}
    evidence = {}
    for direction in [pair, pair[::-1]]:
        obs = Counter(transfer_record(r) for r in raw if (r['From1'], r['To1']) == direction)
        assert len(obs) == 1, (direction, obs)
        transfers[direction] = next(iter(obs))
        evidence['->'.join(direction)] = {'minutes': transfers[direction][0], 'kwh': transfers[direction][1],
                                          'raw_observations': sum(obs.values())}
    assert transfers[pair] == (1, .400000006) and transfers[pair[::-1]] == (0, 0)
    tariff = bundle / 'tariffs/peak08.csv'
    joint.f.prices = {int(r['time_block']): float(r['cost']) for r in read_rows(tariff)}
    baseline = joint.f.baseline()
    assert all(not r['violations'] for r in baseline)
    nominal_targets = dict(joint.f.TARGET)
    if args.clock_capped_targets:
        baseline = clock_capped_baseline(joint)
        joint.f.TARGET = {r['duty']: r['terminal_kwh'] for r in baseline}
    identity = dict(frozen_solver_commit=sources['commit'], wrapper_sha256=digest(Path(__file__)),
                    input_hashes={k: v for k, v in sources['files'].items() if k.startswith('bundle/')},
                    directed_connector=evidence, fixed_assignment=args.arm,
                    objective='electricity cost; zero start fee', sense='minimize',
                    tariff_sha256=digest(tariff), solver_version=list(joint.gp.gurobi.version()))
    identity.update(clock_capped_targets=args.clock_capped_targets, original_terminal_targets=nominal_targets,
                    applied_terminal_targets=joint.f.TARGET, continuous_across_hours=args.across_hours)
    if args.across_hours:
        identity['multi_hour_code_sha256'] = digest(Path(__file__).with_name('multihour.py'))
    if args.arm == 'recorded':
        write_json(out / 'witness.json', baseline)
        write_json(out / 'run.json', identity)
        return
    trips = {int(x['count_trip_id']): x for x in read_rows(bundle / 'outputs/meeting_20260917/route_explainer/inputs/k05.csv')}
    # Retain the historical solver's required input-provenance flag only. It
    # checks that the source route was recovered in the old model. Neither
    # saved SOC, charge clocks, incumbent nor bound is supplied to Gurobi.
    routes = joint.REPLAY[args.arm]['routes']
    traversals = Counter()
    for r in routes:
        slim = []
        for action in r['actions']:
            if action['kind'] != 'charge':
                slim.append(action)
                continue
            src = trips[action['from_trip']]
            dst = trips[action['next_trip']]
            site = action['station']
            inbound = src['To1'], site
            outbound = site, dst['From1']
            assert (inbound == pair and outbound == pair[::-1]) or (inbound == outbound == ('4808', '4808'))
            for direction in (inbound, outbound):
                traversals['->'.join(direction)] += 1
            dt1, e1 = transfers.get(inbound, (0, 0))
            dt2, e2 = transfers.get(outbound, (0, 0))
            arrival = joint.f.minutes(src['End1']) + dt1
            latest = joint.f.minutes(dst['Start1']) - dt2
            assert latest >= arrival
            slim.append(dict(kind='charge', station=site, from_trip=action['from_trip'], next_trip=action['next_trip'],
                             arrival_min=arrival, latest_departure_min=latest, inbound_min=dt1, inbound_kwh=e1,
                             outbound_min=dt2, outbound_kwh=e2))
        r['actions'] = slim
    joint.P = out
    write_json(out / 'paths.json', [{'trips': r['trips'], 'actions': r['actions']} for r in routes])
    identity.update(connector_traversals=traversals, initialization='cold solve; fixed trip orders and station choices only',
                    threads=2, time_limit_seconds=600, seed=0, mip_gap_tolerance=.0001, dependencies=[],
                    scheduler='local', master='fixed-path charging MILP; no route master or pricing step')
    write_json(out / 'run.json', identity)
    if args.across_hours:
        import multihour
        multihour.solve(joint, args.arm, 0, 600, 2, 0)
    else:
        joint.solve(args.arm, 0, 600, 2, 0)


if __name__ == '__main__':
    main()
