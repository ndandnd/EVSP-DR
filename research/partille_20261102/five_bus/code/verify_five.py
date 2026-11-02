"""Solver-free verification against the original activities, directed movements, nonlinear taper and tariff."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def plugs(records):
    grouped = defaultdict(list)
    for c in records:
        assert c['end'] >= c['start'] + 3 - 1e-5
        grouped[c['station']].append((c['start'], c['end']))
    peaks = {}
    overlap = 0.
    for station, intervals in grouped.items():
        if station == 'PARX':
            events = sorted([(a, 1) for a,_ in intervals]+[(b,-1) for _,b in intervals])
            peaks[station] = max(itertools.accumulate(x[1] for x in events))
        else:
            times = sorted(intervals)
            overlap = max([overlap]+[b - c for (a,b),(c,d) in itertools.pairwise(times)])
            assert overlap <= 1e-5, (station, overlap)
            peaks[station] = 1
    return dict(station_peaks=peaks, max_interval_overlap_minutes=overlap)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('root', type=Path)
    cli.add_argument('private', type=Path)
    cli.add_argument('release', type=Path)
    cli.add_argument('model_subdir', nargs='?', default='multihour')
    args = cli.parse_args()
    root, private, release = args.root.resolve(), args.private.resolve(), args.release.resolve()
    path = root/'outputs/giro_transdev_email_20261001/final_check/check_middle_feasibility.py'
    a = load_module('frozen_independent_checker', path)
    profile = load_module('frozen_profile', a.MODEL_PHYSICS).PARTILLE_PROFILES['18E1']
    assert profile.reserve_fraction == .15 and profile.usable_capacity_kwh == 236.44 and profile.idle_kw == .1
    assert profile.opportunity_setup_min == 0 and profile.minimum_recharge_duration_min == 3
    prices = {int(x['time_block']):float(x['cost']) for x in a.read_csv(a.TARIFF_PATH)}
    trips = {int(r['count_trip_id']): {'start': a.minutes(r['Start1']), 'end': a.minutes(r['End1']),
             'energy': float(r['Usage kWh']), 'source_id': int(r['Ordered_Trip_ID']), 'from': r['From1'], 'to': r['To1']}
             for r in a.read_csv(a.TRIPS_PATH)}
    # Use LITERAL endpoints for these two directional moves, never a site alias.
    rows = [dict(r, start_min=a.minutes(r['Start1']), end_min=a.minutes(r['End1'])) for r in a.read_csv(a.CSV_PATH)]
    raw_by_trip = {int(r['Ordered_Trip_ID']): r for r in rows if r['Identifier']=='Regular'}
    for t in trips.values():
        r = raw_by_trip[t['source_id']]
        assert (t['from'],t['to'],t['start'],t['end'],t['energy']) == (r['From1'],r['To1'],r['start_min'],r['end_min'],float(r['Usage kWh']))
    observations = defaultdict(set)
    for r in rows:
        if r['Identifier'] in {'Pull-in','Pull-out','Deadhead'}:
            observations[r['From1'],r['To1']].add((r['end_min']-r['start_min'],float(r['Usage kWh'])))
    independent_static_arc = a.reference_arc_loader()
    def arc(left, right):
        if left == right:
            return 0., 0.
        if (left,right) in [('2190','2190L'),('2190L','2190')]:
            choices = observations[left,right]
            assert len(choices)==1
            return next(iter(choices))
        assert 'PARX' in (left,right) and '2190L' not in (left,right)
        return independent_static_arc(left,right)
    original = a.read_json(a.ORIGINAL_PATH)['routes']
    baseline = a.read_json(private/'recorded/witness.json')
    clipped = a.read_json(private/'recorded/run.json').get('clock_capped_targets',False)
    targets = {}
    per_arm = {}
    hashes = {}
    def verify_coverage(covered):
        assert covered == Counter({t['source_id']:1 for t in trips.values()}) and len(covered)==62
    sums = defaultdict(float)
    covered, sessions, mins, terminals = Counter(), [], [], []
    raw_trace = []
    for ref, saved in zip(original, baseline, strict=True):
        duty = ref['duty_id']
        rr = [r for r in rows if r['VehicleTask']==duty]
        ids = [int(r['Ordered_Trip_ID']) for r in rr if r['Identifier']=='Regular']
        assert ids == ref['source_ordered_trip_ids'] and duty == saved['duty']
        covered.update(ids)
        soc = profile.usable_capacity_kwh
        prev = rr[0]['start_min']
        previous_place = 'PARX'
        trace = [soc]
        charge_checks = []
        for r in rr:
            assert previous_place == r['From1'] and r['start_min'] >= prev and r['end_min'] >= r['start_min']
            soc -= (r['start_min']-prev)*profile.idle_kw/60
            trace.append(soc)
            if r['Identifier'] == 'Recharge':
                q = float(r['Recharge kWh'])
                site = r['From1']
                requested_q = q
                active = a.required_charge_minutes(profile,site,soc,soc+q)
                if clipped:
                    active = min(active,r['end_min']-r['start_min'])
                assert active >= 3-1e-5 and active <= r['end_min']-r['start_min']+1e-5
                charged, cost, segments = a.independent_integral(profile, site, r['start_min'], r['start_min']+active, soc, prices)
                if clipped:
                    q = charged-soc
                    assert -1e-8 <= requested_q-q < 1e-4
                else:
                    a.close(charged-soc, q)
                trace.append(charged)
                charge_checks.append(dict(station=site, start=r['start_min'],end=r['end_min'],kwh=q,raw_capping_kwh=requested_q-q,active_min=active,cost=cost))
                soc = charged - max(0,r['end_min']-r['start_min']-active)*profile.idle_kw/60
            else:
                soc -= float(r['Usage kWh'] or 0)
            trace.append(soc)
            previous_place, prev = r['To1'], r['end_min']
        assert previous_place=='PARX' and max(trace) <= profile.usable_capacity_kwh+1e-5 and min(trace)>=profile.reserve_kwh-1e-5
        assert len(saved['charges'])==len(charge_checks)
        for source, computed in zip(saved['charges'], charge_checks, strict=True):
            for k in ('station','start','end'):
                assert source[k] == computed[k]
            a.close(source['kwh'], computed['kwh'])
            a.close(source['cost'], computed['cost'])
        a.close(soc,saved['terminal_kwh'])
        if not clipped:
            a.close(soc,a.baseline_target(profile,rr),1e-6)
        targets[duty] = soc
        mins.append(min(trace)); terminals.append(soc); sessions.extend(charge_checks)
        raw_trace.append(dict(duty=duty, soc_values=trace, terminal=soc, sessions=charge_checks))
    verify_coverage(covered)
    aggregates = lambda sessions,mins,terminals: dict(buses=5, exactly_once_source_trips=len(covered), starts=len(sessions),
        electricity_cost=sum(x['cost'] for x in sessions), energy_charged_kwh=sum(x['kwh'] for x in sessions),
        min_soc_kwh=min(mins), min_reserve_margin_kwh=min(mins)-profile.reserve_kwh, terminal_sum_kwh=sum(terminals),
        terminal_floor_sum_kwh=sum(targets.values()), **plugs(sessions))
    per_arm['recorded'] = aggregates(sessions,mins,terminals)
    per_arm['recorded']['minimum_terminal_margin_kwh'] = 0.
    per_arm['recorded']['raw_kwh_capping_sum'] = sum(c['raw_capping_kwh'] for c in sessions)
    per_arm['recorded']['raw_kwh_capping_max'] = max(c['raw_capping_kwh'] for c in sessions)
    write(private/'baseline_independent.json', raw_trace)
    hashes['recorded_witness'] = sha(private/'recorded/witness.json')
    hashes['recorded_independent_replay'] = sha(private/'baseline_independent.json')
    saved_routes = a.read_json(a.REPLAY_PATH)
    for arm in ('original','saved_joint_fee0'):
        d = private / args.model_subdir / arm
        result = a.read_json(d / (arm+'.json'))
        if not result['solutions']:
            per_arm[arm] = dict(restricted_solver_status=result['status'], incumbent=False)
            continue
        assert result['fee']==0 and len(result['routes'])==5
        sources = saved_routes[arm]['routes']
        perm = max(itertools.permutations(range(5)),key=lambda p:sum(len(set(r['trips']) & set(original[k]['trips'])) for r,k in zip(sources,p)))
        assert result['original_terminal_match'] == [original[p]['duty_id'] for p in perm]
        covered, sessions, mins, terminals = Counter(), [], [], []
        checks, margins = [], []
        for idx, (source, route) in enumerate(zip(sources, result['routes'], strict=True)):
            sequence = source['trips']
            assert route['trips']==sequence and route['source_index']==idx
            duty = original[perm[idx]]['duty_id']
            a.close(targets[duty],route['target_kwh'],1e-7)
            covered.update(trips[t]['source_id'] for t in sequence)
            dt,kwh = arc('PARX',trips[sequence[0]]['from'])
            actions = [dict(kind='source',next_trip=sequence[0],travel_min=dt, deadhead_kwh=kwh)]
            assert [x['from_trip'] for x in source['actions'][1:]]==sequence
            for old,next_t in zip(source['actions'][1:], [*sequence[1:],None], strict=True):
                prev_t = old['from_trip']
                t = trips[prev_t]
                assert next_t==old['next_trip']
                to = trips[next_t]['from'] if next_t is not None else 'PARX'
                if old['kind']=='direct':
                    dt,q = arc(t['to'],to)
                    wait = trips[next_t]['start']-t['end']-dt if next_t is not None else 0
                    assert wait>=0
                    actions.append(dict(kind='direct',from_trip=prev_t,next_trip=next_t,travel_min=dt,deadhead_kwh=q,
                                        waiting_min=wait,idle_kwh=wait*profile.idle_kw/60))
                else:
                    site=old['station']
                    ti,qi=arc(t['to'],site)
                    to_dt,qo=arc(site,to)
                    actions.append(dict(kind='charge',from_trip=prev_t,next_trip=next_t,station=site,
                        inbound_min=ti,inbound_kwh=qi,outbound_min=to_dt,outbound_kwh=qo,
                        arrival_min=t['end']+ti,latest_departure_min=trips[next_t]['start']-to_dt))
            replay = a.replay_route(profile, route, {'actions':actions}, trips, prices)
            assert not replay['chronology_failures'] and route['starts']==len(replay['charging_sessions'])
            checks.append(replay)
            for c in replay['charging_sessions']:
                sessions.append(dict(station=c['station'],start=c['start_min'],end=c['end_min'],cost=c['electricity_cost'],kwh=c['delivered_kwh']))
            mins.append(replay['minimum_soc_kwh']);terminals.append(replay['terminal_kwh'])
            margins.append(replay['terminal_kwh']-targets[duty])
        verify_coverage(covered)
        stat = aggregates(sessions,mins,terminals)
        a.close(stat['electricity_cost'],result['objective'],1e-4)
        stat.update(restricted_solver_status=result['status'], fixed_path_lower_bound=result['bound'],
            solved_gap_percent=100*(result['objective']-result['bound'])/abs(result['objective']),minimum_terminal_margin_kwh=min(margins))
        per_arm[arm] = stat
        write(d/'independent_replay.json',checks)
        for n in ('paths.json',arm+'.json','independent_replay.json',arm+'.lp',arm+'.gurobi.log'):
            hashes[f'{arm}/{n}'] = sha(d/n)
    fields = list(per_arm)
    out = dict(arms=per_arm,stage_order=fields,outputs_sha256=hashes, verifier_sha256=sha(Path(__file__)), historical_checker_sha256=sha(path),
               static_reference_input_sha256={p.name:sha(p) for p in [a.REFERENCE_DIR/'Ref_dict.csv', a.REFERENCE_DIR/'par_ref_dhd.csv']},
               check_tolerances=dict(soc_kwh=1e-5, time_overlap_min=1e-5, cost_units=1e-4))
    if all('electricity_cost' in v for v in per_arm.values()):
        r,o,s=[x['electricity_cost'] for x in per_arm.values()]
        out['reductions_percent'] = dict(recorded_to_original=100*(r-o)/r, original_to_saved=100*(o-s)/o,recorded_to_saved=100*(r-s)/r)
    write(release / ('audit_'+('clock_capped_' if clipped else '')+(args.model_subdir or 'singlehour')+'.json'),out)


if __name__ == '__main__':
    main()
