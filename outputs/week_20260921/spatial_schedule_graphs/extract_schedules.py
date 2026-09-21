"""Extract all fifteen source schedules without solving or changing source data.

Original clock times are recorded GIRO events; saved-model deadhead clocks use
an explicit feasible convention: travel immediately after service and leave a
charger at the next trip's latest departure. Charge times are actual witnesses.
"""
from pathlib import Path
import collections
import csv
import hashlib
import itertools
import json

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
C = P.parent / 'cleanup_physics'
G = P.parent / 'geography_map'
I = ROOT / 'outputs/meeting_20260917/route_explainer/inputs'
MASTER = ROOT / 'data/Par_VehicleDetails_Updated.csv'
RAW_GIRO = ROOT / 'outputs/meeting_20260910/giro_email_sources/Par_VehicleDetails.xlsx'
IDLE = .1 / 60
TOL = 1e-5


def read_csv(path):
    return [dict(r, source_line=n) for n, r in enumerate(csv.DictReader(path.open()), 2)]


def read_json(path):
    return json.loads(path.read_text())


def minutes(value):
    return sum(int(x) * factor for x, factor in zip(value.split(':'), (60, 1)))


def clock(value):
    seconds = round(value * 60)
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


master = read_csv(MASTER)
master_trips = {r['Ordered_Trip_ID']: r for r in master if r['Identifier'] == 'Regular'}
trips = {int(r['count_trip_id']): r for r in read_csv(I / 'k05.csv')}
source_to_internal = {r['Ordered_Trip_ID']: n for n, r in trips.items()}
original = read_json(I / 'original.json')['routes']
baseline = read_json(C / 'baseline.json')['routes']
replay = read_json(C / 'saved_sequence_replay.json')
arms = {arm: read_json(C / f'{arm}.json') for arm in ['saved_joint_fee0', 'saved_joint_fee5']}
geography = read_json(G / 'model_geography.json')
physical_validation = read_json(C / 'joint_validation.json')


def event(kind, start, end, left, right, consumption=0., charge=0., **extra):
    assert end >= start - TOL
    return dict(kind=kind, start_min=float(start), end_min=float(end),
                start_time=clock(start), end_time=clock(end),
                duration_min=float(end-start), from_code=left, to_code=right,
                source_trip_id=None, internal_trip_id=None,
                energy_consumed_kwh=float(consumption), energy_charged_kwh=float(charge), **extra)


def service(t):
    r = trips[t]
    source = master_trips[r['Ordered_Trip_ID']]
    e = event('service', minutes(r['Start1']), minutes(r['End1']), r['From1'], r['To1'],
              float(r['Usage kWh']), source=str((I/'k05.csv').resolve()), source_line=r['source_line'],
              original_source_duty=source['VehicleTask'], route_code=source['Route'],
              direction=source['Direction'], duration_basis='scheduled passenger service, not empty deadhead time')
    e.update(source_trip_id=r['Ordered_Trip_ID'], internal_trip_id=t)
    return e


def original_events(duty, physics):
    rows = [r for r in master if r['VehicleTask'] == duty]
    charges = {ch['line']: ch for ch in physics['charges']}
    events = []
    for r in rows:
        start, end = minutes(r['Start1']), minutes(r['End1'])
        if events:
            previous = events[-1]
            assert previous['to_code'] == r['From1'], (duty, previous, r)
            gap = start-previous['end_min']
            assert gap >= -TOL
            if gap > TOL:
                events.append(event('wait', previous['end_min'], start, r['From1'], r['From1'], gap*IDLE,
                    source=str(MASTER.resolve()), source_pointer=f'gap before CSV line {r["source_line"]}',
                    timing_basis='unoccupied interval between recorded GIRO events at the same location'))
        identifier = r['Identifier']
        extra = dict(source=str(MASTER.resolve()), source_line=r['source_line'], source_identifier=identifier,
                     timing_basis='recorded GIRO clock interval')
        if identifier == 'Regular':
            e = service(source_to_internal[r['Ordered_Trip_ID']])
            assert e['start_min'] == start and e['end_min'] == end
            e.update(extra)
        elif identifier == 'Recharge':
            ch = charges[r['source_line']]
            active = sum(s['active_min'] for s in ch['segments'])
            assert abs(active-(end-start)) < .0001  # no material hidden connected idle in this cohort
            e = event('charge', start, end, r['From1'], r['To1'],
                      max(0., end-start-active)*IDLE, ch['kwh'],
                      electricity_cost=ch['cost'], charging_active_min=active,
                      charging_basis='recorded recharge kWh; maximum documented taper reconstructs within-window power',
                      witness=str((C/'baseline.json').resolve()), **extra)
        elif identifier in {'Pull-out', 'Pull-in', 'Deadhead'}:
            e = event('deadhead', start, end, r['From1'], r['To1'], float(r['Usage kWh'] or 0),
                      movement_kind=identifier.lower().replace('-', '_'),
                      duration_basis='recorded directional GIRO movement; may use time-dependent band', **extra)
        elif identifier in {'Prep-out', 'Prep-in'}:
            e = event('prep', start, end, r['From1'], r['To1'], float(r['Usage kWh'] or 0), **extra)
        else:
            raise ValueError((duty, identifier))
        events.append(e)
    return events


def saved_events(arm, route):
    ri = route['source_index']
    actions = replay[arm]['routes'][ri]['actions']
    first = actions[0]
    firsttrip = trips[first['next_trip']]
    start = minutes(firsttrip['Start1'])
    events = [event('deadhead', start-first['travel_min'], start, 'PARX', firsttrip['From1'],
        first['deadhead_kwh'], movement_kind='pull_out', source=str((C/'saved_sequence_replay.json').resolve()),
        source_pointer=f'{arm}/routes/{ri}/actions/0', duration_basis='static model reference deadhead',
        timing_basis='latest depot departure that reaches first service on time')]
    used_charges = set()
    for ai, a in enumerate(actions[1:], 1):
        previous_trip = a['from_trip']
        events.append(service(previous_trip))
        end = minutes(trips[previous_trip]['End1'])
        left = trips[previous_trip]['To1']
        nexttrip = a['next_trip']
        right = 'PARX' if nexttrip is None else trips[nexttrip]['From1']
        extra = dict(source=str((C/'saved_sequence_replay.json').resolve()),
                     source_pointer=f'{arm}/routes/{ri}/actions/{ai}',
                     duration_basis='static model reference deadhead',
                     timing_basis='travel immediately after service; station departure at next-trip deadline')
        if a['kind'] == 'direct':
            events.append(event('deadhead', end, end+a['travel_min'], left, right, a['deadhead_kwh'],
                movement_kind='pull_in' if nexttrip is None else 'intertrip_direct', **extra))
            if nexttrip is not None:
                next_start = minutes(trips[nexttrip]['Start1'])
                gap = next_start-end-a['travel_min']
                assert gap >= -TOL and abs(gap*IDLE-a['idle_kwh']) < TOL
                if gap > TOL:
                    events.append(event('wait', end+a['travel_min'], next_start, right, right, gap*IDLE,
                        source=extra['source'], source_pointer=extra['source_pointer'],
                        timing_basis='feasible placement of direct-arc idle at destination; placement is not uniquely optimized'))
        elif a['kind'] == 'charge':
            station, arrival, latest = a['station'], a['arrival_min'], a['latest_departure_min']
            assert abs(arrival-end-a['inbound_min']) < TOL
            events.append(event('deadhead', end, arrival, left, station, a['inbound_kwh'],
                movement_kind='station_inbound', **extra))
            matching = [(n, ch) for n, ch in enumerate(route['charges']) if ch['station'] == station
                        and ch['start'] >= arrival-TOL and ch['end'] <= latest+TOL]
            assert len(matching) <= 1
            cursor = arrival
            if matching:
                ci, ch = matching[0]
                assert ci not in used_charges
                used_charges.add(ci)
                if ch['start']-cursor > TOL:
                    events.append(event('wait', cursor, ch['start'], station, station, (ch['start']-cursor)*IDLE,
                        source=str((C/f'{arm}.json').resolve()), source_pointer=f'routes/{ri}/charges/{ci}',
                        timing_basis='idle before optimized charge in fixed station visit'))
                events.append(event('charge', ch['start'], ch['end'], station, station, 0, ch['kwh'],
                    electricity_cost=ch['cost'], source=str((C/f'{arm}.json').resolve()),
                    source_pointer=f'routes/{ri}/charges/{ci}', timing_basis='actual optimized charge witness',
                    charging_basis='documented 18E1 taper, shared charger count and terminal floor'))
                cursor = ch['end']
            if latest-cursor > TOL:
                events.append(event('wait', cursor, latest, station, station, (latest-cursor)*IDLE,
                    source=extra['source'], source_pointer=extra['source_pointer'],
                    timing_basis='idle after optimized charge, or entire uncharged station visit'))
            events.append(event('deadhead', latest, latest+a['outbound_min'], station, right, a['outbound_kwh'],
                movement_kind='pull_in' if nexttrip is None else 'station_outbound', **extra))
        else:
            raise ValueError(a['kind'])
    assert used_charges == set(range(len(route['charges'])))
    return events


def index_visits(events):
    visits, edges = [], []
    def new_visit(location, when):
        visit = dict(visit_index=len(visits), location=location, arrival_min=when, departure_min=when,
                     event_indices=[], charge_indices=[], wait_indices=[], prep_indices=[])
        visits.append(visit)
        return visit
    current = new_visit(events[0]['from_code'], events[0]['start_min'])
    for i, e in enumerate(events):
        e['event_index'] = i
        assert e['from_code'] == current['location']
        assert abs(e['start_min']-current['departure_min']) < TOL, (i, e, current)
        moving = e['kind'] in {'service', 'deadhead'} and (e['from_code'] != e['to_code'] or e['duration_min'] > TOL)
        if moving:
            previous = current['visit_index']
            current['departure_min'] = e['start_min']
            current = new_visit(e['to_code'], e['end_min'])
            edges.append(dict(from_visit=previous, to_visit=current['visit_index'], event_index=i,
                              kind=e['kind'], source_trip_id=e['source_trip_id'], internal_trip_id=e['internal_trip_id'],
                              from_code=e['from_code'], to_code=e['to_code'],
                              start_min=e['start_min'], end_min=e['end_min'], duration_min=e['duration_min'],
                              energy_consumed_kwh=e['energy_consumed_kwh']))
            e['from_visit'], e['to_visit'] = previous, current['visit_index']
        else:
            current['event_indices'].append(i)
            if e['kind'] in {'charge', 'wait', 'prep'}:
                current[e['kind']+'_indices'].append(i)
            current['departure_min'] = e['end_min']
            e['from_visit'] = e['to_visit'] = current['visit_index']
    for visit in visits:
        visit.update(arrival_time=clock(visit['arrival_min']), departure_time=clock(visit['departure_min']),
                     dwell_min=visit['departure_min']-visit['arrival_min'],
                     energy_charged_kwh=sum(events[i]['energy_charged_kwh'] for i in visit['charge_indices']),
                     charging_starts=len(visit['charge_indices']))
    return visits, edges


pairing_validation = {}
for arm, solution in arms.items():
    rows = solution['routes']
    scores = [(sum(len(set(rows[j]['trips']) & set(original[k]['trips'])) for j, k in enumerate(perm)), perm)
              for perm in itertools.permutations(range(5))]
    best = max(score for score, _ in scores)
    optimal = [perm for score, perm in scores if score == best]
    existing = tuple(next(k for k, r in enumerate(original) if r['duty_id'] == d)
                     for d in solution['original_terminal_match'])
    assert existing in optimal
    pairing_validation[arm] = dict(total_trip_overlap=best, equally_optimal_assignments=len(optimal),
        preserved_assignment=solution['original_terminal_match'],
        overlap_matrix={r['duty_id']: [len(set(r['trips']) & set(s['trips'])) for s in rows] for r in original})

schedules, pairs = [], []
for o in original:
    duty = o['duty_id']
    pair = dict(pair_id='duty_'+duty, original_duty=duty,
                original_internal_trip_ids=o['trips'], original_source_trip_ids=[trips[t]['Ordered_Trip_ID'] for t in o['trips']])
    for short, arm in [('original', None), ('fee0', 'saved_joint_fee0'), ('fee5', 'saved_joint_fee5')]:
        if arm is None:
            route = next(r for r in baseline if r['duty'] == duty)
            sequence, ri = o['trips'], None
            events = original_events(duty, route)
        else:
            ri = arms[arm]['original_terminal_match'].index(duty)
            route = next(r for r in arms[arm]['routes'] if r['source_index'] == ri)
            sequence = route['trips']
            events = saved_events(arm, route)
            assert abs(route['target_kwh']-next(r for r in baseline if r['duty'] == duty)['terminal_kwh']) < TOL
            pair.update({short+'_source_index':ri, short+'_trip_overlap':len(set(sequence)&set(o['trips'])),
                         short+'_trip_count':len(sequence), short+'_same_trip_set':set(sequence)==set(o['trips']),
                         short+'_removed_source_trip_ids':[trips[t]['Ordered_Trip_ID'] for t in o['trips'] if t not in sequence],
                         short+'_added_source_trip_ids':[trips[t]['Ordered_Trip_ID'] for t in sequence if t not in o['trips']]})
        for previous, current in zip(events, events[1:]):
            assert abs(previous['end_min']-current['start_min']) < TOL
            assert previous['to_code'] == current['from_code']
        visits, edges = index_visits(events)
        trip_events = [e for e in events if e['kind']=='service']
        assert [e['internal_trip_id'] for e in trip_events] == sequence
        charges = [e for e in events if e['kind']=='charge']
        assert len(charges) == route['starts']
        consume = sum(e['energy_consumed_kwh'] for e in events)
        charged = sum(e['energy_charged_kwh'] for e in events)
        balance = 236.44 + charged - consume - route['terminal_kwh']
        assert abs(balance) < .0001, (duty, short, balance)
        soc = 236.44
        for e in events:
            e['soc_before_kwh'] = soc
            soc += e['energy_charged_kwh']-e['energy_consumed_kwh']
            e['soc_after_kwh'] = soc
            assert 236.44*.15-TOL <= soc <= 236.44+TOL, (duty,short,e['event_index'],soc)
        pullout = next(e for e in events if e.get('movement_kind')=='pull_out')
        pullin = next(e for e in reversed(events) if e.get('movement_kind')=='pull_in')
        metrics = dict(trip_count=len(sequence), starts=len(charges), charged_kwh=charged,
            electricity_cost=route['electricity_cost'], total_consumed_kwh=consume, terminal_kwh=route['terminal_kwh'],
            depot_departure_min=pullout['start_min'], depot_return_min=pullin['end_min'],
            depot_departure_time=clock(pullout['start_min']), depot_return_time=clock(pullin['end_min']),
            schedule_start_min=events[0]['start_min'], schedule_end_min=events[-1]['end_min'],
            service_minutes=sum(e['duration_min'] for e in trip_events),
            movement_minutes=sum(e['duration_min'] for e in events if e['kind']=='deadhead'),
            noncharging_wait_minutes=sum(e['duration_min'] for e in events if e['kind']=='wait'),
            charging_minutes=sum(e['duration_min'] for e in charges),
            prep_minutes=sum(e['duration_min'] for e in events if e['kind']=='prep'))
        schedules.append(dict(schedule_id=pair['pair_id']+'_'+short, pair_id=pair['pair_id'], arm=short,
            source_arm=arm or 'original_GIRO', original_duty=duty, source_index=ri,
            source_trip_ids=[trips[t]['Ordered_Trip_ID'] for t in sequence], internal_trip_ids=sequence,
            metrics=metrics, events=events, visits=visits, edges=edges,
            extraction_validation=dict(chronology_continuous=True, locations_continuous=True,
                source_trips_preserved=True, charging_starts_match=True, energy_balance_residual_kwh=balance)))
        pair[short+'_starts'] = len(charges)
    fee0 = next(s for s in schedules if s['pair_id']==pair['pair_id'] and s['arm']=='fee0')
    fee5 = next(s for s in schedules if s['pair_id']==pair['pair_id'] and s['arm']=='fee5')
    pair['fee0_fee5_trip_overlap'] = len(set(fee0['internal_trip_ids']) & set(fee5['internal_trip_ids']))
    pair['fee0_fee5_same_trip_set'] = set(fee0['internal_trip_ids']) == set(fee5['internal_trip_ids'])
    pairs.append(pair)

ranking = {
    '13414': (1, 'Only original-versus-fee5 pair with identical twelve service trips; nine to six charges, with original ET_R visits and a different thirteen-trip fee0 assignment.'),
    '13403': (2, 'Largest original-versus-fee5 charge-count change, twelve to five; both modeled arms also finish substantially earlier because trip assignments differ.'),
    '13405': (3, 'Twelve to nine to six charges, with substantially later final modeled service and depot return; highlights route reassignment and longer charging intervals.'),
    '13401': (4, 'Ten to eight to six charges; fee0 returns much later while fee5 keeps the original return time despite a different trip assignment.'),
    '13408': (5, 'Nine to eight to seven charges and low original-fee0 overlap; useful illustration of why bus labels should not imply identical service assignments.')}
for pair in pairs:
    pair['visual_priority'], pair['visual_interest_reason'] = ranking[pair['original_duty']]
pairs.sort(key=lambda p:p['visual_priority'])

for arm in ['original','fee0','fee5']:
    assert collections.Counter(t for s in schedules if s['arm']==arm for t in s['internal_trip_ids']) == collections.Counter({t:1 for t in trips})
assert abs(sum(s['metrics']['terminal_kwh'] for s in schedules if s['arm']=='original')-250.7183243523532) < TOL
capacity_peaks = {}
for arm in ['original','fee0','fee5']:
    site_events = collections.defaultdict(list)
    for s in schedules:
        if s['arm'] != arm:
            continue
        for e in s['events']:
            if e['kind']=='charge':
                site_events[e['from_code']].extend([(e['start_min'],1),(e['end_min'],-1)])
    capacity_peaks[arm] = {}
    for site, changes in site_events.items():
        current = peak = 0
        for _,delta in sorted(changes, key=lambda item:(round(item[0],5),item[1])):
            current += delta
            peak=max(peak,current)
        assert current==0 and peak<=1
        capacity_peaks[arm][site]=peak

def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (P/name).open('w') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)

def flat_rows(key):
    return [dict(schedule_id=s['schedule_id'], pair_id=s['pair_id'], arm=s['arm'], original_duty=s['original_duty'],
                 source_index=s['source_index'], **r) for s in schedules for r in s[key]]

sources=[MASTER,RAW_GIRO,I/'k05.csv',I/'original.json',C/'baseline.json',C/'saved_joint_fee0.json',C/'saved_joint_fee5.json',
         C/'saved_sequence_replay.json',C/'joint_validation.json',G/'model_geography.json',G/'coordinates.csv',Path(__file__)]
caveats=[
    'Trip numbers (source_trip_id) are stable Ordered_Trip_ID labels added in our prepared full input, not GIRO-supplied journey numbers. The raw GIRO workbook has no such field. count_trip_id is the separate local0–61 model index.',
    'All three fleets cover the same62 source trips exactly once; most paired individual buses do not share the same trip set.',
    'Pairing preserves the one-to-one maximum-total-trip-overlap assignment already used for terminal floors. It is an identification convention, not an operational bus identity.',
    'Fee0 and fee5 inherit different trip sequences and station paths; their differences are not a controlled fee-only causal effect.',
    'Recorded GIRO movements use source clocks and directional links. Saved-model movements use static symmetric reference deadheads; 2190→2190L is zero in the model but one minute in recorded GIRO.',
    'For modeled direct arcs, travel is placed immediately after service and idle at the destination. Within fixed station paths, the bus arrives immediately and departs at the next-trip deadline. These valid clock placements are reconstructed, not uniquely optimized.',
    'Repeated location visits should remain separate graph nodes. Straight edges connect events; they are not road routes or passenger stop geometry.',
    'ET_R has no documented coordinate. It may be an explicitly unlocated auxiliary diagram node, never a geocoded point. 2190 is terminal vicinity, not an established exact passenger platform.',
    'Baseline prep activities are included because recorded; modeled sequences have no invented prep activities. Waiting is idle, not an established crew break.',
    '236.44kWh initial/battery,15% reserve,0.1kW idle,documented18E1 opportunity taper,PARX60kW and minimum3minute active charging. Equal matched terminal floors; capacity counts hold within these five buses only.',
    'Full GIRO operational equivalence is not established: time-dependent/directional model deadheads, platform/departure blocking, FIFO, crew and other buses sharing chargers remain outside this validation.'
]
result=dict(schema='k5_all_bus_spatial_schedule_extraction_v1',pairs=pairs,schedules=schedules,
    pairing_validation=pairing_validation,locations=geography['locations'],coordinate_records=read_csv(G/'coordinates.csv'),
    fleet_validation=dict(fleets=3,buses_per_fleet=5,source_trips_per_fleet=62,exact_once=True,
        complete_chronology=True,continuous_locations=True,source_charge_intervals_preserved=True,
        event_endpoint_soc_within_battery_and_reserve=True,extracted_charge_capacity_peaks=capacity_peaks,
        maximum_energy_balance_residual_kwh=max(abs(s['extraction_validation']['energy_balance_residual_kwh']) for s in schedules),
        charge_starts={a:sum(s['metrics']['starts'] for s in schedules if s['arm']==a) for a in ['original','fee0','fee5']},
        independent_physical_validation=physical_validation),
    caveats=caveats,sources={str(p.resolve()):sha(p) for p in sources})
(P/'schedules.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
write_csv('pairings.csv',pairs)
write_csv('schedule_metrics.csv',[dict(schedule_id=s['schedule_id'],pair_id=s['pair_id'],arm=s['arm'],source_index=s['source_index'],**s['metrics']) for s in schedules])
for key in ['events','visits','edges']:write_csv(key+'.csv',flat_rows(key))
write_csv('locations.csv',result['locations'])
(P/'extraction_validation.json').write_text(json.dumps({k:result[k] for k in ['schema','pairing_validation','fleet_validation','sources','caveats']},indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(pairs=pairs,schedule_count=len(schedules),event_count=sum(len(s['events']) for s in schedules),
    visit_count=sum(len(s['visits']) for s in schedules),validation=result['fleet_validation']),indent=2))
