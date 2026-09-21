"""Recorded multi-terminal GIRO days; extraction only, no optimization."""
from pathlib import Path
import csv, json, collections, hashlib
from openpyxl import load_workbook

P=Path(__file__).resolve().parent;ROOT=P.parents[2]
MASTER=ROOT/'data/Par_VehicleDetails_Updated.csv'
RAW=ROOT/'outputs/meeting_20260910/giro_email_sources/Par_VehicleDetails.xlsx'
REF=P.parent/'cleanup_physics/inputs/Ref_dict.csv'
COORD=P.parent/'geography_map/coordinates.csv'
PLOT_COORD=P/'coordinates.csv'
STRICT=P.parent/'capacity_strict/strict_original/cases/w5_k15_18E2/mip/341291_r0'
STRICT_INPUT=ROOT/'outputs/independent_review_20260916/execution/p2_strict/inputs/w5_k15_18E2.csv'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readcsv(p):return [dict(r,source_line=i) for i,r in enumerate(csv.DictReader(p.open()),2)]
def mins(t):return sum(int(x)*y for x,y in zip(t.split(':'),(60,1)))
def clock(t):return f'{int(t)//60:02d}:{int(t)%60:02d}'
def num(t):return float(t) if t not in ['',None] else None
rows=readcsv(MASTER);ref={r['Location']:r['Ref'] for r in readcsv(REF)}
groups=collections.defaultdict(list)
for r in rows:groups[r['VehicleTask']].append(r)
area=lambda code:ref.get(code,code)
names={'13215':'Partille centrum','13410':'Jons väg','13801':'PARX depot','3127':'Heden','7581':'Gamlestads Torg','7880':'Östra Sjukhuset','13722':'13722 (name and position unresolved)'}
coordinates=readcsv(COORD)
coord_by_ref={area(r['code']):r for r in coordinates if r.get('latitude') and r.get('longitude')}
if PLOT_COORD.exists():
    for r in readcsv(PLOT_COORD):
        coord_by_ref[r['area']]={**r,'scope':r['coordinate_scope']}
workbook=load_workbook(RAW,read_only=True,data_only=True)['Data'];rawrows=list(workbook.values);rawheader=list(rawrows[0])

candidates=[]
for duty,rr in groups.items():
    services=[r for r in rr if r['Identifier']=='Regular'];charges=[r for r in rr if r['Identifier']=='Recharge']
    endpoints=sorted({r[x] for r in services for x in ['From1','To1']})
    locations=sorted({r[x] for r in rr for x in ['From1','To1']});areas=sorted({area(x) for x in locations})
    sites=sorted({r['From1'] for r in charges});opps=[x for x in sites if x!='PARX']
    candidates.append(dict(duty_id=duty,arm='recorded_GIRO',trips=len(services),service_endpoints=endpoints,
        service_endpoint_count=len(endpoints),locations=locations,raw_location_count=len(locations),reference_areas=areas,
        area_count_including_depot=len(areas),charging_sites=sites,opportunity_charging_sites=opps,
        charging_starts=len(charges),actual_charging_site_count=len(sites),opportunity_charging_site_count=len(opps),
        source=str(MASTER.resolve()),raw_source=str(RAW.resolve()),
        selection_eligible=len(opps)>=2 and len({area(x) for x in endpoints})>2,
        matching_fee0_fee5_arms_verified=False))
eligible=[c for c in candidates if c['selection_eligible']]
eligible.sort(key=lambda c:(0 if 4<=c['area_count_including_depot']<=5 else 1,c['area_count_including_depot'],c['trips']))
for i,c in enumerate(eligible,1):c['candidate_rank']=i
selected=['13309','13320'];assert eligible[0]['duty_id']=='13309'

def extract(duty):
    rr=groups[duty];events=[];original_rows=[];transfers=[]
    def emit(kind,start,end,left,right,**kwargs):
        e=dict(event_index=len(events),kind=kind,start_min=start,end_min=end,start_time=clock(start),end_time=clock(end),
            duration_min=end-start,from_code=left,to_code=right,from_ref=area(left),to_ref=area(right),
            source_trip_id=None,energy_consumed_kwh=None,energy_charged_kwh=0.,**kwargs)
        assert end>=start
        events.append(e);return e
    for r in rr:
        line=r['source_line'];raw=dict(zip(rawheader,rawrows[line-1]))
        for key in ['VehicleTask','Identifier','From1','To1','Start1','End1']:
            assert str(raw[key])==r[key],(duty,line,key,raw[key],r[key])
        for key in ['Usage kWh','Recharge kWh','SOC before','SOC after']:
            x,y=num(r[key]),raw[key]
            assert (x is None and y is None) or (x is not None and y is not None and abs(x-float(y))<1e-8)
        start,end=mins(r['Start1']),mins(r['End1'])
        if events:
            prev=events[-1];gap=start-prev['end_min']
            assert gap>=0 and prev['to_ref']==area(r['From1']),(duty,line,'missing inter-area movement')
            if gap or prev['to_code']!=r['From1']:
                kind='wait' if prev['to_code']==r['From1'] else 'same_reference_gap'
                e=emit(kind,prev['end_min'],start,prev['to_code'],r['From1'],
                    timing_basis='interval between adjacent recorded activities; within-area movement not separately specified' if kind=='same_reference_gap' else 'interval between adjacent recorded activities at same place',
                    source=str(RAW.resolve()),source_pointer=f'Data!rows {line-1}–{line}',
                    recorded_soc_before_percent=prev.get('recorded_soc_after_percent'),recorded_soc_after_percent=num(r['SOC before']))
                if kind=='same_reference_gap':transfers.append(e['event_index'])
        identifier=r['Identifier'];kind={'Regular':'service','Recharge':'charge','Pull-in':'deadhead','Pull-out':'deadhead','Deadhead':'deadhead','Prep-in':'prep','Prep-out':'prep'}[identifier]
        e=emit(kind,start,end,r['From1'],r['To1'],source_identifier=identifier,
            source=str(RAW.resolve()),source_pointer=f'Data!A{line}:X{line}',prepared_source=str(MASTER.resolve()),prepared_source_line=line,
            route_code=r['Route'],direction=r['Direction'],recorded_soc_before_percent=num(r['SOC before']),recorded_soc_after_percent=num(r['SOC after']),
            timing_basis='recorded GIRO clock interval')
        e['source_trip_id']=r['Ordered_Trip_ID'] or None
        e['energy_consumed_kwh']=num(r['Usage kWh']);e['energy_charged_kwh']=num(r['Recharge kWh']) or 0.
        if kind=='deadhead':e['movement_kind']=identifier.lower().replace('-','_')
        original_rows.append(r)
    for a,b in zip(events,events[1:]):assert a['end_min']==b['start_min'] and a['to_code']==b['from_code']
    visits=[];edges=[]
    def visit(refcode,t):
        v=dict(visit_index=len(visits),area=refcode,location=refcode,name=names.get(refcode,refcode),arrival_min=t,departure_min=t,event_indices=[],charge_indices=[],raw_codes=[])
        visits.append(v);return v
    current=visit(events[0]['from_ref'],events[0]['start_min'])
    for e in events:
        assert e['from_ref']==current['area'] and e['start_min']==current['departure_min']
        if e['from_code'] not in current['raw_codes']:current['raw_codes'].append(e['from_code'])
        if e['from_ref']!=e['to_ref']:
            before=current['visit_index'];current=visit(e['to_ref'],e['end_min']);current['raw_codes'].append(e['to_code'])
            edges.append(dict(from_visit=before,to_visit=current['visit_index'],event_index=e['event_index'],kind=e['kind'],
                from_ref=e['from_ref'],to_ref=e['to_ref'],from_code=e['from_code'],to_code=e['to_code'],source_trip_id=e['source_trip_id'],
                start_min=e['start_min'],end_min=e['end_min'],duration_min=e['duration_min'],energy_consumed_kwh=e['energy_consumed_kwh']))
        else:
            current['departure_min']=e['end_min'];current['event_indices'].append(e['event_index'])
            if e['kind']=='charge':current['charge_indices'].append(e['event_index'])
            if e['to_code'] not in current['raw_codes']:current['raw_codes'].append(e['to_code'])
    for v in visits:
        v.update(arrival_time=clock(v['arrival_min']),departure_time=clock(v['departure_min']),
                 charging_starts=len(v['charge_indices']),energy_charged_kwh=sum(events[i]['energy_charged_kwh'] for i in v['charge_indices']))
    services=[e for e in events if e['kind']=='service'];charges=[e for e in events if e['kind']=='charge']
    pulls=[e for e in events if e.get('movement_kind') in ['pull_in','pull_out']]
    trips=[e['source_trip_id'] for e in services];assert len(set(trips))==len(trips)
    charge_soc_residual=max(abs(e['energy_charged_kwh']-(e['recorded_soc_after_percent']-e['recorded_soc_before_percent'])*239.01/100) for e in charges)
    assert charge_soc_residual<.0001
    candidate=next(c for c in candidates if c['duty_id']==duty)
    return dict(schedule_id='original_'+duty,duty_id=duty,arm='recorded_GIRO',candidate=candidate,
        metrics=dict(trip_count=len(services),starts=len(charges),charging_sites=candidate['charging_sites'],
            reference_area_count=candidate['area_count_including_depot'],raw_location_count=candidate['raw_location_count'],
            depot_departure_min=pulls[0]['start_min'],depot_return_min=pulls[-1]['end_min'],
            depot_departure_time=pulls[0]['start_time'],depot_return_time=pulls[-1]['end_time'],
            charged_kwh=sum(e['energy_charged_kwh'] for e in charges),
            recorded_usage_kwh=sum(e['energy_consumed_kwh'] or 0 for e in events),
            min_recorded_soc_percent=min(v for e in events for v in [e.get('recorded_soc_before_percent'),e.get('recorded_soc_after_percent')] if v is not None)),
        source_trip_ids=trips,events=events,visits=visits,edges=edges,depot_movements=pulls,
        within_area_gaps_without_explicit_movement=transfers,original_source_rows=original_rows,
        validation=dict(raw_workbook_rows_match=True,all_recorded_activities_included=True,complete_continuous_clock=True,
            raw_locations_continuous_with_explicit_unspecified_same_reference_transfers=True,
            no_invented_inter_area_movements=True,source_trip_ids_unique=True,
            recharge_soc_consistent_with_239_01_kwh_capacity=True,maximum_charge_soc_residual_kwh=charge_soc_residual,
            new_solver_or_physics_validation=False))

schedules=[extract(d) for d in selected]
strict_result=json.loads((STRICT/'result.json').read_text());strict_routes=json.loads((STRICT/'selected_routes.json').read_text())
assert sha(STRICT_INPUT)==strict_result['provenance']['instance_sha256']
st=list(csv.DictReader(STRICT_INPUT.open()))
saved=[]
for i,r in enumerate(strict_routes):
    sites=sorted({s.rsplit('_',1)[0] for s in r['charging_stops']['stations']})
    if len(sites)<2:continue
    endpoints=sorted({st[int(t)][key] for t in r['trips'] for key in ['From1','To1']})
    refs=sorted({area(x) for x in endpoints+sites+['PARX']})
    saved.append(dict(source_index=i,trips=len(r['trips']),service_endpoints=endpoints,service_endpoint_count=len(endpoints),
        reference_areas=refs,area_count_including_depot=len(refs),charging_sites=sites,charging_starts=len(r['charging_stops']['stations']),
        source=str((STRICT/'selected_routes.json').resolve()),input_source=str(STRICT_INPUT.resolve()),
        assessment='Not selected: no comparable fee0/fee5 pair; saved fleet has duplicate service and fails documented capacity, uses constant240kW opportunity physics; not a corrected operational witness.'))
availability=dict(matched_k3='Original13401/13403/13405, all route21 shuttle: no complex spatial example.',
    matched_k5='Original13401/13403/13405/13408/13414, all route21 shuttle: no complex spatial example.',
    inspected_saved_18E2=saved,conclusion='No ready matched fee0/fee5 operationally validated comparison for chosen original13309/13320 established by this bounded search.',
    saved_fleet_duplicate_service=strict_result['duplicate_service_audit'],saved_fleet_capacity=strict_result['physical_station_capacity_audit'])
areas=[]
for code in sorted({area(e[k]) for s in schedules for e in s['events'] for k in ['from_code','to_code']}):
    cr=coord_by_ref.get(code)
    areas.append(dict(reference_code=code,name=names.get(code,code),raw_codes=sorted({e[k] for s in schedules for e in s['events'] for k in ['from_code','to_code'] if area(e[k])==code}),
        latitude=float(cr['latitude']) if cr else None,longitude=float(cr['longitude']) if cr else None,
        coordinate_source=cr['source'] if cr else None,coordinate_scope=cr['scope'] if cr else 'Coordinate not in prior geography table; plotting agent has separate verified proxies for13215/7581;13722 unresolved.',
        reference_mapping_source=str(REF.resolve())))
sources=[MASTER,RAW,REF,COORD,STRICT/'selected_routes.json',STRICT/'result.json',STRICT_INPUT,
    ROOT/'outputs/research_day_20260918/matched_k3/manifest.json',P.parent/'cleanup_physics/saved_joint_fee0.json',P.parent/'cleanup_physics/saved_joint_fee5.json',Path(__file__)]
if PLOT_COORD.exists():sources.append(PLOT_COORD)
notes=['Original GIRO recorded schedules only. No optimized fee0/fee5 counterpart is asserted.',
    'Diagram areas collapse only documented Ref_dict platform/reference memberships; exact rawplatform/charger codes remain in every event.',
    '13309 visits five reference areas including depot and charges at two opportunity sites plus depot.13320 visits six areas including unresolved13722.',
    'Same-reference gaps preserve unrecorded platform transfer/idle without inventing a travel time, path, or energy. Depot waiting need not consume idle energy; no new energy model is applied.',
    'Trip labels are prepared Ordered_Trip_ID values, not GIRO-supplied journey numbers. Rawsource row pointers retain actual GIRO duty and activities.',
    'Recorded SOC/energy/clock values are copied, not a new feasibility or optimality certificate. These133xx duties use239.01kWh documented18E2 capacity, unlike236.44kWh134xx18E1 figures.',
    'Straight diagram edges show ordered activities, not road geometry. Geographic proxies are not surveyed exactplatform/charger coordinates.']
result=dict(schema='complex_recorded_GIRO_days_v1',schedules=schedules,candidates=eligible[:10],areas=areas,saved_availability=availability,caveats=notes,sources={str(p.resolve()):sha(p) for p in sources})
def csvout(name,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (P/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in records)
(P/'schedules.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
csvout('candidates.csv',eligible[:10]);csvout('areas.csv',areas);csvout('saved_candidates.csv',saved)
for key in ['events','visits','edges']:
    csvout(key+'.csv',[dict(duty_id=s['duty_id'],**e) for s in schedules for e in s[key]])
csvout('charging.csv',[dict(duty_id=s['duty_id'],**e) for s in schedules for e in s['events'] if e['kind']=='charge'])
(P/'validation.json').write_text(json.dumps(dict(schedules=[dict(duty_id=s['duty_id'],metrics=s['metrics'],validation=s['validation']) for s in schedules],sources=result['sources'],caveats=notes),indent=2,ensure_ascii=False)+'\n')
print(json.dumps([dict(duty=s['duty_id'],metrics=s['metrics'],events=len(s['events']),visits=len(s['visits']),edges=len(s['edges'])) for s in schedules],indent=2))
