"""Replay one failed witness after a deterministic 6 ms charging extension.

Does not solve, alter a model, change validation tolerances, or modify raw artifacts.
"""
from pathlib import Path
import copy,csv,hashlib,json,time
import run_case as rc
import joint_sequence_charging as j
P=Path(__file__).resolve().parent
case_id='original_peak08_fee5'
raw=next((P/'native/results'/case_id).glob('*/original.json'))
source=json.loads(raw.read_text())
receipt=json.loads((raw.parent/'receipt.json').read_text())
assert 'ended_epoch' not in receipt
assert source['status']==2 and source['fee']==5
j.f.prices={int(r['time_block']):float(r['cost']) for r in csv.DictReader((P/'bundle/tariffs/peak08.csv').open())}
# The original tolerance must reject the unmodified witness.
try:
    rc.validate(source,j)
except AssertionError:
    pass
else:
    raise AssertionError('Expected raw validation failure disappeared')
fixed=copy.deepcopy(source)
route=fixed['routes'][2]
old_end=route['charges'][3]['end']
route['charges'][3]['end']+=0.0001  # minutes = 0.006 seconds, within existing visit/hour slack
source_route=j.REPLAY['original']['routes'][2]
idle=j.pr.idle_kw/60
soc=j.pr.usable_capacity_kwh-source_route['actions'][0]['deadhead_kwh']
first=j.TR[source_route['trips'][0]]
trace=[[first['start']-source_route['actions'][0]['travel_min'],j.pr.usable_capacity_kwh],[first['start'],soc]]
for action in source_route['actions'][1:]:
    trip=j.TR[action['from_trip']]
    soc-=trip['energy'];trace.append([trip['end'],soc])
    if action['kind']=='direct':
        soc-=action['deadhead_kwh']+action['idle_kwh']
    else:
        arrival,latest=action['arrival_min'],action['latest_departure_min']
        soc-=action['inbound_kwh']
        matches=[c for c in route['charges'] if c['station']==action['station'] and c['start']>=arrival-1e-5 and c['end']<=latest+1e-5]
        assert len(matches)<=1
        if matches:
            c=matches[0];soc-=idle*(c['start']-arrival)
            after,cost=rc.independent_charge_integral(j.pr,c['station'],c['start'],c['end'],soc,j.f.prices)
            checkcost,points=j.f.cost_profile(j.pr,c['station'],c['start'],soc,after)
            assert abs(checkcost-cost)<1e-8 and abs(points[-1][0]-c['end'])<1e-7
            c.update(kwh=after-soc,cost=cost,points=points)
            trace.append([c['start'],soc]);trace.extend(points[1:])
            soc=after-idle*(latest-c['end'])
        else:
            soc-=idle*(latest-arrival)
        soc-=action['outbound_kwh']
    end=(j.TR[action['next_trip']]['start'] if action['next_trip'] is not None else (action['latest_departure_min']+action['outbound_min'] if action['kind']=='charge' else trip['end']+action['travel_min']))
    trace.append([end,soc])
route.update(trace=trace,terminal_kwh=soc,electricity_cost=sum(c['cost'] for c in route['charges']))
fixed['solver_objective_before_repair']=source['objective']
fixed['objective']=sum(r['electricity_cost']+fixed['fee']*r['starts'] for r in fixed['routes'])
fixed['physical_witness_kind']='deterministic numerical repair; original solver status and bound retained'
check=rc.validate(fixed,j)
assert fixed['bound']==source['bound'] and fixed['objective']>=fixed['bound']
# Confirm the repair cannot conceal a material floor or energy error.
for field in ['terminal_kwh','target_kwh']:
    bad=copy.deepcopy(fixed);bad['routes'][2][field]+=0.001
    try:rc.validate(bad,j)
    except AssertionError:pass
    else:raise AssertionError('Corrupted repaired witness passed')
changes=[]
for before,after in zip(source['routes'][2]['charges'],route['charges']):
    changes.append(dict(station=before['station'],start=before['start'],end_before=before['end'],end_after=after['end'],kwh_change=after['kwh']-before['kwh'],cost_change=after['cost']-before['cost']))
report=dict(case_id=case_id,raw_result_path=str(raw),raw_result_sha256=rc.sha(raw),raw_receipt_sha256=rc.sha(raw.parent/'receipt.json'),raw_failed_terminal_shortfall_kwh=2.8157760262104148e-5,
    repair='Extend route index 2 charge index 3 at 2190L by 0.0001 minute (6 milliseconds); independently replay its entire duty and recompute every downstream energy and tariff cost.',
    altered_clock_count=1,charge_end_before=old_end,charge_end_after=route['charges'][3]['end'],terminal_floor_margin_kwh=soc-route['target_kwh'],
    raw_solver_objective=source['objective'],feasible_repaired_objective=fixed['objective'],unchanged_solver_bound=source['bound'],repaired_relative_gap=(fixed['objective']-source['bound'])/abs(fixed['objective']),
    validation_thresholds_unchanged=True,solver_rerun=False,raw_artifacts_unchanged=True,charges=changes,
    validator_sha256=rc.sha(P/'run_case.py'),repair_script_sha256=rc.sha(Path(__file__)),corrupt_repaired_terminal_and_target_rejected=True)
fixed['repair_provenance']=report
check['witness_repaired']=True
out=P/'postprocess_repair'/case_id
out.mkdir(parents=True,exist_ok=True)
(out/'original.json').write_text(json.dumps(fixed,indent=2)+'\n')
(out/'validation.json').write_text(json.dumps(check,indent=2)+'\n')
(out/'repair_report.json').write_text(json.dumps(report,indent=2)+'\n')
r=copy.deepcopy(receipt)
r.update(ended_epoch=time.time(),physical_validation=True,status=source['status'],result_sha256=rc.sha(out/'original.json'),validation_sha256=rc.sha(out/'validation.json'),
    witness_repaired=True,solver_artifacts_directory=str(raw.parent),raw_receipt_path=str(raw.parent/'receipt.json'),raw_receipt_sha256=rc.sha(raw.parent/'receipt.json'))
r['artifact_sha256']={q.name:rc.sha(q) for q in out.iterdir() if q.is_file() and q.name!='receipt.json'}
(out/'receipt.json').write_text(json.dumps(r,indent=2)+'\n')
assert rc.sha(raw)==report['raw_result_sha256']
print(json.dumps(report,indent=2))
