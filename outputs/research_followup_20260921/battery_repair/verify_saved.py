"""Reload serialized witnesses; independently check every occurrence against sources."""
from pathlib import Path
import json,hashlib,csv,statistics,collections
P=Path(__file__).resolve().parent
ns={'__file__':str(P/'repair.py')}
exec((P/'repair.py').read_text().split('rr=csvrows(')[0],ns)
B=ns['B'];replay=ns['replay'];geometry=ns['geometry'];MAXTOL=1e-6
saved=json.loads((P/'repaired_schedules.json').read_text());rows=list(csv.DictReader((P/'per_route.csv').open()));fleet=list(csv.DictReader((P/'per_solution.csv').open()))
peak=0.;count=0;tripcount=0;starts=0;window_spare=0.;within=0;retimed=0;min_soc=236.44;max_soc=0.
for w in saved:
 source=json.loads((B/'sources'/f"{w['case']}_{w['arm']}_result.json").read_text())['selected_routes'][w['route_index']]
 tr=list(csv.DictReader((B/'sources'/(w['case']+'.csv')).open()))
 assert w['trips']==source['trips'] and w['route_nodes']==source['route_nodes']
 assert w['capacity_kwh']==236.44 and w['charge_kw']==240 and w['reserve_kwh']==0 and w['terminal_floor_kwh'] is None
 validation=replay(source,tr,w['slots']);assert validation['valid'],(w['case'],w['arm'],w['route_index'],validation)
 min_soc=min(min_soc,validation['min_soc_kwh']);max_soc=max(max_soc,validation['max_soc_kwh'])
 assert abs(validation['terminal_kwh']-w['metrics']['after_terminal_kwh'])<MAXTOL
 old=source['continuous_realized_charging_blocks'];stage=w['metrics']['repair_stage']
 if stage in ['unchanged','existing_intervals']:
  assert len(old)==len(w['slots'])
  for before,after in zip(old,w['slots']):
   assert before['stop_index']==after['stop_index'] and before['station']==after['station']
   assert abs(before['start_min']-after['start_min'])<MAXTOL and abs(before['end_min']-after['end_min'])<MAXTOL
  within+=1
 else:
  assert stage=='retimed_same_visits';retimed+=1
 original_visits=len(source['charging_stops']['kwh']);original_starts=sum(x>MAXTOL for x in source['charging_stops']['kwh']);new_starts=sum(sum(b['kwh'] for b in w['slots'] if b['stop_index']==i)>MAXTOL for i in range(original_visits));assert original_starts==new_starts
 starts+=new_starts;tripcount+=len(source['trips']);count+=1
 for b in w['slots']:
  if b['end_min']>b['start_min']:peak=max(peak,b['kwh']/(b['end_min']-b['start_min'])*60)
 for b in old:window_spare+=(b['end_min']-b['start_min'])*4-b['realized_kwh']
failed=[r for r in rows if r['failed_before']=='True']
summary=dict(valid=True,route_occurrences_checked=count,original_trip_order_and_nodes_preserved=True,trip_occurrences_preserved=tripcount,charging_starts_before_and_after=starts,occurrences_with_original_intervals_preserved=within,occurrences_with_retiming=retimed,max_power_kw=peak,min_soc_kwh=min_soc,max_soc_kwh=max_soc,baseline_total_unused_interval_energy_capacity_kwh=window_spare,zero_net_energy_change_repairs=sum(abs(float(r['net_added_kwh']))<1e-6 for r in failed),modified_route_net_energy_kwh=dict(min=min(float(r['net_added_kwh']) for r in failed),median=statistics.median(float(r['net_added_kwh']) for r in failed),max=max(float(r['net_added_kwh']) for r in failed)),modified_route_gross_addition_kwh=dict(min=min(float(r['gross_added_kwh']) for r in failed),median=statistics.median(float(r['gross_added_kwh']) for r in failed),max=max(float(r['gross_added_kwh']) for r in failed)),fleet_cost_increment=dict(min=min(float(r['charging_objective_change']) for r in fleet),median=statistics.median(float(r['charging_objective_change']) for r in fleet),max=max(float(r['charging_objective_change']) for r in fleet)),witness_sha256=hashlib.sha256((P/'repaired_schedules.json').read_bytes()).hexdigest(),validation_scope='Fixed route chronology, arcs/deadheads, energy endpoints and power; no charger capacity or operational/full-model proof')
(P/'independent_validation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
