"""Extract geographical semantics of the exact three buses in slide12.
No geocoding or road-routing assumptions. Coordinates stay null until sourced.
"""
from pathlib import Path
import csv,json,hashlib,collections
from openpyxl import load_workbook
P=Path(__file__).resolve().parent;ROOT=P.parents[2];C=P.parent/'cleanup_physics';I=ROOT/'outputs/meeting_20260917/route_explainer/inputs'
MASTER=ROOT/'data/Par_VehicleDetails_Updated.csv';K5=I/'k05.csv';REF=C/'inputs/Ref_dict.csv';DHD=C/'inputs/par_ref_dhd.csv';RAW=ROOT/'outputs/meeting_20260910/giro_email_sources/Par_DHD.xlsm';AUDIT=ROOT/'outputs/meeting_20260910/GIRO_EMAIL_ATTACHMENT_AUDIT.md'
mins=lambda s:sum(int(a)*b for a,b in zip(s.split(':'),(60,1)))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(path):
 return [dict(r,source_line=i) for i,r in enumerate(csv.DictReader(path.open()),2)]
master=read(MASTER);master_trip={r['Ordered_Trip_ID']:r for r in master if r['Identifier']=='Regular'};tr={int(r['count_trip_id']):r for r in read(K5)};byid={r['Ordered_Trip_ID']:i for i,r in tr.items()}
ref={r['Location']:dict(reference=r['Ref'],line=r['source_line']) for r in read(REF)}
refd=read(DHD);lookup={}
for r in refd:
 key=tuple(sorted([r['Start Place'],r['End Place']]))
 if key not in lookup or float(r['Base Duration'])<float(lookup[key]['Base Duration']):lookup[key]=r
raw=[];ws=load_workbook(RAW,read_only=True,data_only=True)['Deadhead'];headers=next(ws.values)
for n,row in enumerate(ws.values,1):
 if n>1:raw.append(dict(zip(headers,row),worksheet_row=n))

def model_arc(a,b):
 ar=ref.get(a,{}).get('reference',a);br=ref.get(b,{}).get('reference',b)
 if ar==br:return dict(travel_min=0.0,energy_kwh=0.0,model_source=str(REF.resolve()),model_source_lines=';'.join(str(ref[x]['line']) for x in [a,b] if x in ref),model_lookup=f'{a}->{ar};{b}->{br};same-reference zero arc',model_basis='static reference graph: coincident reference nodes')
 r=lookup[tuple(sorted([ar,br]))]
 return dict(travel_min=float(r['Base Duration']),energy_kwh=float(r['Energy used']),model_source=str(DHD.resolve()),model_source_lines=str(r['source_line']),model_lookup=f'symmetric unordered reference pair {ar}<->{br}',model_basis='static symmetric reference DHD; minimum-duration row')

names={'PARX':('Partille garage','depot;60kW charger'), '2190':('Eketragatan','service terminal/platform'), '2190L':('Eketragatan departure charger','opportunity charger;1unit'), '4808':('Merkuriusgatan','service terminal;opportunity charger;1unit'), 'ET_R':('ET_R (layover code; formal name unverified)','original duty layover;not a documented charger'), '3127L':('Heden layover area','opportunity charger;2units'), '7880C':('Östra Sjukhuset arrival stop','opportunity charger;1unit'), 'JON_A':('Jons väg departure stop','opportunity charger;1unit')}
locations=[]
for code,(name,role) in names.items():
 locations.append(dict(code=code,name=name,role=role,reference_code=ref.get(code,{}).get('reference'),used_in_plotted_buses=code in {'PARX','2190','2190L','4808','ET_R'},latitude=None,longitude=None,coordinate_status='not present in inspected source data; external verified geocoding required',name_source=str(AUDIT.resolve()) if code!='ET_R' else str(MASTER.resolve()),geographical_note='Same reference node as2190L is a MODEL simplification; do not imply charger and platform physically identical.' if code=='2190' else 'One-minute GIRO links associate ET_R with Eketragatan, but exact position is unverified.' if code=='ET_R' else ''))

trips=[];movements=[];charges=[];activities=[]
selected=[('original_giro_13414',None,None),('saved_joint_fee0',1,0),('saved_joint_fee5',0,5)]
original=[r for r in master if r['VehicleTask']=='13414'];replay=json.loads((C/'saved_sequence_replay.json').read_text());summary=[]
for arm,index,fee in selected:
 if index is None:
  rr=[byid[r['Ordered_Trip_ID']] for r in original if r['Identifier']=='Regular'];physical=json.loads((C/'baseline.json').read_text());pr=next(r for r in physical['routes'] if r['duty']=='13414');actual_charges=pr['charges']
 else:
  sol=json.loads((C/f'{arm}.json').read_text());pr=next(r for r in sol['routes'] if r['source_index']==index);rr=pr['trips'];actual_charges=pr['charges']
 summary.append(dict(arm=arm,source_bus_index=index,original_duty_id='13414' if index is None else None,service_trip_count=len(rr),source_trip_ids=[tr[t]['Ordered_Trip_ID'] for t in rr],charging_starts=len(actual_charges),electricity_cost=pr['electricity_cost'],terminal_kwh=pr['terminal_kwh'],source=str((C/'baseline.json' if index is None else C/f'{arm}.json').resolve())))
 for position,t in enumerate(rr):
  r=tr[t];trips.append(dict(arm=arm,position=position,internal_trip_id=t,source_trip_id=r['Ordered_Trip_ID'],original_trip_duty=master_trip[r['Ordered_Trip_ID']]['VehicleTask'],route_code=master_trip[r['Ordered_Trip_ID']]['Route'],direction=master_trip[r['Ordered_Trip_ID']]['Direction'],from_code=r['From1'],to_code=r['To1'],start_min=mins(r['Start1']),end_min=mins(r['End1']),scheduled_service_duration_min=mins(r['End1'])-mins(r['Start1']),service_energy_kwh=float(r['Usage kWh']),source=str(K5.resolve()),source_line=r['source_line'],basis='scheduled passenger service, NOT deadhead travel time'))
 for ci,ch in enumerate(actual_charges):
  charges.append(dict(arm=arm,charge_index=ci,station=ch['station'],start_min=ch['start'],end_min=ch['end'],charging_duration_min=ch['end']-ch['start'],energy_kwh=ch['kwh'],source=str((C/'baseline.json' if index is None else C/f'{arm}.json').resolve()),source_pointer=f'route duty13414/charges/{ci}' if index is None else f'routes/{index}/charges/{ci}',basis='original recorded connection window' if index is None else 'capacity-rescheduled actual charge interval'))
 if index is None:
  for r in original:
   if r['Identifier'] not in {'Deadhead','Pull-in','Pull-out'}:continue
   record=dict(arm=arm,kind=r['Identifier'],from_code=r['From1'],to_code=r['To1'],start_min=mins(r['Start1']),end_min=mins(r['End1']),travel_min=mins(r['End1'])-mins(r['Start1']),energy_kwh=float(r['Usage kWh'] or 0),distance_km=float(r['Distance1'] or 0),source=str(MASTER.resolve()),source_pointer=f'CSV line{r["source_line"]}',basis='original scheduled directional movement; may reflect departure-time DHD interval')
   try:record['static_model_comparison']=model_arc(r['From1'],r['To1'])
   except KeyError:record['static_model_comparison']=None
   movements.append(record)
 else:
  r=replay[arm]['routes'][index]
  for ai,a in enumerate(r['actions']):
   source=str((C/'saved_sequence_replay.json').resolve());pointer=f'{arm}/routes/{index}/actions/{ai}'
   if a['kind']=='source':
    tid=a['next_trip'];start=mins(tr[tid]['Start1']);pieces=[('pull_out','PARX',tr[tid]['From1'],start-a['travel_min'],start,a['travel_min'],a['deadhead_kwh'])]
   elif a['kind']=='direct':
    tid=a['from_trip'];nextid=a['next_trip'];start=mins(tr[tid]['End1']);pieces=[('pull_in' if nextid is None else 'intertrip_direct',tr[tid]['To1'],'PARX' if nextid is None else tr[nextid]['From1'],start,start+a['travel_min'],a['travel_min'],a['deadhead_kwh'])]
   else:
    tid=a['from_trip'];nextid=a['next_trip'];start=mins(tr[tid]['End1']);st=a['station'];pieces=[('station_inbound',tr[tid]['To1'],st,start,a['arrival_min'],a['inbound_min'],a['inbound_kwh']),('station_outbound',st,'PARX' if nextid is None else tr[nextid]['From1'],a['latest_departure_min'],a['latest_departure_min']+a['outbound_min'],a['outbound_min'],a['outbound_kwh'])]
   for kind,left,right,start,end,dur,en in pieces:
    arcinfo=model_arc(left,right);assert abs(arcinfo['travel_min']-dur)<1e-6 and abs(arcinfo['energy_kwh']-en)<1e-6
    movements.append(dict(arm=arm,kind=kind,from_code=left,to_code=right,start_min=start,end_min=end,travel_min=dur,energy_kwh=en,distance_km=None,source=source,source_pointer=pointer,basis='static model deadhead; event clocks reconstructed by depart-after-service / leave-station-at-next-trip deadline',static_model_comparison=arcinfo))

corridors=[]
for arm in [r[0] for r in selected]:
 for a,b in sorted({(r['from_code'],r['to_code']) for r in trips if r['arm']==arm}):
  group=[r for r in trips if r['arm']==arm and r['from_code']==a and r['to_code']==b]
  corridors.append(dict(arm=arm,from_code=a,to_code=b,service_trip_count=len(group),scheduled_service_min=min(r['scheduled_service_duration_min'] for r in group),scheduled_service_max=max(r['scheduled_service_duration_min'] for r in group),service_kwh_min=min(r['service_energy_kwh'] for r in group),service_kwh_max=max(r['service_energy_kwh'] for r in group),source_trip_ids=';'.join(r['source_trip_id'] for r in group),**{'empty_deadhead_'+k:v for k,v in model_arc(a,b).items() if k in {'travel_min','energy_kwh'}},note='Service duration includes scheduled passenger stops; empty deadhead has a different path/service pattern.'))

raw_subset=[r for r in raw if str(r['Start Place']) in names and str(r['End Place']) in names]
model_links=[]
for a,b in [('PARX','4808'),('4808','PARX'),('PARX','2190'),('2190','PARX'),('2190','4808'),('4808','2190'),('2190','2190L'),('2190L','2190')]:model_links.append(dict(from_code=a,to_code=b,**model_arc(a,b)))

def export(name,rr):
 if not rr:return
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with (P/name).open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rr])
for name,rr in [('locations.csv',locations),('service_trips.csv',trips),('movements.csv',movements),('charging_events.csv',charges),('service_corridors.csv',corridors),('model_deadhead_links.csv',model_links),('raw_directional_dhd.csv',raw_subset)]:export(name,rr)
result=dict(schema='k5_matched_geography_extraction_v1',selected_buses=summary,locations=locations,trips=trips,movements=movements,charging=charges,service_corridors=corridors,model_deadhead_links=model_links,raw_directional_dhd=raw_subset,coordinates_available=False,coordinate_caveat='The old global_stops_hubs.html uses abstract x/y graph coordinates; it is not a geographic coordinate source.',raw_dhd_time_bands=dict(peak1='06:30–08:40;15:00–18:30',morning2='08:41–09:30;14:30–14:59',night3='00:00–05:59;21:00–23:59',base='other times; modulo24h at departure'),sources={str(q.resolve()):sha(q) for q in [MASTER,K5,REF,DHD,RAW,AUDIT,C/'saved_sequence_replay.json',C/'saved_joint_fee0.json',C/'saved_joint_fee5.json',C/'baseline.json',C/'joint_figure_metrics.json',Path(__file__)]},notes=['All travel/energy numbers are source quantities, not geographic distances or Google/OSM road-route estimates.','Do not substitute model deadhead22minutes for the scheduled route21service duration.','Same-reference2190→2190L zeroarc is a known simplification; rawGIRO requires1minute and0.2km≈0.4kWh in that direction.','Fivebusmodelcapacity excludes othercohortbuses; platform/FIFO, directional and time-dependentDHD remain unvalidated.'])
(P/'model_geography.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(summary,indent=2));print(json.dumps(corridors,indent=2))
