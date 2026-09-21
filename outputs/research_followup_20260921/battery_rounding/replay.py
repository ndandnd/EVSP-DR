"""Deterministic frozen-schedule capacity replay; standard-library only; no solves.
Run from any directory: python3 /absolute/path/to/replay.py.
All energy values are kWh. Charge blocks and route choices are immutable.
"""
from pathlib import Path
import csv,json,hashlib,subprocess,collections,math,sys,statistics
P=Path(__file__).resolve().parent; ROOT=P.parents[2]; S=P/'sources'; TOL=1e-5
used={}
def read(p):
 p=Path(p); b=p.read_bytes(); used[str(p.relative_to(ROOT))]=dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)); return b
def js(p):return json.loads(read(p))
def rows(p):return list(csv.DictReader(read(p).decode('utf-8-sig').splitlines()))
def save(name,data):
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
def norm(x):
 try:return str(int(float(x)))
 except:return x.strip()
def base(x):
 a,sep,b=str(x).rpartition('_');return a if sep and b.isdigit() else str(x)
def minutes(x):
 a,b=x.split(':');return int(a)*60+int(b)
refs={r['Location']:norm(r['Ref']) for r in rows(S/'Ref_dict.csv') if r['Location'] and r['Ref']}
for k,v in list(refs.items()):refs.setdefault(base(k),v)
pairs={}
for r in rows(S/'par_ref_dhd.csv'):
 a,b=norm(r['Start Place']),norm(r['End Place']);key=tuple(sorted([a,b]));v=float(r['Base Duration']),float(r['Energy used'])
 if a!=b and (key not in pairs or v[0]<pairs[key][0]):pairs[key]=v
known={x for p in pairs for x in p}
def ref(x):
 for t in [str(x),base(x),norm(x),norm(base(x))]:
  if t in refs:return refs[t]
  if t in known:return t
 raise ValueError(('unknown location',x))
def arc(a,b):
 if a==b:return 0.,0.
 a,b=ref(a),ref(b)
 return (0.,0.) if a==b else pairs[tuple(sorted([a,b]))]
groups=collections.defaultdict(set)
for r in rows(S/'giro40_duty_manifest.csv'):
 group='18E1' if r['duty_id'].startswith('134') else '18E2'
 for t in json.loads(r['source_ordered_trip_ids_json']):groups[int(t)].add(group)
# Include alternate service-day duty variants excluded from the 40-duty manifest.
for f in sorted((ROOT/'outputs/meeting_20260910/giro_k1_recovery_review/inputs/k1').glob('duty_*.csv')):
 group='18E1' if f.name.startswith('duty_134') else '18E2'
 for r in rows(f):groups[int(r['Ordered_Trip_ID'])].add(group)
source_meta=js(P/'remote_sources.json');targets=js(ROOT/'outputs/cumulative_budget_20260913/audit/targets.json')['targets']
route_rows=[]; solution_rows=[];trace_rows=[]
for file in sorted(S.glob('*_result.json')):
 case,arm=file.name.rsplit('_',2)[:2]; sol=js(file); tr=rows(S/(case+'.csv'))
 assert hashlib.sha256((S/(case+'.csv')).read_bytes()).hexdigest()==targets[case]['input_sha256']
 ph=sol['physics']; assert ph['g_kwh']==240 and ph['charge_kw']==240 and ph['min_soc_frac']==0
 for j,r in enumerate(sol['selected_routes']):
  tg=[groups[int(tr[t]['Ordered_Trip_ID'])] for t in r['trips']];gs=set().union(*tg); unknown=sum(not x for x in tg)
  mapping='unmapped' if unknown else ('homogeneous_source_group' if len(gs)==1 else 'mixed_source_groups_conservative_18E1')
  actual=236.44 if gs!={'18E2'} or unknown else 239.01
  energy=240.; values=[('initial','PARX_0',0.,energy)]; time=None;si=0;issues=[]
  st=r['charging_stops'];stations=st['stations'];blocks=r['continuous_realized_charging_blocks']
  for pos,(prev,node) in enumerate(zip(r['route_nodes'],r['route_nodes'][1:]),1):
   left=tr[prev]['To1'] if isinstance(prev,int) else prev;right=tr[node]['From1'] if isinstance(node,int) else node
   travel,dh=arc(left,right);energy-=dh;arrival=None if time is None else time+travel
   values.append(('deadhead_arrival',node,arrival,energy))
   if isinstance(node,int):
    start=minutes(tr[node]['Start1']);end=minutes(tr[node]['End1'])
    if arrival is not None and arrival>start+TOL:issues.append('late_trip')
    energy-=float(tr[node]['Usage kWh']);time=end;values.append(('service_end',node,time,energy))
   elif pos==len(r['route_nodes'])-1:time=arrival
   elif si<len(stations) and stations[si]==node:
    start,end,kwh=st['cst'][si],st['cet'][si],st['kwh'][si]
    if arrival is not None and arrival>start+1+TOL:issues.append('late_station')
    bb=[b for b in blocks if b['stop_index']==si];assert abs(sum(b['realized_kwh'] for b in bb)-kwh)<TOL
    for b in bb:
     assert b['station']==node
     effective=max(b['start_min'],arrival) if arrival is not None else b['start_min']
     if b['realized_kwh']>(b['end_min']-effective)*240/60+TOL:issues.append('power')
     energy+=b['realized_kwh'];values.append(('charge_block_end',node,b['end_min'],energy))
    time=end;si+=1
   else:time=arrival
  assert si==len(stations)
  assert not issues,(file,j,issues)
  assert abs(energy-r['physical_realization']['continuous_terminal_soc_kwh'])<TOL,(file,j,energy)
  minimum=min(x[3] for x in values);maximum=max(x[3] for x in values)
  assert minimum>=-TOL and maximum<=240+TOL,(file,j,minimum,maximum)
  for event,node,t,e in values:trace_rows.append(dict(case=case,arm=arm,route_index=j,event=event,node=node,time_min=t,soc_240_kwh=e))
  fingerprint=hashlib.sha256(json.dumps(dict(nodes=[int(tr[n]['Ordered_Trip_ID']) if isinstance(n,int) else n for n in r['route_nodes']],blocks=blocks),sort_keys=True).encode()).hexdigest()
  for scenario,cap in [('saved_240_control',240.),('all_18E2_capacity_sensitivity',239.01),('all_18E1_capacity_sensitivity',236.44),('source_group_or_conservative_mixed',actual)]:
   lo=minimum+cap-240;hi=maximum+cap-240;end=energy+cap-240
   route_rows.append(dict(case=case,arm=arm,route_index=j,route_schedule_sha256=fingerprint,scenario=scenario,source_groups=';'.join(sorted(gs)),mapping=mapping,unmapped_trips=unknown,trip_count=len(r['trips']),battery_kwh=cap,initial_kwh=cap,reserve_kwh=0,terminal_floor_kwh='',charge_kw=240,min_soc_kwh=lo,max_soc_kwh=hi,terminal_kwh=end,below_reserve=lo<-TOL,over_capacity=hi>cap+TOL,terminal_violation=False,frozen_schedule_valid=lo>=-TOL and hi<=cap+TOL,minimum_extra_initial_energy_kwh=max(0,-lo),source=str(file.relative_to(ROOT))))
 for scenario in sorted({r['scenario'] for r in route_rows}):
  rr=[r for r in route_rows if r['case']==case and r['arm']==arm and r['scenario']==scenario]
  solution_rows.append(dict(case=case,arm=arm,scenario=scenario,routes=len(rr),failed_routes=sum(not r['frozen_schedule_valid'] for r in rr),below_reserve_routes=sum(r['below_reserve'] for r in rr),over_capacity_routes=sum(r['over_capacity'] for r in rr),min_soc_kwh=min(r['min_soc_kwh'] for r in rr),whole_solution_valid=all(r['frozen_schedule_valid'] for r in rr),mixed_routes=sum(r['mapping'].startswith('mixed') for r in rr),unmapped_routes=sum(bool(r['unmapped_trips']) for r in rr)))
save('per_route.csv',route_rows);save('per_solution.csv',solution_rows);save('baseline_checkpoints.csv',trace_rows)
# Independent arithmetic replay of the already matched 236.44 kWh fee0/5 witnesses.
C=ROOT/'outputs/week_20260921/cleanup_physics'; witness=js(C/'saved_sequence_replay.json');tr=rows(ROOT/'outputs/meeting_20260917/route_explainer/inputs/k05.csv');tr={int(t['count_trip_id']):t for t in tr};matched=[]
for arm in ['saved_joint_fee0','saved_joint_fee5']:
 sol=js(C/(arm+'.json'))
 for r,source in zip(sol['routes'],witness[arm]['routes']):
  e=236.44;vals=[e];e-=source['actions'][0]['deadhead_kwh'];vals.append(e);consumed=0
  for a in source['actions'][1:]:
   e-=float(tr[a['from_trip']]['Usage kWh']);vals.append(e)
   if a['kind']=='direct':e-=a['deadhead_kwh']+a['idle_kwh'];vals.append(e)
   else:
    arrival,end=a['arrival_min'],a['latest_departure_min'];e-=a['inbound_kwh'];vals.append(e)
    cc=[c for c in r['charges'] if c['station']==a['station'] and c['start']>=arrival-TOL and c['end']<=end+TOL];assert len(cc)<=1
    if cc:
     c=cc[0];e-=(c['start']-arrival)*.1/60;vals.append(e)
     # Exact SOC-dependent curve time integral; no optimization and no altered charge.
     remaining=c['kwh'];cur=e;needed=0
     for fraction,power in zip([.1,.2,.3,.4,.5,.6,.7,.8,.9,1],[371.5,357,342.5,328,313.5,299,284.5,270,150,120]):
      gain=min(remaining,max(0,236.44*fraction-cur));needed+=gain/(60 if base(c['station'])=='PARX' else power)*60;cur+=gain;remaining-=gain
     assert remaining<TOL and needed<=c['end']-c['start']+TOL
     e+=c['kwh'];vals.append(e);e-=(end-c['end'])*.1/60;consumed+=1
    else:e-=(end-arrival)*.1/60
    vals.append(e);e-=a['outbound_kwh'];vals.append(e)
  assert consumed==len(r['charges']) and abs(e-r['terminal_kwh'])<TOL
  matched.append(dict(arm=arm,route_index=r['source_index'],battery_kwh=236.44,initial_kwh=236.44,reserve_kwh=35.466,terminal_floor_kwh=r['target_kwh'],min_soc_kwh=min(vals),max_soc_kwh=max(vals),terminal_kwh=e,curve_power_check_passed=True,frozen_schedule_valid=min(vals)>=35.466-TOL and max(vals)<=236.44+TOL and e>=r['target_kwh']-TOL))
save('matched_physics_checks.csv',matched)
summary={}
for arm in ['base','warm','all']:
 summary[arm]={}
 for scenario in sorted({r['scenario'] for r in route_rows}):
  rr=[r for r in route_rows if r['scenario']==scenario and (arm=='all' or r['arm']==arm)];ss=[r for r in solution_rows if r['scenario']==scenario and (arm=='all' or r['arm']==arm)]
  summary[arm][scenario]=dict(routes=len(rr),failed_routes=sum(not r['frozen_schedule_valid'] for r in rr),solutions=len(ss),failed_solutions=sum(not r['whole_solution_valid'] for r in ss),min_soc_kwh=min(r['min_soc_kwh'] for r in rr),over_capacity_routes=sum(r['over_capacity'] for r in rr),mixed_routes=sum(r['mapping'].startswith('mixed') for r in rr),unmapped_routes=sum(bool(r['unmapped_trips']) for r in rr))
# Occurrence-weighted deficits; repeated routes are not independent observations.
for arm in ['base','warm','all']:
 for scenario,ss in summary[arm].items():
  rr=[r for r in route_rows if r['scenario']==scenario and (arm=='all' or r['arm']==arm)]
  dd=sorted(r['minimum_extra_initial_energy_kwh'] for r in rr if not r['frozen_schedule_valid'])
  quant=lambda q: dd[int((len(dd)-1)*q)] if dd else 0.0
  unique={r['route_schedule_sha256']:r for r in rr}
  ss.update(unique_route_schedules=len(unique),unique_failed_route_schedules=sum(not r['frozen_schedule_valid'] for r in unique.values()),failed_route_deficit_median_kwh=statistics.median(dd) if dd else 0,failed_route_deficit_p90_nearest_lower_kwh=quant(.9),failed_route_deficit_p95_nearest_lower_kwh=quant(.95),failed_route_deficit_max_kwh=max(dd,default=0))
summary['matched_physics']=dict(routes=len(matched),failed_routes=sum(not r['frozen_schedule_valid'] for r in matched))
(P/'summary.json').write_text(json.dumps(summary,indent=2));read(__file__)
for pp in [ROOT/'.codex-work/review-strict-chain-20260916/src/giro_partille_physics.py',ROOT/'.codex-work/terminal-replay-compat-20260910/src/run_exact_pool_mip.py',ROOT/'.codex-work/terminal-replay-compat-20260910/src/audit_giro_known_columns.py',ROOT/'outputs/research_management_20260921/paper_results/provenance.json']:read(pp)
(P/'provenance.json').write_text(json.dumps(dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=used,remote_paths=source_meta,scope='Frozen continuous selected schedules; capacity/full initial energy only; no solver or cluster submission',tolerance_kwh=TOL,resource_requests='Local single Python process; no scheduler job or dependency',output_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.glob('*.csv')}),indent=2))
print(json.dumps(summary,indent=2))
