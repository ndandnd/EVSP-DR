"""Bounded local LP repair; fixed route nodes and historical physics, capacity236.44.
No CG, no route changes, no new station visits. Uses SciPy HiGHS LP only.
"""
from pathlib import Path
import csv,json,hashlib,collections,time,subprocess,sys
import numpy as np
from scipy.optimize import linprog
P=Path(__file__).resolve().parent; B=P.parent/'battery_rounding'; ROOT=P.parents[2]
ns={'__file__':str(B/'replay.py')};exec((B/'replay.py').read_text().split('source_meta=')[0],ns)
arc,minutes=ns['arc'],ns['minutes'];CAP=236.44;RATE=4.;TOL=1e-6
sourcehash={};records=[];solutions=[];witness=[];cache={};started=time.perf_counter()
def js(p):
 b=p.read_bytes();sourcehash[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return json.loads(b)
def csvrows(p):
 b=p.read_bytes();sourcehash[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return list(csv.DictReader(b.decode().splitlines()))
def writecsv(name,r):
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,list(r[0]));w.writeheader();w.writerows(r)
def stops_at_nodes(r):
 stops=r['charging_stops']; si=0;result={}
 for pos,node in enumerate(r['route_nodes'][1:-1],1):
  if isinstance(node,str) and si<len(stops['stations']) and node==stops['stations'][si]:result[pos]=si;si+=1
 assert si==len(stops['stations'])
 return result
def geometry(r,tr):
 nodes=r['route_nodes'];sm=stops_at_nodes(r); arcs=[]; windows={}
 for pos,(left,right) in enumerate(zip(nodes,nodes[1:]),1):
  a=tr[left]['To1'] if isinstance(left,int) else left;b=tr[right]['From1'] if isinstance(right,int) else right
  arcs.append(arc(a,b))
 for pos,si in sm.items():
  prev,nxt=nodes[pos-1],nodes[pos+1]
  assert isinstance(prev,int) or pos==1
  assert isinstance(nxt,int) or pos==len(nodes)-2
  arrival=minutes(tr[prev]['End1'])+arcs[pos-1][0] if isinstance(prev,int) else arcs[pos-1][0]
  deadline=minutes(tr[nxt]['Start1'])-arcs[pos][0] if isinstance(nxt,int) else 1560-arcs[pos][0]
  windows[si]=(arrival,deadline)
 return arcs,sm,windows

def solve(r,tr,stage):
 arcs,sm,windows=geometry(r,tr);blocks=r['continuous_realized_charging_blocks'];st=r['charging_stops']
 if stage=='existing_intervals':
  slots=[dict(stop_index=b['stop_index'],station=b['station'],start_min=max(b['start_min'],windows[b['stop_index']][0]),end_min=b['end_min'],old_kwh=b['realized_kwh'],price=b['price_per_kwh']) for b in blocks]
 else:
  slots=[]
  for si,station in enumerate(st['stations']):
   prices={b['price_per_kwh'] for b in blocks if b['stop_index']==si}; assert len(prices)==1
   arrival,deadline=windows[si]
   slots.append(dict(stop_index=si,station=station,start_min=arrival,end_min=deadline,old_kwh=st['kwh'][si],price=prices.pop()))
 n=len(slots)
 if not n:return None,dict(status='no_existing_charging_visit',solver_status=2)
 # x energy, plus/minus deviations; minimize total absolute energy perturbation.
 A=[];rhs=[];row=np.zeros(n);constant=CAP
 def bounds():
  A.extend([np.r_[-row,np.zeros(2*n)],np.r_[row,np.zeros(2*n)]]);rhs.extend([constant,CAP-constant])
 bounds()
 for pos,node in enumerate(r['route_nodes'][1:],1):
  constant-=arcs[pos-1][1];bounds()
  if isinstance(node,int):constant-=float(tr[node]['Usage kWh']);bounds()
  elif pos in sm:
   for i,s in enumerate(slots):
    if s['stop_index']==sm[pos]:row[i]=1;bounds()
 eq=np.c_[np.eye(n),-np.eye(n),np.eye(n)];orig=np.array([s['old_kwh'] for s in slots]);caps=[max(0,(s['end_min']-s['start_min'])*RATE) for s in slots]
 result=linprog(np.r_[np.zeros(n),np.ones(2*n)],A_ub=np.array(A),b_ub=np.array(rhs),A_eq=eq,b_eq=orig,bounds=[(0,c) for c in caps]+[(0,None)]*(2*n),method='highs',options={'time_limit':10,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
 info=dict(status=result.message,solver_status=int(result.status))
 if not result.success:return None,info
 out=[]
 for s,kwh in zip(slots,result.x[:n]):
  s=dict(s,kwh=max(0.,float(kwh)))
  if stage=='retimed_same_visits':
   duration=s['kwh']/RATE; oldstart=st['cst'][s['stop_index']]
   s['start_min']=min(max(oldstart,s['start_min']),s['end_min']-duration);s['end_min']=s['start_min']+duration
  out.append(s)
 return out,info

def replay(r,tr,slots):
 """Independent forward arithmetic and chronology check, no LP matrices reused."""
 nodes=r['route_nodes']; stopmap=stops_at_nodes(r);e=CAP;minimum=e;maximum=e;t=None;viol=[];trace=[];consumed=set()
 for pos,(prev,node) in enumerate(zip(nodes,nodes[1:]),1):
  a=tr[prev]['To1'] if isinstance(prev,int) else prev;b=tr[node]['From1'] if isinstance(node,int) else node
  travel,dh=arc(a,b);arrival=None if t is None else t+travel;e-=dh;minimum=min(minimum,e);maximum=max(maximum,e);trace.append([pos,'deadhead',e])
  if isinstance(node,int):
   if arrival is not None and arrival>minutes(tr[node]['Start1'])+TOL:viol.append('late_trip')
   e-=float(tr[node]['Usage kWh']);minimum=min(minimum,e);maximum=max(maximum,e);t=minutes(tr[node]['End1']);trace.append([pos,'trip',e])
  elif pos in stopmap:
   ss=sorted([(i,s) for i,s in enumerate(slots) if s['stop_index']==stopmap[pos]],key=lambda z:z[1]['start_min']);last=arrival
   for i,s in ss:
    if last is not None and s['start_min']<last-TOL:viol.append('overlap_or_late_charge')
    if s['end_min']<s['start_min']-TOL or s['kwh']>(s['end_min']-s['start_min'])*RATE+TOL:viol.append('charge_power')
    if s['station']!=node:viol.append('station_changed')
    e+=s['kwh'];minimum=min(minimum,e);maximum=max(maximum,e);trace.append([pos,'charge',e]);last=s['end_min'];consumed.add(i)
   t=last
  else:t=arrival
 if e<-TOL:viol.append('terminal_below_zero')
 if minimum<-TOL:viol.append('reserve')
 if maximum>CAP+TOL:viol.append('capacity')
 if t is not None and t>1560+TOL:viol.append('horizon')
 if len(consumed)!=len(slots):viol.append('unconsumed_charges')
 return dict(valid=not viol,violations=viol,min_soc_kwh=minimum,max_soc_kwh=maximum,terminal_kwh=e,trace=trace)

rr=csvrows(B/'per_route.csv');lookup={(r['case'],r['arm'],int(r['route_index'])):r for r in rr if r['scenario']=='all_18E1_capacity_sensitivity'}
for f in sorted((B/'sources').glob('*_result.json')):
 case,arm=f.name.rsplit('_',2)[:2];sol=js(f);tr=csvrows(B/'sources'/(case+'.csv'))
 for index,r in enumerate(sol['selected_routes']):
  key=case,arm,index;base=lookup[key];fingerprint=base['route_schedule_sha256'];failed=base['frozen_schedule_valid']=='False'
  oldslots=[dict(stop_index=b['stop_index'],station=b['station'],start_min=b['start_min'],end_min=b['end_min'],kwh=b['realized_kwh'],old_kwh=b['realized_kwh'],price=b['price_per_kwh']) for b in r['continuous_realized_charging_blocks']]
  if fingerprint in cache:out=cache[fingerprint]
  else:
   stage='unchanged';slots=oldslots;attempts={};before=replay(r,tr,oldslots)
   assert before['valid']==(not failed),(key,before)
   if failed:
    for stage in ['existing_intervals','retimed_same_visits']:
     slots,info=solve(r,tr,stage);attempts[stage]=info
     if slots is not None:break
    if slots is None:stage='unresolved'
   after=replay(r,tr,slots) if slots is not None else None
   assert after is None or after['valid'],(key,after)
   oldstop=r['charging_stops'];oldenergy=sum(oldstop['kwh']);newenergy=sum(s['kwh'] for s in slots) if slots is not None else None
   oldcost=sum(s['kwh']*s['price'] for s in oldslots)
   # Compare per-stop changes irrespective of block timing representation.
   newer=[sum(s['kwh'] for s in slots if s['stop_index']==j) for j in range(len(oldstop['kwh']))] if slots is not None else []
   grossadd=sum(max(0,n-o) for n,o in zip(newer,oldstop['kwh']));grossreduce=sum(max(0,o-n) for n,o in zip(newer,oldstop['kwh']))
   addedwindow=0.;startshift=0.;endshift=0.
   if slots is not None:
    for si in range(len(newer)):
     ss=[s for s in slots if s['stop_index']==si]
     if ss:
      start=min(s['start_min'] for s in ss);end=max(s['end_min'] for s in ss)
      addedwindow+=max(0,oldstop['cst'][si]-start)+max(0,end-oldstop['cet'][si]);startshift+=abs(start-oldstop['cst'][si]);endshift+=abs(end-oldstop['cet'][si])
   newstarts=sum(x>1e-6 for x in newer);oldstarts=sum(x>1e-6 for x in oldstop['kwh']);newcost=sum(s['kwh']*s['price'] for s in slots) if slots is not None else None
   metrics=dict(failed_before=failed,repair_stage=stage,repaired=slots is not None,independent_replay_passed=bool(after and after['valid']),before_min_soc_kwh=before['min_soc_kwh'],after_min_soc_kwh=after['min_soc_kwh'] if after else '',after_terminal_kwh=after['terminal_kwh'] if after else '',old_charged_kwh=oldenergy,new_charged_kwh=newenergy,net_added_kwh=newenergy-oldenergy if after else '',gross_added_kwh=grossadd,gross_reduced_kwh=grossreduce,added_full_power_equivalent_minutes=grossadd/RATE,added_outside_original_windows_min=addedwindow,total_start_shift_min=startshift,total_end_shift_min=endshift,old_electricity_cost=oldcost,new_electricity_cost=newcost,electricity_cost_change=newcost-oldcost if after else '',old_charge_starts=oldstarts,new_charge_starts=newstarts,start_fee=5,charging_objective_change=(newcost-oldcost+5*(newstarts-oldstarts)) if after else '',continuous_outside_original_event_grid=failed and slots is not None,unresolved_reason=json.dumps(attempts) if stage=='unresolved' else '')
   out=dict(metrics=metrics,slots=slots,replay=after,attempts=attempts);cache[fingerprint]=out
  # Numeric schedule fingerprint normalizes source IDs; map references back to this case's nodes.
  if out['slots'] is not None:assert replay(r,tr,out['slots'])['valid']
  records.append(dict(case=case,arm=arm,route_index=index,route_schedule_sha256=fingerprint,**out['metrics']))
  witness.append(dict(case=case,arm=arm,route_index=index,route_nodes=r['route_nodes'],trips=r['trips'],capacity_kwh=CAP,charge_kw=240,reserve_kwh=0,terminal_floor_kwh=None,**out))
for case,arm in sorted({(r['case'],r['arm']) for r in records}):
 rs=[r for r in records if r['case']==case and r['arm']==arm]
 solutions.append(dict(case=case,arm=arm,routes=len(rs),failed_before=sum(r['failed_before'] for r in rs),fixed_existing_intervals=sum(r['repair_stage']=='existing_intervals' for r in rs),fixed_retimed_visits=sum(r['repair_stage']=='retimed_same_visits' for r in rs),unresolved=sum(not r['repaired'] for r in rs),whole_solution_energy_valid=all(r['independent_replay_passed'] for r in rs),net_added_kwh=sum(r['net_added_kwh'] for r in rs if r['net_added_kwh']!=''),gross_added_kwh=sum(r['gross_added_kwh'] for r in rs),electricity_cost_change=sum(r['electricity_cost_change'] for r in rs if r['electricity_cost_change']!=''),charging_objective_change=sum(r['charging_objective_change'] for r in rs if r['charging_objective_change']!=''),added_outside_original_windows_min=sum(r['added_outside_original_windows_min'] for r in rs)))
writecsv('per_route.csv',records);writecsv('per_solution.csv',solutions);(P/'repaired_schedules.json').write_text(json.dumps(witness,indent=2))
summary=dict(route_occurrences=len(records),distinct_schedules=len(cache),initial_failures=sum(r['failed_before'] for r in records),stages=dict(collections.Counter(r['repair_stage'] for r in records)),distinct_stages=dict(collections.Counter(r['metrics']['repair_stage'] for r in cache.values())),whole_solutions_valid=sum(r['whole_solution_energy_valid'] for r in solutions),solutions=len(solutions),net_added_energy_kwh=sum(r['net_added_kwh'] for r in records if r['net_added_kwh']!=''),gross_added_energy_kwh=sum(r['gross_added_kwh'] for r in records),electricity_cost_change=sum(r['electricity_cost_change'] for r in records if r['electricity_cost_change']!=''),charging_objective_change=sum(r['charging_objective_change'] for r in records if r['charging_objective_change']!=''),added_outside_original_windows_min=sum(r['added_outside_original_windows_min'] for r in records),wall_seconds=time.perf_counter()-started)
(P/'summary.json').write_text(json.dumps(summary,indent=2));sourcehash[str((B/'replay.py').relative_to(ROOT))]=hashlib.sha256((B/'replay.py').read_bytes()).hexdigest()
(P/'provenance.json').write_text(json.dumps(dict(execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_hashes=sourcehash,lookup_source_hashes=ns['used'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),resources='Local serial SciPy HiGHS LP; no scheduler jobs, no CG, per-LP10second limit',physics=dict(capacity_kwh=CAP,initial_kwh=CAP,charge_kw=240,reserve_kwh=0,terminal_floor=None,shared_capacity=False,idle_kw=0),objective='Minimize L1 change of charged energy at fixed existing blocks; fallback same existing stop visits with continuous time windows',master_sense='No fleet/master solve; original cover selections retained',initialization='Frozen saved MIP selected schedules',dependencies='Original immutable result/instance/reference inputs; no new scheduler dependency',output_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.name!='provenance.json' and f.is_file()}),indent=2))
print(json.dumps(summary,indent=2))
