"""Matched GIRO physics, existing movement and charging-window envelopes.
No route reordering. Gurobi PWL charging-time potential exactly integrates
piecewise-constant taper; charging inside each tariff-hour segment runs at
maximum feasible power, then idles connected. A single source recharge window
is one connection/start even if held at zero power across a tariff boundary.
This is a restricted fixed-duty charging MILP, not full joint CG.
"""
from pathlib import Path
import csv,json,hashlib,sys,math
import numpy as np
import gurobipy as gp
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
CODE=ROOT/'.codex-work/review-strict-chain-20260916/src';sys.path.insert(0,str(CODE))
from giro_partille_physics import profile_for_duty,base_site,charge_soc_after_minutes
SOURCE=ROOT/'data/Par_VehicleDetails_Updated.csv'
ORIG=ROOT/'outputs/meeting_20260917/route_explainer/inputs/original.json'
TARIFF=P/'inputs/peak08_h26.csv'
PROFILES=CODE/'giro_partille_physics.py'
TOL=1e-5
minutes=lambda s:sum(int(x)*m for x,m in zip(s.split(':'),(60,1)))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prices={int(r['time_block']):float(r['cost']) for r in csv.DictReader(TARIFF.open())}
ids=[r['duty_id'] for r in json.loads(ORIG.read_text())['routes']]
rows={i:[] for i in ids}
for lineno,r in enumerate(csv.DictReader(SOURCE.open()),2):
 if r['VehicleTask'] in rows:
  r=dict(r,line=lineno,start=minutes(r['Start1']),end=minutes(r['End1']))
  rows[r['VehicleTask']].append(r)

def potential(profile,station):
 c=profile.usable_capacity_kwh
 if base_site(station)=='PARX':return [0,c],[0,c]
 assert base_site(station) in profile.allowed_opportunity_sites
 xx=[0]+[x.maximum_soc_fraction*c for x in profile.opportunity_curve]
 yy=[0]
 for a,b,band in zip(xx,xx[1:],profile.opportunity_curve):
  power=271 if profile.name=='18E2' and base_site(station)=='3127L' and abs(band.maximum_soc_fraction-.8)<1e-9 else band.power_kw
  yy.append(yy[-1]+(b-a)*60/power)
 return xx,yy

def h(profile,station,s):
 xx,yy=potential(profile,station);return float(np.interp(s,xx,yy))

def split(a,b):
 while a<b-1e-8:
  e=min(b,(int(a//60)+1)*60);yield a,e;a=e

def cost_profile(profile,station,start,before,after):
 # Exact tariff invoice for maximum-taper charging from start to target SOC.
 soc=before;t=start;cost=0;points=[(t,soc)];xx,yy=potential(profile,station)
 while soc<after-1e-7:
  band=next((b for b in xx[1:] if b>soc+1e-7),xx[-1]);end_soc=min(after,band)
  dur=h(profile,station,end_soc)-h(profile,station,soc)
  end_t=t+dur;energy=end_soc-soc
  for a,b in split(t,end_t):cost+=energy*(b-a)/dur*prices[int(a//60)]
  t=end_t;soc=end_soc;points.append((t,soc))
 return cost,points

def baseline():
 result=[]
 for duty,rr in rows.items():
  profile=profile_for_duty(duty);soc=profile.usable_capacity_kwh;prev=rr[0]['start'];trace=[];charges=[];viol=[];cost=0
  for r in rr:
   gap=r['start']-prev;assert gap>=0
   soc-=gap*profile.idle_kw/60;trace.append([r['start'],soc])
   before=soc
   if r['Identifier']=='Recharge':
    q=float(r['Recharge kWh']);station=base_site(r['From1']);dur=h(profile,station,soc+q)-h(profile,station,soc)
    if dur>r['end']-r['start']+1e-4:viol.append({'line':r['line'],'type':'curve_time','required':dur,'available':r['end']-r['start']})
    actual_cost,pts=cost_profile(profile,station,r['start'],soc,soc+q);cost+=actual_cost;trace.extend(pts[1:]);soc+=q;soc-=max(0,r['end']-r['start']-dur)*profile.idle_kw/60
    charges.append(dict(line=r['line'],station=station,start=r['start'],end=r['end'],kwh=q,segments=[dict(start=r['start'],end=r['end'],before=before,after_charge=before+q,active_min=dur,points=pts)],cost=actual_cost))
   else:soc-=float(r['Usage kWh'] or 0)
   trace.append([r['end'],soc]);prev=r['end']
   if soc<profile.reserve_kwh-1e-4:viol.append({'line':r['line'],'type':'reserve','soc':soc})
  result.append(dict(duty=duty,charges=charges,trace=trace,terminal_kwh=soc,electricity_cost=cost,starts=len(charges),minimum_soc_kwh=min(x[1] for x in trace),violations=viol))
 return result

BASE=baseline();TARGET={r['duty']:r['terminal_kwh'] for r in BASE}

def solve(fee):
 m=gp.Model('k5_matched_fixed_window');m.Params.Threads=4;m.Params.TimeLimit=120;m.Params.MIPGap=0;m.Params.LogFile=str(P/f'fee{fee}.gurobi.log');m.Params.FeasibilityTol=1e-7;m.Params.NumericFocus=3;m.Params.Presolve=2
 records={};objective=gp.LinExpr()
 for duty,rr in rows.items():
  pr=profile_for_duty(duty);c=pr.usable_capacity_kwh;idle=pr.idle_kw/60
  soc=m.addVar(lb=c,ub=c,name=f'{duty}_initial');prev=rr[0]['start'];records[duty]=[]
  for r in rr:
   before=m.addVar(lb=pr.reserve_kwh,ub=c,name=f's{r["line"]}');m.addConstr(before==soc-idle*(r['start']-prev))
   if r['Identifier']!='Recharge':
    after=m.addVar(lb=pr.reserve_kwh,ub=c);m.addConstr(after==before-float(r['Usage kWh'] or 0));soc=after
    records[duty].append(dict(row=r,before=before,after=after));prev=r['end'];continue
   station=base_site(r['From1']);xx,yy=potential(pr,station);y=m.addVar(vtype=gp.GRB.BINARY,name=f'connection{r["line"]}');objective+=fee*y;parts=[];durations=[]
   for seg,(a,b) in enumerate(split(r['start'],r['end'])):
    ch=m.addVar(lb=pr.reserve_kwh,ub=c);ha=m.addVar(lb=0,ub=yy[-1]);hb=m.addVar(lb=0,ub=yy[-1]);d=m.addVar(lb=0,ub=b-a);q=m.addVar(lb=0,ub=c);end=m.addVar(lb=pr.reserve_kwh,ub=c)
    m.addGenConstrPWL(before,ha,xx,yy);m.addGenConstrPWL(ch,hb,xx,yy);m.addConstr(d==hb-ha);m.addConstr(q==ch-before);m.addConstr(q<=c*y);m.addConstr(end==ch-idle*((b-a)-d));objective+=prices[int(a//60)]*q
    parts.append(dict(start=a,end=b,before=before,after_charge=ch,after=end,active=d,q=q));durations.append(d);before=end
   m.addConstr(gp.quicksum(durations)>=pr.minimum_recharge_duration_min*y)
   records[duty].append(dict(row=r,before=parts[0]['before'],after=end,parts=parts,connection=y));soc=end;prev=r['end']
  m.addConstr(soc>=TARGET[duty]-0.001,name=f'equal_terminal_lower_{duty}');m.addConstr(soc<=TARGET[duty]+0.001,name=f'equal_terminal_upper_{duty}')
 m.setObjective(objective);m.optimize();assert m.SolCount>0
 result=[]
 for duty,rr in records.items():
  pr=profile_for_duty(duty);trace=[];charges=[];cost=0;viol=[]
  for rec in rr:
   r=rec['row'];trace.append([r['start'],rec['before'].X])
   if 'parts' in rec:
    parts=[];qsum=0;cc=0
    for p in rec['parts']:
     before=p['before'].X;after=p['after_charge'].X;dur=p['active'].X;q=p['q'].X
     assert abs((h(pr,r['From1'],after)-h(pr,r['From1'],before))-dur)<1e-5
     assert abs(charge_soc_after_minutes(pr,r['From1'],before,dur)-after)<1e-4
     cs,pts=cost_profile(pr,r['From1'],p['start'],before,after);cost+=cs;cc+=cs;qsum+=q;trace.extend(pts[1:]);trace.append([p['end'],p['after'].X])
     parts.append(dict(start=p['start'],end=p['end'],before=before,after_charge=after,active_min=dur,kwh=q,points=pts))
    if rec['connection'].X>.5:charges.append(dict(line=r['line'],station=base_site(r['From1']),start=r['start'],end=r['end'],kwh=qsum,segments=parts,cost=cc))
   trace.append([r['end'],rec['after'].X])
  assert min(x[1] for x in trace)>=pr.reserve_kwh-1e-5
  assert abs(trace[-1][1]-TARGET[duty])<0.00101
  result.append(dict(duty=duty,charges=charges,trace=trace,terminal_kwh=trace[-1][1],electricity_cost=cost,starts=len(charges),minimum_soc_kwh=min(x[1] for x in trace),violations=viol))
 expected=sum(r['electricity_cost']+fee*r['starts'] for r in result);assert abs(expected-m.ObjVal)<1e-4
 out=dict(fee=fee,solver_status=m.Status,objective=m.ObjVal,bound=m.ObjBound,gap=m.MIPGap,wall_seconds=m.Runtime,routes=result,scope='fixed original movement, existing charging-window envelope; exact taper time, startfee per connection',physics=dict(group='18E1',battery_kwh=236.44,reserve_fraction=.15,idle_kw=.1,opportunity_curve_kw=[x.power_kw for x in pr.opportunity_curve],parx_kw=60,minimum_active_charge_minutes=3,setup_minutes=0,terminal='each duty within 0.001 kWh of original replay (rounding tolerance)'),capacity='original full occupancy windows retained as upper envelope; no new occupancy')
 (P/f'fee{fee}.json').write_text(json.dumps(out,indent=2));m.write(str(P/f'fee{fee}.lp'));m.dispose();return out

if __name__=='__main__':
 assert all(not r['violations'] for r in BASE),BASE
 (P/'baseline.json').write_text(json.dumps(dict(routes=BASE),indent=2))
 out=[solve(0),solve(5)]
 manifest=dict(inputs={str(p):sha(p) for p in [SOURCE,ORIG,TARIFF,PROFILES]},execution_script_sha256=sha(__file__),comparison_scope='All five original GIRO duties, same62trips, reoptimized within existing charge windows. Not jointCG or route reassignment.',original_duties=ids,targets=TARGET,results=[dict(fee=r['fee'],status=r['solver_status'],cost=sum(x['electricity_cost'] for x in r['routes']),starts=sum(x['starts'] for x in r['routes']),objective=r['objective']) for r in out])
 (P/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest['results'],indent=2))
