"""Reoptimize saved joint-CG sequences on reconstructed, fixed station paths.
Documented E1 energy physics and shared charger count; static reference DHD.
One optional connection per inherited station visit, confined to one tariff hour.
The site/path choice is fixed from nonlinear recovery. No route search, FIFO or
platform-blocking certificate. This restricted model may fail despite feasible
wider joint space. Complete Gurobi logs and physical witnesses are preserved.
"""
from pathlib import Path
import json,sys,itertools,csv
import gurobipy as gp
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import matched_fixed_windows as f
from giro_partille_physics import PARTILLE_PROFILES,charge_soc_after_minutes
REPLAY=json.loads((P/'saved_sequence_replay.json').read_text());I=f.ROOT/'outputs/meeting_20260917/route_explainer/inputs'
TR={int(r['count_trip_id']):dict(start=f.minutes(r['Start1']),end=f.minutes(r['End1']),energy=float(r['Usage kWh']),source_id=r['Ordered_Trip_ID']) for r in csv.DictReader((I/'k05.csv').open())}
ORIG=json.loads((I/'original.json').read_text())['routes'];pr=PARTILLE_PROFILES['18E1'];C=pr.usable_capacity_kwh;reserve=pr.reserve_kwh;idle=pr.idle_kw/60

def solve(arm,fee,seconds=600,threads=2,seed=0):
 routes=REPLAY[arm]['routes'];assert all(r['feasible'] for r in routes)
 perm=max(itertools.permutations(range(5)),key=lambda pp:sum(len(set(routes[j]['trips'])&set(ORIG[k]['trips'])) for j,k in enumerate(pp)))
 target=[f.TARGET[ORIG[k]['duty_id']] for k in perm]
 m=gp.Model(arm+'_matched_recharge');m.Params.Threads=threads;m.Params.TimeLimit=seconds;m.Params.Seed=seed;m.Params.MIPGap=.0001;m.Params.LogFile=str(P/f'{arm}.gurobi.log');m.Params.NumericFocus=2
 conn=[];allrecords=[];obj=gp.LinExpr()
 for ri,r in enumerate(routes):
  first=r['actions'][0];soc=m.addVar(lb=C-first['deadhead_kwh'],ub=C-first['deadhead_kwh']);records=[]
  for k,a in enumerate(r['actions'][1:]):
   t=TR[a['from_trip']];aftertrip=m.addVar(lb=reserve,ub=C);m.addConstr(aftertrip==soc-t['energy'])
   if a['kind']=='direct':
    soc=m.addVar(lb=reserve,ub=C);m.addConstr(soc==aftertrip-a['deadhead_kwh']-a['idle_kwh']);records.append(dict(trip=a['from_trip'],action=a,before=aftertrip,after=soc));continue
   arrival=a['arrival_min'];latest=a['latest_departure_min'];site=a['station'];av=latest-arrival
   y=m.addVar(vtype=gp.GRB.BINARY,name=f'y{ri}_{k}');start=m.addVar(lb=arrival,ub=latest);duration=m.addVar(lb=0,ub=av);q=m.addVar(lb=0,ub=C);before=m.addVar(lb=reserve,ub=C);charged=m.addVar(lb=reserve,ub=C);soc=m.addVar(lb=reserve,ub=C)
   m.addConstr(before==aftertrip-a['inbound_kwh']-idle*(start-arrival));m.addConstr(charged==before+q);m.addConstr(soc==charged-idle*(latest-start-duration)-a['outbound_kwh']);m.addConstr(start+duration<=latest);m.addConstr(q<=C*y);m.addConstr(duration>=3*y);m.addConstr(duration<=av*y)
   xx,yy=f.potential(pr,site);ha=m.addVar(lb=0,ub=yy[-1]);hb=m.addVar(lb=0,ub=yy[-1]);m.addGenConstrPWL(before,ha,xx,yy);m.addGenConstrPWL(charged,hb,xx,yy);m.addConstr(duration==hb-ha)
   zs=[];qs=[]
   for h in range(int(arrival//60),int(latest//60)+1):
    left=max(arrival,h*60);right=min(latest,(h+1)*60)
    if right-left<3-1e-8:continue
    z=m.addVar(vtype=gp.GRB.BINARY);qh=m.addVar(lb=0,ub=C);m.addConstr(start>=left-1600*(1-z));m.addConstr(start+duration<=right+1600*(1-z));m.addConstr(qh<=C*z);zs.append(z);qs.append(qh);obj+=f.prices[h]*qh
   m.addConstr(gp.quicksum(zs)==y);m.addConstr(gp.quicksum(qs)==q);obj+=fee*y
   rec=dict(trip=a['from_trip'],action=a,before=aftertrip,after=soc,y=y,start=start,duration=duration,q=q,charge_before=before,charge_after=charged,route_index=ri)
   records.append(rec);conn.append(rec)
  m.addConstr(soc>=target[ri]);allrecords.append(records)
 for x,y in itertools.combinations(conn,2):
  if x['route_index']==y['route_index'] or x['action']['station']!=y['action']['station'] or x['action']['station']=='PARX':continue
  xa,ya=x['action'],y['action']
  if xa['latest_departure_min']<=ya['arrival_min'] or ya['latest_departure_min']<=xa['arrival_min']:continue
  order=m.addVar(vtype=gp.GRB.BINARY);relax=1600*(2-x['y']-y['y']);m.addConstr(x['start']+x['duration']<=y['start']+1600*(1-order)+relax);m.addConstr(y['start']+y['duration']<=x['start']+1600*order+relax)
 m.setObjective(obj);m.optimize();result=dict(arm=arm,fee=fee,status=m.Status,solutions=m.SolCount,objective=m.ObjVal if m.SolCount else None,bound=m.ObjBound if abs(m.ObjBound)<1e50 else None,wall_seconds=m.Runtime,original_terminal_match=[ORIG[k]['duty_id'] for k in perm],routes=[])
 if m.SolCount:
  for ri,records in enumerate(allrecords):
   charges=[];trace=[];cost=0
   r=routes[ri];first=r['actions'][0];firsttrip=TR[r['trips'][0]];trace.extend([[firsttrip['start']-first['travel_min'],C],[firsttrip['start'],C-first['deadhead_kwh']]])
   for rec in records:
    a=rec['action'];trip=TR[rec['trip']];trace.append([trip['end'],rec['before'].X])
    if 'y' in rec and rec['y'].X>.5:
     before=rec['charge_before'].X;after=rec['charge_after'].X;start=rec['start'].X;dur=rec['duration'].X;q=rec['q'].X
     assert abs(charge_soc_after_minutes(pr,a['station'],before,dur)-after)<1e-4
     cc,pts=f.cost_profile(pr,a['station'],start,before,after);cost+=cc;trace.append([start,before]);trace.extend(pts[1:]);charges.append(dict(station=a['station'],start=start,end=start+dur,kwh=q,points=pts,cost=cc))
    end=(TR[a['next_trip']]['start'] if a['next_trip'] is not None else (a['latest_departure_min']+a['outbound_min'] if a['kind']=='charge' else trip['end']+a['travel_min']))
    trace.append([end,rec['after'].X])
   result['routes'].append(dict(source_index=ri,trips=r['trips'],charges=charges,trace=trace,starts=len(charges),electricity_cost=cost,terminal_kwh=records[-1]['after'].X,target_kwh=target[ri]))
  actual=sum(r['electricity_cost']+fee*r['starts'] for r in result['routes']);assert abs(actual-result['objective'])<1e-4
  events={}
  for r in result['routes']:
   for c in r['charges']:events.setdefault(c['station'],[]).extend([(c['start'],1),(c['end'],-1)])
  capacity={}
  for st,ev in events.items():
   cur=peak=0
   for t,v in sorted(ev,key=lambda z:(round(z[0],5),z[1])):cur+=v;peak=max(peak,cur)
   capacity[st]=peak
  assert all(n<=1 for st,n in capacity.items() if st!='PARX');result['capacity_peaks']=capacity
 result['scope']='documented18E1 physics and charger counts; fixed saved joint tripsequences and stationpaths; staticDHD, platform/FIFO/driver rules unvalidated; terminal perbus matchedoriginal floor'
 (P/f'{arm}.json').write_text(json.dumps(result,indent=2));m.write(str(P/f'{arm}.lp'));m.dispose();print(arm,result['status'],result['objective'],result.get('capacity_peaks'));return result
if __name__=='__main__':
 solve('saved_joint_fee0',0);solve('saved_joint_fee5',5)
