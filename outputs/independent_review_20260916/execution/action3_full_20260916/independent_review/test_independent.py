"""Independent, bounded math-equivalence checks; no scheduler or production IO."""
from pathlib import Path
import sys,json,random,itertools,dataclasses,hashlib,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[5]
CODE=ROOT/'.codex-work/action3-full-20260916'
sys.path.insert(0,str(CODE/'src'))
import event_pricer_network as e
from audit_giro_known_columns import ProblemData,STATIONS
from utils_v2 import base_station_name
from types import SimpleNamespace
from run_capacity_speed_event_cg import build_network
rng=random.Random(20260916);records=[];shortcut=e._best_charge_window
for i in range(600):
 arrival=rng.uniform(0,1250); energy=rng.uniform(.01,240);power=rng.choice([60,240]);deadline=arrival+rng.uniform(0,260);price=rng.choice([-.05,0,.0992,.13]);args=dict(event_times={'PARX_1':tuple(range(0,1561,5))},station_prices={'PARX':{h:price for h in range(27)}},charge_kw=power)
 options=e._charge_window_options('PARX_1',arrival,deadline,energy,**args);actual=shortcut('PARX_1',arrival,deadline,energy,**args)
 assert (actual is None)==(not options)
 if actual:assert abs(actual[0]-min(options)[0])<1e-8
records.append({'test':'flat_window_reference_equivalence','cases':600,'passed':True})
# Each station offers a different constant tariff. There is no shared capacity.
trips=(0,1,2);start={0:60.,1:130.,2:220.};end={0:70.,1:140.,2:230.};energy={0:150.,1:100.,2:80.}
adjacency={'PARX_0':[(t,0.,0.,'depot_trip') for t in trips]}
for t in trips:
 adjacency[t]=[(u,0.,0.,'trip_trip') for u in trips if u>t]+[('PARX_0',0.,0.,'trip_depot')]+[(s,2.,1.,'trip_station') for s in ['PARX_1','2190L_0']]
for s in ['PARX_1','2190L_0']:adjacency[s]=[(t,2.,1.,'station_trip') for t in trips]+[('PARX_0',2.,1.,'station_depot')]
p=ProblemData(pd.DataFrame(),trips,adjacency,start,end,energy)
prices={s:{h:(.10 if base_station_name(s)=='PARX' else .12) for h in range(27)} for s in {base_station_name(x) for x in STATIONS}}
# Reference enumeration only changes charge-window optimization, not graph code.
def reference(*args,**kwargs):
 options=e._charge_window_options(*args,**kwargs)
 return min(options) if options else None
count=0;duals_count=0
for battery,reserve,arm in [(240,0,'baseline'),(240,0,'parx60'),(240,36,'baseline'),(236.44,0,'baseline'),(239.01,0,'baseline')]:
 a=SimpleNamespace(soc_step=2.5,block_min=5,battery_kwh=battery,non_parx_kw=240,reserve_kwh=reserve,arm=arm,network_arc_mode='explicit')
 e._best_charge_window=reference;old=build_network(a,p,prices)
 e._best_charge_window=shortcut;new=build_network(a,p,prices);a.network_arc_mode='lazy';compact=build_network(a,p,prices)
 for size in range(1,4):
  for seq in itertools.combinations(trips,size):
   rows=[n.fixed_sequence_record(seq) for n in [old,new,compact]]
   assert len({x is None for x in rows})==1,(battery,reserve,arm,seq)
   if rows[0]:
    assert max(r['cost'] for r in rows)-min(r['cost'] for r in rows)<1e-7
   count+=1
 for j in range(15):
  alpha={t:rng.uniform(-10000,200000) for t in trips}
  for objective in ['combined-cost','fleet-only','charging-cost']:
   route_dual=123.4 if objective=='charging-cost' else 0
   out=[n.min_reduced_cost_route(alpha,objective=objective,route_dual=route_dual) for n in [old,new,compact]]
   assert len({r is None for r in out})==1
   if out[0]:assert max(r['rc'] for r in out)-min(r['rc'] for r in out)<1e-7
   duals_count+=1
 # A same-group subset's event lattice may differ; its fixed sequence costs must not.
 seq=(0,2);allow=set(seq);adj={s:[x for x in arcs if type(x[0]) is not int or x[0] in allow] for s,arcs in p.adjacency.items() if type(s) is not int or s in allow}
 subset=dataclasses.replace(p,trips=seq,adjacency=adj)
 sub=build_network(a,subset,prices);x=compact.fixed_sequence_record(seq);y=sub.fixed_sequence_record(seq)
 assert (x is None)==(y is None)
 if x:assert abs(x['cost']-y['cost'])<1e-7
records.extend([{'test':'reference_explicit_vs_shortcut_explicit_vs_compact_fixed_sequences','cases':count,'passed':True},{'test':'three_objective_reduced_cost_equivalence','cases':duals_count,'passed':True},{'test':'group_subset_event_lattice_equivalence_flat_tariff','cases':5,'passed':True}])
try:build_network(SimpleNamespace(arm='capacity',network_arc_mode='lazy'),None,None);raise AssertionError('capacity accepted')
except ValueError as exc:assert 'capacity duals' in str(exc)
records.append({'test':'compact_rejects_shared_capacity','passed':True})
print(json.dumps({'tests':records,'source_sha256':{name:hashlib.sha256((CODE/'src'/name).read_bytes()).hexdigest() for name in ['event_pricer_network.py','run_capacity_speed_event_cg.py']},'no_scheduler_calls':True},indent=2))
