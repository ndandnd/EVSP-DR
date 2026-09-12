#!/usr/bin/env python3
"""Additional bounded full-path and master-row reduced-cost oracles."""
import benchmark as b
import json, math
from pathlib import Path
s=b.ep.STATIONS[0]; base=b.base_station_name(s)
args=dict(soc_step=15,block_min=5,g_kwh=240.,charge_kw=240.,reserve_kwh=0.,arc_mode='explicit',station_charge_kw={s:60.})
net=b.PrefixMemoNetwork(b.fixture(),b.prices(),**args)
results=[]
for duals in [{(base,60):-123.4567},{(base,i):-100. for i in range(50,100)}]:
    alpha={0:100000.,1:100000.}
    route=net.min_reduced_cost_route(alpha,capacity_duals=duals,capacity_sites={base})
    rec=route['_event_record']; stops=rec['expanded_grid_charging_stops']; rows=set()
    for station,start,end in zip(stops['stations'],stops['cst'],stops['cet']):
        rows.update(b.ep.conservative_capacity_rows(dict(kind='charge',station=station,cst=start,cet=end),sites={base}))
    replay=rec['cost']-sum(alpha[t] for t in rec['trips'])-sum(duals.get(r,0.) for r in rows)
    replay_pass=math.isclose(route['rc'],replay,rel_tol=0.,abs_tol=1e-8)
    costs=[]
    def walk(source,value):
        if source==net.SINK:
            costs.append(value); return
        for target,cost,dual,action in net.out[source]:
            adjusted,_=b.ep.EventExpandedNetwork._capacity_adjusted_arc(net,cost,action,duals,{base},1)
            walk(target,value+adjusted-(alpha.get(net.problem.trips[dual],0.) if dual>=0 else 0.))
    walk(0,0.)
    assert math.isclose(min(costs),route['rc'],rel_tol=0.,abs_tol=1e-8)
    results.append(dict(master_row_replay_pass=replay_pass,paths_enumerated=len(costs),global_rc=route['rc'],exhaustive_min=min(costs),master_row_replay_rc=replay,max_error=max(abs(min(costs)-route['rc']),abs(replay-route['rc']))))
output=dict(script_sha256=b.sha(__file__),benchmark_sha256=b.sha(b.__file__),results=results)
Path(__file__).with_name('extra_validation.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
