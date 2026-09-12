#!/usr/bin/env python3
"""Minimal baseline-only station-power versus tariff-record regression fixture."""
import benchmark as b
import json,math
from pathlib import Path
station=b.ep.STATIONS[0]; base=b.base_station_name(station); output=[]
for station_kw in [60.,240.]:
 for tariff in ['variable','flat']:
  for dual_kind in ['one_row','minute50_to99']:
   prices=b.prices()
   if tariff=='flat': prices={s:{h:.1 for h in range(27)} for s in prices}
   args=dict(soc_step=15,block_min=5,g_kwh=240.,charge_kw=240.,reserve_kwh=0.,arc_mode='explicit',station_charge_kw={station:station_kw})
   net=b.ep.EventExpandedNetwork(b.fixture(),prices,**args)
   alpha={0:100000.,1:100000.}
   duals={(base,60):-123.4567} if dual_kind=='one_row' else {(base,i):-100. for i in range(50,100)}
   route=net.min_reduced_cost_route(alpha,capacity_duals=duals,capacity_sites={base})
   rec=route['_event_record']; node=net.SINK; actions=[]; rowset=set(); adjusted_path_cost=0.
   while node:
    source,selected=route['_parent'][node]
    matches=[(cost,action) for target,cost,dual,action in net.out[source] if target==node and {k:v for k,v in action.items() if k not in ('cst','cet')}=={k:v for k,v in selected.items() if k not in ('cst','cet')}]
    assert len(matches)==1
    static_cost,static=matches[0]
    adjusted,chosen=net._capacity_adjusted_arc(static_cost,static,duals,{base},1)
    assert chosen==selected
    adjusted_path_cost+=adjusted
    if selected.get('kind')=='charge':
     rows=b.ep.conservative_capacity_rows(selected,sites={base}); rowset.update(rows)
     actions.append(dict(static_arc_cost=static_cost,static_action=static,selected_action=selected,station_kw=net._charge_power(selected['station']),occupied_master_rows=sorted(rows),selected_adjusted_arc_cost=adjusted))
    node=source
   replay=rec['cost']-sum(alpha[t] for t in rec['trips'])-sum(duals.get(r,0.) for r in rowset)
   output.append(dict(station_kw=station_kw,global_kw=240.,tariff=tariff,dual_kind=dual_kind,alpha=alpha,capacity_duals=[[s,m,v] for (s,m),v in sorted(duals.items())],rc=route['rc'],adjusted_path_cost=adjusted_path_cost,record_cost=rec['cost'],record_cost_semantics=rec['cost_semantics'],expanded_grid_cost=rec['expanded_grid_cost'],continuous_realized_cost=rec['continuous_realized_cost'],independent_rc=replay,residual=route['rc']-replay,capacity_driver_check_pass=math.isclose(route['rc'],replay,abs_tol=1e-5),actions=actions,returned_tariff_blocks=rec['continuous_realized_charging_blocks']))
result=dict(source_commit=b.PIN,script_sha256=b.sha(__file__),cases=output)
Path(__file__).with_name('record_mismatch.json').write_text(json.dumps(result,indent=2)+'\n')
for r in output: print({k:r[k] for k in ('station_kw','tariff','dual_kind','residual','capacity_driver_check_pass')})
