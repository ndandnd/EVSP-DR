"""Independent route energy/timing audit for the new F6 experiment."""
from collections import Counter

def audit_routes(problem,routes,validator,horizon,*,reserve=36,maximum_kw=350):
 arcs={(u,v):(t,e) for u,aa in problem.adjacency.items() for v,t,e,_ in aa};rows=[]
 for index,r in enumerate(routes):
  reason=validator(problem,r,240,maximum_kw,reserve,horizon,arrival_grace_min=0,rate_grace_min=0)
  if reason is not None:raise ValueError(reason)
  stops=r['charging_stops'];soc=low=240.;stop=0;active=idle=0;durations=[]
  for u,v in zip(r['route_nodes'],r['route_nodes'][1:]):
   _,energy=(0,0) if u==v else arcs[(u,v)];soc-=energy;low=min(low,soc)
   if isinstance(v,int):soc-=problem.trip_energy[v];low=min(low,soc)
   elif stop<len(stops['stations']) and v==stops['stations'][stop]:
    a,z,e=(float(stops[k][stop]) for k in ['cst','cet','kwh']);assert z>a and 0<=e<=(z-a)*maximum_kw/60+1e-6
    if e>1e-9:assert z-a>=3-1e-8;active+=1;durations.append(z-a)
    else:idle+=1
    soc+=e;assert soc<=240+1e-6;stop+=1
  assert stop==len(stops['stations']) and low>=reserve-1e-6
  rows.append(dict(route_index=index,duty_id=r.get('duty_id'),minimum_soc_kwh=low,ending_soc_kwh=soc,active_charge_starts=active,idle_station_visits=idle,shortest_active_minutes=min(durations) if durations else None))
 counts=Counter(t for r in routes for t in r['trips'])
 if routes:assert set(counts)==set(problem.trips)
 return dict(physical_routes_validated=bool(routes),reserve_kwh=reserve,max_kw=maximum_kw,minimum_active_minutes=3,buses=len(routes),trip_count=len(counts),trip_occurrences=sum(counts.values()),duplicate_occurrences=sum(n-1 for n in counts.values()),exact_once=bool(routes) and all(n==1 for n in counts.values()),aggregate_continuous_ending_kwh=sum(r['ending_soc_kwh'] for r in rows),per_bus=rows)
