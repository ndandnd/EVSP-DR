#!/usr/bin/env python3
import csv,json,hashlib,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'f4/pinned/src'))
from audit_giro_known_columns import build_problem,HORIZON_MIN
from run_exact_pool_mip import validate_injected_route
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
inst=ROOT/'.codex-work/terminal-replay-compat-2424369-20260910/data/scale_ladder/instances/original_replay_eligible_20260908/Practice_Custom_DutyUnion_original_eligible_k05_20260908.csv'
refs=ROOT/'.codex-work/zero-fee-terminal-cg/data'
problem=build_problem(inst.parent,inst.name,max_station_to_trip_wait_min=HORIZON_MIN,reference_data_dir=refs)
arcs={(u,v):(t,e) for u,aa in problem.adjacency.items() for v,t,e,_ in aa}
summary=[];busrows=[];traces={};source_files={str(inst):sha(inst)}
for peak in ['peak08','peak12','peak18']:
 base=OUT/'sources/results'/peak/'fee0';comparison_path=base/'joint/comparison.json';frontier_path=base/'frontier.json';original_path=OUT/'sources/repriced_sources'/peak/'fee0/original.json'
 cp=json.load(open(comparison_path));fp=json.load(open(frontier_path));op=json.load(open(original_path))
 assert cp['cell']['instance_sha256']==sha(inst)
 assert cp['cell']['source_hashes']['original.json']==sha(original_path)
 cleanup=list((ROOT/'outputs/terminal_duplicate_cleanup_20260916/results'/f'{peak}_fresh').glob('*/selected_routes.json'));assert len(cleanup)==1
 cleanup_path=cleanup[0];jp=json.load(open(cleanup_path))
 for p in [comparison_path,frontier_path,original_path,cleanup_path]:source_files[str(p.relative_to(ROOT))]=sha(p)
 arms=[('Original GIRO, unchanged charging',op['routes']),('Fixed GIRO trips, charging optimized',fp['fixed_solution']['selected_routes']),('Fresh CG plus exact-once cleanup',jp)]
 for arm,routes in arms:
  starts=under3=0;total_terminal=0;minimum=240;energy_total=0;durations=[]
  cover=collections.Counter(t for r in routes for t in r['trips']);assert len(routes)==5 and len(cover)==62 and set(cover.values())=={1}
  for i,r in enumerate(routes):
   reason=validate_injected_route(problem,r,240,350,0,HORIZON_MIN,arrival_grace_min=0,rate_grace_min=0);assert reason is None,(peak,arm,i,reason)
   soc=low=240.;nodes=r['route_nodes'];stops=r['charging_stops'];sidx=0;trace=[{'where':'initial','soc_kwh':soc}]
   for u,v in zip(nodes,nodes[1:]):
    travel,energy=(0,0) if u==v else arcs[(u,v)]
    soc-=energy;low=min(low,soc);trace.append({'where':f'after_deadhead_to_{v}','soc_kwh':soc})
    if isinstance(v,int):soc-=problem.trip_energy[v];low=min(low,soc);trace.append({'where':f'after_trip_{v}','soc_kwh':soc})
    elif sidx<len(stops['stations']) and v==stops['stations'][sidx]:
     soc+=stops['kwh'][sidx];trace.append({'where':f'after_charge_{sidx}','soc_kwh':soc});sidx+=1
   assert sidx==len(stops['stations']);assert low>=-1e-5
   reported=r.get('continuous_terminal_energy_kwh',r.get('continuous_realization',{}).get('continuous_terminal_soc_kwh',r.get('production_model_check',{}).get('terminal_soc_kwh')))
   assert reported is not None and abs(soc-reported)<1e-5,(peak,arm,i,soc,reported)
   dd=[float(e)-float(s) for s,e,k in zip(stops['cst'],stops['cet'],stops['kwh']) if k>1e-8]
   starts+=len(dd);under3+=sum(d<3-1e-8 for d in dd);durations+=dd;total_terminal+=soc;minimum=min(minimum,low);energy_total+=sum(stops['kwh'])
   busid=r.get('duty_id',f'route{i+1}')
   row={'peak':peak,'arm':arm,'bus_or_giro_duty':busid,'trips':len(r['trips']),'positive_charging_windows':len(dd),'windows_shorter_than_3_min':sum(d<3-1e-8 for d in dd),'shortest_window_seconds':60*min(dd),'minimum_soc_kwh_production_model':low,'minimum_soc_percent':100*low/240,'ending_soc_kwh_production_model':soc,'ending_soc_percent':100*soc/240,'recorded_activity_minimum_soc_kwh_if_original':r.get('recorded_activity_check',{}).get('minimum_soc_kwh'),'recorded_activity_ending_soc_kwh_if_original':r.get('recorded_activity_check',{}).get('terminal_soc_kwh')}
   busrows.append(row);traces[f'{peak}/{arm}/{busid}']=trace
  if arm.startswith('Original'):
   cost=cp['original_giro'];lower=cost['charging_cost_lower'];upper=cost['charging_cost_upper'];exact=cost['charging_cost_exact']
  else:exact=sum(r['continuous_realized_cost']-100000 for r in routes);lower=upper=exact
  summary.append({'peak':peak,'arm':arm,'fleet':len(routes),'trips_exactly_once':len(cover),'charging_cost_lower':lower,'charging_cost_upper':upper,'charging_cost_exact':exact,'positive_charging_windows':starts,'windows_shorter_than_3_min':under3,'shortest_window_seconds':min(durations)*60,'minimum_soc_kwh_across_buses':minimum,'ending_soc_total_kwh':total_terminal,'charged_kwh':energy_total})
for name,rr in [('comparison.csv',summary),('per_bus_soc.csv',busrows)]:
 with open(OUT/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
json.dump({'summary':summary,'per_bus':busrows,'source_sha256':source_files,'script_sha256':sha(__file__),'validator_commit':'a0e0bb7681c8451e3cbbbfa06aef390026d9af4b','scope':'All3 arms physical replay with240kWh/350kW,reserve0,free perbusend,aggregateendmin280.7833253 foroptimizedarms; no sharedcapacity/minimum3min/setup. Original cost interval because within-windowpowertrace unobserved. Continuouscost has no full-model chargingoptimalityclaim.'},open(OUT/'results.json','w'),indent=2)
json.dump(traces,open(OUT/'soc_traces.json','w'),indent=2)
print(json.dumps(summary,indent=2))
