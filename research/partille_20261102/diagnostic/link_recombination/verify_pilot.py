#!/usr/bin/env python3
"""Replay a fixed blind path-cover candidate; independently check certificate, exact service and plugs."""
import argparse,collections,hashlib,json,math,pathlib,time

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',type=pathlib.Path,required=True);a.add_argument('--instance',type=pathlib.Path,required=True);a.add_argument('--out-dir',type=pathlib.Path,required=True);a.add_argument('--start-fee',type=float,choices=[5],required=True);z=a.parse_args();t0=time.monotonic()
 from run_capacity_terminal_cg import DEFAULT_PLUGS,problem_kwargs
 from audit_giro_known_columns import build_problem,HORIZON_MIN
 from config import CHARGING_STATIONS
 from expanded_path_realization import normalize_event_station_prices,realize_expanded_path,realized_costs,validate_continuous_charging_blocks
 from run_exact_pool_mip import validate_injected_route
 from station_power import StationPower
 from capacity_event_pricer import record_occupancy
 from utils_v2 import load_station_hourly_prices
 result={'schema':'blind-link-independent-final-replay-v1','validated':False,'witness_data_read':False}
 try:
  paths=json.loads((z.out_dir/'paths.json').read_text());cert=json.loads((z.out_dir/'graph_certificate.json').read_text());routes=json.loads((z.out_dir/'partial_routes.json').read_text());r0=json.loads((z.out_dir/'receipt.json').read_text())
  assert sha(z.instance)==r0['source_instance_sha256'] and sha(z.out_dir/'paths.json')==r0['paths_sha256'] and sha(z.out_dir/'graph_certificate.json')==r0['graph_certificate_sha256']
  edges={(u,v) for u,v,_ in cert['edges']};matching=cert['matching'];cl=set(cert['min_vertex_cover_left']);cr=set(cert['min_vertex_cover_right'])
  assert len({u for u,v in matching})==len(matching)==len({v for u,v in matching}) and all(tuple(e) in edges for e in matching)
  assert all(u in cl or v in cr for u,v in edges) and len(cl)+len(cr)==len(matching)==165
  assert [r['trips'] for r in routes]==paths and len(routes)==8
  problem=build_problem(z.data,str(z.instance),**problem_kwargs());prices=normalize_event_station_prices(load_station_hourly_prices(z.data/'hourly_prices_flat.csv',CHARGING_STATIONS),horizon_min=HORIZON_MIN,strict_tariff_coverage=False);power=StationPower(240,{'PARX':60})
  assert all(problem.end_min[u]<=problem.start_min[v] for u,v in edges)
  occupancy=collections.Counter();coverage=collections.Counter();terminal=actual_terminal=grid_cost=actual_cost=0;charging_sessions=0;blocks=[]
  for route in routes:
   realized,detail=realize_expanded_path(problem,route,g_kwh=240,charge_kw=power,reserve_kwh=0,soc_step=2.5,block_min=5,time_model='event')
   assert realized is not None,detail
   reason=validate_injected_route(problem,route,240,power,0,HORIZON_MIN,arrival_grace_min=0);assert reason is None,reason
   costs=realized_costs(realized,detail['mapping'],station_prices=prices)
   stored=validate_continuous_charging_blocks(route,route['continuous_realized_charging_blocks'],station_prices=prices,charge_kw=power)
   assert stored['block_schedule_sha256']==costs['continuous_realized_charging_blocks_sha256']
   assert math.isclose(route['cost'],costs['recomputed_expanded_grid_cost'],abs_tol=1e-6)
   assert math.isclose(route['continuous_realized_cost'],costs['continuous_realized_cost'],abs_tol=1e-6)
   assert math.isclose(route['terminal_energy_kwh'],detail['mapping']['expanded_grid_terminal_soc_kwh'],abs_tol=1e-6)
   occupancy.update(record_occupancy(route));coverage.update(route['trips']);terminal+=detail['mapping']['expanded_grid_terminal_soc_kwh'];actual_terminal+=detail['mapping']['continuous_terminal_soc_kwh'];grid_cost+=costs['recomputed_expanded_grid_cost']-100000;actual_cost+=costs['continuous_realized_cost']-100000;charging_sessions+=len(route['charging_stops']['stations'])
  overload=sum(DEFAULT_PLUGS[s] is not None and n>DEFAULT_PLUGS[s] for (s,m),n in occupancy.items())
  assert coverage==collections.Counter({i:1 for i in range(173)}) and overload==0 and terminal>=-1e-6
  result.update(validated=True,fleet=8,trip_count=173,exact_once=True,minimum_paths_on_observed_adjacency_dag=173-len(matching),grid_plug_overload_minutes=overload,grid_terminal_kwh=terminal,continuous_terminal_kwh=actual_terminal,expanded_grid_nonfleet_cost=grid_cost,continuous_nonfleet_cost=actual_cost,charging_sessions=charging_sessions,scope='Feasible candidate under independently replayed matching C2 chain physics; no CG/full-model pricing or charging-cost optimality certificate claimed',plugs=DEFAULT_PLUGS)
 except Exception as e:result.update(reason=str(e),error_type=type(e).__name__)
 result.update(elapsed_seconds=time.monotonic()-t0,code_sha256=sha(__file__),routes_sha256=sha(z.out_dir/'partial_routes.json'),graph_certificate_sha256=sha(z.out_dir/'graph_certificate.json'),paths_sha256=sha(z.out_dir/'paths.json'))
 (z.out_dir/'final_replay_receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2));return 0 if result['validated'] else 3
if __name__=='__main__':raise SystemExit(main())
