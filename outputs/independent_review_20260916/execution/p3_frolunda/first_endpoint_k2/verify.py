"""Independent read-only F8 k2 endpoint verification from preserved native artifacts."""
from pathlib import Path
import sys,json,csv,hashlib,collections,subprocess
P=Path(__file__).resolve().parent;F=P.parent;ROOT=F.parents[3]
source=json.loads((P/'source.json').read_text());r=json.loads((P/'result.json').read_text());cg=json.loads((P/'cg.json').read_text());m=json.loads((F/'manifest.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(P/'result.json')==source['mip_result_sha256'];assert sha(P/'cg.json')==source['cg_result_sha256'];assert sha(F/'data/fdl_k02.csv')==source['input_sha256']==r['physical_pool_audit']['input_hashes']['instance_sha256']
code=ROOT/'.codex-work/review-fdl-mip-20260916';expected=m['code']['mip']['commit'];assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==expected
# Local worktree is a branch, but source module bytes must equal the submitted detached pin.
for name in ['audit_giro_known_columns.py','run_exact_pool_mip.py','config.py','pricing_dp_og.py','expanded_path_realization.py','utils_v2.py']:
 assert (code/'src'/name).read_bytes()==subprocess.check_output(['git','-C',str(code),'show',expected+':src/'+name])
sys.path.insert(0,str(code/'src'))
from audit_giro_known_columns import build_problem,HORIZON_MIN,DEPOT
from run_exact_pool_mip import validate_injected_route,charging_stop_arrivals
from expanded_path_realization import validate_continuous_charging_blocks,normalize_event_station_prices
from utils_v2 import load_station_hourly_prices
from config import CHARGING_STATIONS,CHARGE_START_COST
assert DEPOT=='KEX_0' and CHARGE_START_COST==5
assert r['physics']['g_kwh']==r['physics']['charge_kw']==240 and r['physics']['min_soc_frac']==0
assert cg['provenance']['git_commit']==m['code']['cg']['commit'] and cg['master_sense']=='cover' and cg['certified_rc_optimal'] and cg['final']['artificials']==0
assert not r['partitioning'] and r['two_stage'] and r['buses']==2 and len(r['selected_routes'])==2
for f,key in [('Ref_dict.csv','reference_sha256'),('par_ref_dhd.csv','deadhead_sha256'),('hourly_prices_flat.csv','prices_sha256')]:assert sha(F/'data'/f)==r['physical_pool_audit']['input_hashes'][key]==m['data_hashes'][f]
problem=build_problem(F/'data','fdl_k02.csv');prices=normalize_event_station_prices(load_station_hourly_prices(F/'data/hourly_prices_flat.csv',CHARGING_STATIONS),horizon_min=HORIZON_MIN);counts=collections.Counter();routes=[]
for i,route in enumerate(r['selected_routes']):
 counts.update(route['trips']);reason=validate_injected_route(problem,route,240,240,0,HORIZON_MIN,arrival_grace_min=0.0);assert reason is None,(i,reason)
 blocks=route['continuous_realized_charging_blocks'];arrivals=charging_stop_arrivals(problem,route);assert all(float(b['start_min'])>=arrivals[int(b['stop_index'])]-1e-6 for b in blocks)
 z=validate_continuous_charging_blocks(route,blocks,station_prices=prices,charge_kw=240,expected_continuous_cost=route['continuous_realized_cost'])
 routes.append(dict(route_index=i,trip_count=len(route['trips']),ordered_trip_ids=[int(problem.frame.iloc[t]['Ordered_Trip_ID']) for t in route['trips']],zero_arrival_grace_replay=True,continuous_blocks_start_after_arrival=True,charging_blocks_cost_verified=True,continuous_cost=z['continuous_realized_cost'],grid_cost=z['recomputed_expanded_grid_cost']))
assert set(counts)==set(range(38)) and set(counts.values())=={1}
# Independent fleet lower bound from two service trips that overlap in time.
conflict=next(({'trip_a':int(problem.frame.iloc[a]['Ordered_Trip_ID']),'trip_b':int(problem.frame.iloc[b]['Ordered_Trip_ID']),'a_time':[problem.start_min[a],problem.end_min[a]],'b_time':[problem.start_min[b],problem.end_min[b]]} for a in problem.trips for b in problem.trips if a<b and max(problem.start_min[a],problem.start_min[b])<min(problem.end_min[a],problem.end_min[b])),None);assert conflict
out={'finding':'F8','verdict':'VERIFIED: two reference duties/38trips recovered by two physically replay-valid buses, exact-once service','source_paths':{'cg':source['cg_result_path'],'mip':source['mip_result_path']},'source_sha256':{'cg':sha(P/'cg.json'),'mip':sha(P/'result.json')},'input_sha256':sha(F/'data/fdl_k02.csv'),'physics':r['physics'],'objective':'100000 + electricity +5perchargingstart','master_sense':'cover','coverage':{'trips':38,'occurrences':sum(counts.values()),'exactly_once':True},'routes':routes,'cg_native_claim':{'certificate':cg['certified_rc_optimal'],'iterations':cg['iterations'],'wall_s':cg['wall_s'],'final':cg['final']},'mip_native_claim':{'fleet_proven_in_pool':r['fleet_proven'],'pool_fleet_bound':r['fleet_bound']},'independent_fleet_lower_bound':{'bound':2,'proof':'Two mandatory service trips overlap strictly in time, so one vehicle cannot serve both. Together with the validated two-vehicle dispatch this proves the fleet optimum for this instance. It does not prove charging optimality.','witness':conflict},'not_checked':['Full GIRO power curves, reserve/endSOC/group constraints','Shared station capacity','Independent rerun of pricing global optimum or MIP branch-and-bound proof'],'replay_code_commit':expected}
(P/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
