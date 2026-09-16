from pathlib import Path
import json,sys,hashlib,time
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];DATA=OUT/'data';meta=json.load((OUT/'inputs.json').open());code=ROOT/'.codex-work/review-fdl-cg-20260916';sys.path.insert(0,str(code/'src'))
from audit_giro_known_columns import build_problem,DEPOT,STATIONS,HORIZON_MIN
from fixed_duty_continuous_optimizer import optimize_fixed_duty_continuous
from utils_v2 import load_station_hourly_prices
from config import CHARGING_STATIONS
assert DEPOT=='KEX_0' and set(CHARGING_STATIONS)==set(meta['chargers'])
prices=load_station_hourly_prices(DATA/'hourly_prices_flat.csv',CHARGING_STATIONS)
rows=[]
for cid,c in meta['cases'].items():
 p=build_problem(DATA,c['csv'])
 # Validate each singleton has a depot connection and nonnegative energy.
 missing=[t for t in p.trips if not any(v==t for v,*_ in p.adjacency.get(DEPOT,[]))]
 rows.append({'case':cid,'trips':len(p.trips),'adjacency_nodes':len(p.adjacency),'no_direct_depot_start_arc':missing,'stations':list(STATIONS),'max_service_energy':max(p.trip_energy.values()),'latest_trip_end':max(p.end_min.values())})
 assert not missing,(cid,missing)
# Reoptimize first duty independently as a schema and physical-graph smoke check.
cid=next(iter(meta['cases']));c=meta['cases'][cid];p=build_problem(DATA,c['csv'])
r=optimize_fixed_duty_continuous(p,sorted(p.trips,key=lambda t:p.start_min[t]),prices,g_kwh=240,charge_kw=240,reserve_kwh=0,charge_start_cost=5,terminal_soc_policy='free',timing_mode='optimized',tariff_id='flat',tariff_sha256=hashlib.sha256((DATA/'hourly_prices_flat.csv').read_bytes()).hexdigest(),instance_sha256=c['input_sha256'],time_limit_s=60)
(OUT/'k1_fixed_smoke.json').write_text(json.dumps(r,indent=2)+'\n');(OUT/'input_validation.json').write_text(json.dumps({'status':'passed','cases':rows,'k1_fixed_feasible':r.get('feasible'),'k1_replay':r.get('physical_replay_status'),'scope':'Input/graph construction with default220min station-to-trip window + firstduty physical feasibility; no eventCG endpoint'},indent=2)+'\n');print('inputvalid',len(rows),'k1',r.get('feasible'),r.get('physical_replay_status'))
