#!/usr/bin/env python3
"""Independent current rerun of continuous fixed-trip charging, preserving all witnesses."""
import sys,json,csv,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'pinned/src'))
from audit_giro_known_columns import build_problem,HORIZON_MIN
from fixed_duty_continuous_optimizer import optimize_fixed_duty_continuous
from config import CHARGING_STATIONS
from utils_v2 import load_station_hourly_prices
refs=ROOT/'.codex-work/zero-fee-terminal-cg/data';tariff=refs/'tariff_response/flat_h26.csv'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
prices=load_station_hourly_prices(tariff,CHARGING_STATIONS)
old={x['duty_id']:x for x in csv.DictReader(open(ROOT/'outputs/chain_extension_20260913/inputs/sources/known_duty_continuous_240_240.csv'))}
(OUT/'optimized_fixed_duties').mkdir(exist_ok=True)
rows=[];start=time.monotonic()
for p in sorted((OUT/'duty_inputs').glob('*.csv')):
 problem=build_problem(p.parent,p.name,max_station_to_trip_wait_min=HORIZON_MIN,reference_data_dir=refs)
 trips=sorted(problem.trips,key=lambda t:(problem.start_min[t],t))
 result=optimize_fixed_duty_continuous(problem,trips,prices,g_kwh=240,charge_kw=240,reserve_kwh=0,charge_start_cost=5,terminal_soc_policy='free',timing_mode='optimized',tariff_id='flat_h26',tariff_sha256=sha(tariff),instance_sha256=sha(p),time_limit_s=60)
 result['audit_duty_id']=p.stem;result['source_input_sha256']=sha(p)
 out=OUT/'optimized_fixed_duties'/f'{p.stem}.json'
 json.dump(result,open(out,'w'),indent=2,allow_nan=False)
 r={'duty':p.stem,'feasible':result.get('feasible'),'replay':result.get('physical_replay_status'),'certified':result.get('certificate',{}).get('certified'),'objective':result.get('objective'),'prior_objective':float(old[p.stem]['objective']),'objective_delta':None if result.get('objective') is None else result['objective']-float(old[p.stem]['objective']),'runtime_s':result.get('runtime_s'),'result_sha256':sha(out)}
 rows.append(r);print(json.dumps(r),flush=True)
json.dump({'findings':['F4'],'execution_commit':'a0e0bb7681c8451e3cbbbfa06aef390026d9af4b','script_sha256':sha(__file__),'tariff_sha256':sha(tariff),'elapsed_s':time.monotonic()-start,'results':rows,'scope':'Continuous fixed-trip charging feasibility and replay, not event-lattice representability; no station capacity, reserve0, uniform240kW, initial240kWh, terminalfree, fee5.'},open(OUT/'fixed_duty_rerun.json','w'),indent=2)
