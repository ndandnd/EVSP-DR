import sys,time,json,hashlib,resource,random
from pathlib import Path
code=Path(sys.argv[1]);mode=sys.argv[2];out=Path(sys.argv[3]);sys.path.insert(0,str(code/'src'))
from audit_giro_known_columns import build_problem,HORIZON_MIN,STATIONS
from utils_v2 import base_station_name,load_station_hourly_prices
from event_pricer_network import EventExpandedNetwork
from run_exact_pool_mip import validate_injected_route
inp=Path(sys.argv[4])
p=build_problem(inp.parent,inp.name,reference_data_dir=code/'data',max_station_to_trip_wait_min=1560)
prices=load_station_hourly_prices(code/'data/hourly_prices_flat.csv',sorted({base_station_name(s) for s in STATIONS}))
t=time.perf_counter();n=EventExpandedNetwork(p,prices,soc_step=2.5,block_min=5,g_kwh=239.01,charge_kw=240,reserve_kwh=35.8515,station_charge_kw={'PARX':60},arc_mode=mode,strict_tariff_coverage=False);build=time.perf_counter()-t
results=[];rng=random.Random(20260921)
for i,alpha in enumerate([{t:100000 for t in p.trips},{t:0 for t in p.trips}]+[{t:rng.uniform(50000,150000) for t in p.trips} for _ in range(3)]):
 t=time.perf_counter();r=n.min_reduced_cost_route(alpha);wall=time.perf_counter()-t
 reason=validate_injected_route(p,r['_event_record'],239.01,240,35.8515,HORIZON_MIN,arrival_grace_min=0,station_charge_kw={'PARX':60});assert reason is None,reason
 results.append({'index':i,'rc':r['rc'],'trips':r['trips'],'route':r['_event_record'],'pricing_s':wall,'physical_replay_pass':True})
out.write_text(json.dumps({'mode':mode,'build_s':build,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'input_sha256':hashlib.sha256(inp.read_bytes()).hexdigest(),'metrics':n.metrics(),'pricing':results},indent=2))
