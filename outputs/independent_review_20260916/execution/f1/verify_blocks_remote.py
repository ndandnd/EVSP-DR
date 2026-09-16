import sys,json,hashlib
from pathlib import Path
CODE=Path('/home/nc437/ladder-lite/execution/871d057e1067411f09581e37d78f7c1ca43f68bb');sys.path.insert(0,str(CODE/'src'))
from expanded_path_realization import validate_continuous_charging_blocks,normalize_event_station_prices
from utils_v2 import load_station_hourly_prices
from config import CHARGING_STATIONS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
requests=json.loads(sys.stdin.read());out=[]
for q in requests:
 p=Path(q['path']);assert sha(p)==q['sha256'];r=json.loads(p.read_text());data=Path(q['cg_path']).parents[2]/'code/data';h=r['physical_pool_audit']['input_hashes'];refs={}
 for f,k in [('Ref_dict.csv','reference_sha256'),('par_ref_dhd.csv','deadhead_sha256'),('hourly_prices_flat.csv','prices_sha256')]:
  refs[f]=sha(data/f);assert refs[f]==h[k]
 prices=normalize_event_station_prices(load_station_hourly_prices(data/'hourly_prices_flat.csv',CHARGING_STATIONS),horizon_min=1560)
 rows=[]
 for i,v in enumerate(r['selected_routes']):
  a=validate_continuous_charging_blocks(v,v['continuous_realized_charging_blocks'],station_prices=prices,charge_kw=240,expected_continuous_cost=v['continuous_realized_cost']);rows.append(dict(route=i,**a))
 out.append(dict(case_id=q['case_id'],cohort=q['cohort'],source_sha256=q['sha256'],reference_hashes=refs,routes=rows,charging_blocks_and_cost_verified=True))
print(json.dumps(dict(results=out,module_hashes={f:sha(CODE/'src'/f) for f in ['expanded_path_realization.py','config.py','pricing_dp_og.py','utils_v2.py']})))
