"""Fresh selected-route replay on the predeclared grid; covering and partition are separate outcomes."""
import argparse, collections, hashlib, json, math
from pathlib import Path
from run_capacity_terminal_cg import DEFAULT_PLUGS, problem_kwargs
from audit_giro_known_columns import HORIZON_MIN, build_problem
from config import CHARGING_STATIONS, BUS_COST_KX, CHARGE_START_COST
from expanded_path_realization import normalize_event_station_prices, realize_expanded_path, realized_costs, validate_continuous_charging_blocks
from run_exact_pool_mip import validate_injected_route
from station_power import StationPower
from utils_v2 import base_station_name,load_station_hourly_prices

def require(value,msg):
    if not value: raise ValueError(msg)

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else stream_digest(f)

def stream_digest(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def close(a,b):return math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-12)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('run','output'):p.add_argument('--'+x,type=Path,required=True)
    p.add_argument('--soc-step',type=float,choices=[2.5,5.],required=True);p.add_argument('--start-fee',type=float,choices=[5.],required=True)
    p.add_argument('--expected-input-hashes', type=Path, required=True); a=p.parse_args()
    s=json.loads((a.run/'summary.json').read_text());ph=s['physics']; expected_physics={'soc_step':a.soc_step,'time_step':5,'station_kw':{'PARX':60},'initial_kwh':240,'battery_kwh':240,'charge_kw':240,'reserve_kwh':0,'charge_start_fee':5,'coverage':'cover','shared_capacity':True,'non_strict_tariff':True,'plug_rows':'1-minute conservative, lazy','bus_cost_kx':100000,'max_trip2trip_min':57,'max_station_to_trip_wait_min':1560}
    require(ph==expected_physics and s['plugs']==DEFAULT_PLUGS and s['terminal_target_kwh']==0 and s['fleet_cap']==12,'physics mismatch');require(CHARGE_START_COST==5 and not s['seed_routes'],'fee or seed mismatch')
    source=s['inputs'];data=Path(source['data_dir']);instance=Path(source['instance']);tariff=Path(source['tariff']); digests={Path(k).name:sha(Path(k)) for k in s['input_hashes']}
    require(digests==json.loads(a.expected_input_hashes.read_text())=={Path(k).name:v for k,v in s['input_hashes'].items()},'input hashes mismatch')
    out={'summary_sha256':sha(a.run/'summary.json'),'soc_step':a.soc_step,'inputs_sha256':digests,'cg_pricing_certified':s['cg_pricing_certified'],'cg_stop':s['cg_stop'],'pricing_scope':'conservative weighted covering model on stated SOC grid','selection_state':s['selection_state'],'root_columns':s['columns'],'final_pool_rows':0}
    selection=a.run/'selected_routes.json';selected=json.loads(selection.read_text()) if selection.exists() else [];require(bool(selected)==(s['selection_state']=='selected'),'selection state mismatch'); target_lines={(json.dumps(r)+'\n').encode() for r in selected};require(len(target_lines)==len(selected),'duplicate selected record');found=set();prefix=hashlib.sha256();full=hashlib.sha256()
    with (a.run/'columns.jsonl').open('rb') as f:
        for i,line in enumerate(f,1):
            full.update(line);out['final_pool_rows']=i
            if i<=s['columns']: prefix.update(line)
            if line in target_lines:found.add(line)
    require(found==target_lines and prefix.hexdigest()==s['columns_sha256'],'pool identity mismatch');require(out['final_pool_rows']==s.get('dive',{}).get('columns_after_dive',s['columns']),'post-dive length mismatch');out.update(final_pool_sha256=full.hexdigest(),root_prefix_sha256=prefix.hexdigest())
    if selected:
        problem=build_problem(data,str(instance),**problem_kwargs());prices=normalize_event_station_prices(load_station_hourly_prices(tariff,CHARGING_STATIONS),horizon_min=HORIZON_MIN,strict_tariff_coverage=False);power=StationPower(240,{'PARX':60});loads=collections.Counter();coverage=collections.Counter();cost=actual=end=0.
        for r in selected:
            real,detail=realize_expanded_path(problem,r,g_kwh=240,charge_kw=power,reserve_kwh=0,soc_step=a.soc_step,block_min=5,time_model='event');require(real is not None,'grid replay failed');require(validate_injected_route(problem,r,240,power,0,HORIZON_MIN,arrival_grace_min=0) is None,'continuous replay failed')
            m=detail['mapping']; c=realized_costs(real,m,station_prices=prices);b=validate_continuous_charging_blocks(r,r['continuous_realized_charging_blocks'],station_prices=prices,charge_kw=power)
            require(b['block_schedule_sha256']==c['continuous_realized_charging_blocks_sha256'] and close(r['cost'],c['recomputed_expanded_grid_cost']) and close(r['continuous_realized_cost'],c['continuous_realized_cost']) and close(r['terminal_energy_kwh'],m['expanded_grid_terminal_soc_kwh']),'energy or billed schedule mismatch')
            cost+=c['recomputed_expanded_grid_cost']-BUS_COST_KX;actual+=c['continuous_realized_cost']-BUS_COST_KX;end+=m['expanded_grid_terminal_soc_kwh'];coverage.update(r['trips']);require(len(set(r['trips']))==len(r['trips']),'repeated trip in duty')
            stops=r['expanded_grid_charging_stops'];local=collections.Counter()
            for name,begin,finish in zip(stops['stations'],stops['cst'],stops['cet']):
                station=base_station_name(name);require(station in DEFAULT_PLUGS,'unknown station')
                if DEFAULT_PLUGS[station] is not None:
                    for minute in range(math.floor(begin+1e-9),math.ceil(finish-1e-9)):local[station,minute]+=1
            require(local=={(station,minute):n for station,minute,n in r['capacity_occupancy']},'saved plug incidences differ');loads.update(local)
        require(set(coverage)==set(problem.trips) and all(v<=DEFAULT_PLUGS[k[0]] for k,v in loads.items()) and close(cost,s['charging_stage']['cost']) and close(actual,s['selected_continuous_charging_cost']) and close(end,s['selected_terminal_kwh']) and len(selected)==s['charging_stage']['fleet']<=12,'selection mismatches pool output');extra=sum(v-1 for v in coverage.values());out.update(selected_sha256=sha(selection),fleet=len(selected),covers_all=True,extra_trip_visits=extra,exactly_once=extra==0,shared_plug_overloads=0,grid_nonfleet_cost=cost,continuous_nonfleet_cost=actual,grid_terminal_energy_kwh=end,per_route_replayed=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
