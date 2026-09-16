from pathlib import Path
import sys,json,hashlib,csv
import pandas as pd
B=Path(__file__).resolve().parent;REPO=B.parents[4];CODE=REPO/'.codex-work/review-dr-mincharge-20260916';sys.path.insert(0,str(CODE/'src'))
from tariff_response_core import giro_routes_for_instance,_minutes,_station_node
from utils_v2 import load_station_hourly_prices
from expanded_path_realization import _tariff_identity
from config import CHARGING_STATIONS
H=Path('/home/nc437/ladder-lite');REMOTE=H/'review_dr_mincharge_20260916';PIN=H/'code_pins/review_dr_mincharge_67636d44'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def main():
 source=REPO/'outputs/independent_review_20260916/execution/f4/duty_replay.json';audit={x['duty']:x for x in json.loads(source.read_text()) if x['charge_kw']==240}
 master=CODE/'data/Par_VehicleDetails_Updated.csv';df=pd.read_csv(master);df['VehicleTask']=df['VehicleTask'].astype(str)
 p1=json.loads((B.parents[1]/'p1/manifest.json').read_text());cases={};original_public={};original_private={};pairs=[]
 tariffs={f'peak{h}':CODE/f'data/tariff_response/peak{h}_h26.csv' for h in ['08','12','18']};tariffs['se3']=B.parent/'tariffs/se3_20250915_h26.csv'
 for chain in range(1,7):
  old=p1['cases'][f'fresh_c{chain}_k15_seed0'];relative=old['csv'];instance=CODE/'data'/relative;assert sha(instance)==old['input_sha256']
  duties=giro_routes_for_instance(master,instance);assert len(duties)==15
  target=sum(audit[d['duty_id']]['source_terminal_soc_kwh'] for d in duties)
  assert 0<=target<=3600
  for tariff,localtariff in tariffs.items():
   pair=f'c{chain}_k15_{tariff}';remote_tariff=REMOTE/'tariffs'/localtariff.name;prices=load_station_hourly_prices(localtariff,CHARGING_STATIONS)
   events=[]
   for duty in duties:
    for index,row in df[(df.VehicleTask==duty['duty_id'])&(df.Identifier=='Recharge')].iterrows():
     start,end=_minutes(str(row.Start1)),_minutes(str(row.End1));energy=float(row['Recharge kWh']);station=_station_node(str(row.From1));cursor=start;cost=0
     while cursor<end-1e-9:
      stop=min(end,(int(cursor//60)+1)*60);cost+=energy*(stop-cursor)/(end-start)*_tariff_identity(station,cursor,prices)['price_per_kwh'];cursor=stop
     events.append(dict(duty_id=duty['duty_id'],source_row=int(index+2),station=station,start=start,end=end,kwh=energy,cost=cost))
   original=dict(active_charge_starts=len(events),energy_kwh=sum(e['kwh'] for e in events),electricity_cost=sum(e['cost'] for e in events),original_soc_ledger_target_kwh=target,interpretation='Original GIRO recorded recharge windows/kWh, assumed uniform power within window. Accounting comparator only; not assumed feasible under240kW. Common target from source-energy ledger at initial240; not a source-model terminal feasibility proof.',source_energy_ledger_sha256=sha(source),master_sha256=sha(master),tariff_sha256=sha(localtariff),events=events)
   (original_private if tariff=='se3' else original_public)[pair]=original
   for arm in ['cg','fixed']:
    cid=f'{pair}_{arm}';paths={str(PIN/'data'/relative):sha(instance),str(PIN/'data/Ref_dict.csv'):sha(CODE/'data/Ref_dict.csv'),str(PIN/'data/par_ref_dhd.csv'):sha(CODE/'data/par_ref_dhd.csv'),str(PIN/'data/Par_VehicleDetails_Updated.csv'):sha(master),str(remote_tariff):sha(localtariff)}
    common=['--data-dir',str(PIN/'data'),'--csv',relative,'--prices',str(remote_tariff),'--target',str(target),'--fleet-cap','15','--out','{out}']
    if arm=='cg':cmd=[str(PIN/'src/run_terminal_energy_cg.py'),*common,'--cg-seconds','28800','--mip-seconds','3600','--charge-kw','240','--minimum-charge-minutes','3'];wall=13*3600
    else:cmd=[str(PIN/'src/run_minimum_charge_fixed.py'),*common,'--master',str(PIN/'data/Par_VehicleDetails_Updated.csv'),'--frontier-seconds','14400','--mip-seconds','3600'];wall=9*3600
    cases[cid]=dict(id=cid,pair=pair,arm=arm,chain=chain,k=15,trip_count=len(pd.read_csv(instance)),tariff=tariff,publication='internal_only' if tariff=='se3' else 'publishable',input_hashes=paths,terminal_target_kwh=target,target_scope='common aggregate minimum derived from original source-energy ledger; physical original replay not asserted',duty_ids=[d['duty_id'] for d in duties],baseline_original_duties_valid=sum(audit[d['duty_id']]['both_valid'] for d in duties),argv=['/home/nc437/evsp_env/bin/python',*cmd],resources=dict(cpus=8,mem='32G',allocation_s=wall),watchdog_s=wall-300)
   pairs.append(dict(pair=pair,target_kwh=target,chain=chain,tariff=tariff,cg_case=pair+'_cg',fixed_case=pair+'_fixed'))
 commit=__import__('subprocess').check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
 manifest=dict(schema='evsp-review-mincharge-DR-v1',review_items=[10],findings=['F6','F7'],execution_commit=commit,code_path=str(PIN),cases=cases,pairs=pairs,source_energy_ledger_sha256=sha(source),physics=dict(battery_kwh=240,initial_soc_kwh=240,max_charge_kw=240,reserve_kwh=0,charge_start_fee=0,minimum_active_charge_minutes=3,charging_control='constant adjustable power within original graph charge window',zero_actual_energy='idle station visit, excluded active start count',shared_capacity='unconstrained',master_sense='cover',soc_step_kwh=2.5,time_step_minutes=5,terminal_energy='aggregate minimum common to both arms'),initialization=dict(cg='fresh singleton columns; phase-I artificials; no fixed/GIRO seed',fixed='original GIRO trip sequence per duty; complete same-graph terminal frontiers'),proof_scope='LP certificate for discretized graph objective only. MIP proof finite pool. Continuous realized cost reported separately. GIRO-as-is is accounting only.',time_accounting='Record total worker wall including graph construction separately from graph, CG, pricing, LP, fleet MIP, charging MIP, and fixed frontier phases.',tariffs={k:dict(file_name=v.name,sha256=sha(v),internal_only=k=='se3') for k,v in tariffs.items()},status='prepared_not_submitted')
 dump(B/'manifest.json',manifest);dump(B/'original_giro_synthetic.json',original_public);dump(B/'private_internal/original_giro_se3.json',original_private)
 print(json.dumps(dict(cases=len(cases),pairs=len(pairs),targets={x['chain']:x['target_kwh'] for x in pairs})))
if __name__=='__main__':main()
