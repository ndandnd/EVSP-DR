"""Freeze C5 class-separated prefixes; no submissions."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=ROOT/'outputs/chain_extension_20260913/inputs/manifest.json';full=json.load(open(source));c5=full['chains']['5'];order=c5['original_addition_order']+c5['added_duty_order']
master=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv'
assert sha(master)==full['source_hashes']['Par_VehicleDetails_Updated.csv']
rows=list(csv.DictReader(open(master)));duties={d:[r for r in rows if r['VehicleTask']==d and r['Identifier']=='Regular'] for d in order}
params={'18E1':{'battery_kwh':236.44,'reserve_kwh':35.466},'18E2':{'battery_kwh':239.01,'reserve_kwh':35.8515}}
(OUT/'inputs').mkdir(exist_ok=True);previous={};groups={'18E1':[],'18E2':[]};cases=[];unions=[]
for k,d in enumerate(order[:31],1):
 g='18E1' if d.startswith('134') else '18E2';groups[g].append(d);rr=[r for duty in groups[g] for r in duties[duty]]
 rr.sort(key=lambda r:(sum(int(v)*q for v,q in zip(r['Start1'].split(':'),[60,1])),int(float(r['Ordered_Trip_ID']))))
 id=f'w5_k{k:02d}_{g}';p=OUT/'inputs'/f'{id}.csv'
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
 case={'id':id,'global_target_k':k,'group':g,'group_reference_duties':len(groups[g]),'duties':list(groups[g]),'trip_count':len(rr),'input':str(p.relative_to(OUT)),'input_sha256':sha(p),'previous_group_case':previous.get(g),'arm':'parx60','non_parx_kw':240,**params[g],'cg_wall_s':14400,'max_iters':100000,'mip_wall_s':3600,'threads':8}
 cases.append(case);previous[g]=id;unions.append({'global_target_k':k,'duties':order[:k],'component_cases':dict(previous),'trip_count':sum(len(duties[t]) for t in order[:k])})
manifest={'finding':'F4','review_item':11,'state':'prepared_not_submitted','base_execution_commit':'309d98d266ebaf6b7e99543a67f8f2be5736874a','execution_commit':'50ceb6c095a580f79f87b53bef536cac31f81963','source_manifest_sha256':sha(source),'source_master_sha256':sha(master),'chain':5,'through_k':31,'source_duty_order':order,'cases':cases,'unions':unions,'physics':{'depot':'PARX60kW','opportunity_kw':240,'battery_and_reserve':params,'full_initial_soc':True,'terminal_floor':'same15percentreserve','shared_capacity':False,'three_minute_charge_minimum':False,'setup_time':False,'nonlinear_power_curve':False,'idle_energy':False,'service_vehicle_group_mixing':False,'charger_group_compatibility_filter':False,'tariff':'hourly_prices_flat.csv','charge_start_fee':5,'master_sense':'cover','soc_step_kwh':2.5,'event_block_min':5,'max_station_wait_min':1560},'initialization':'all previous same-group pool columns with stable-ID remapping and physical replay; singleton routes for current trips; no original GIRO route injection','scope':'combined physical-constraint sensitivity; not a one-factor algorithm comparison or complete GIRO model','resources':{'partition':'default_partition','exclude':'scaglione-compute-01','cpus':8,'memory_gb':48,'wall_hours':6,'concurrency':'all eligible cases; only 2 true group dependency chains'},'algorithm_difference':'capacity-speed driver adds one exact best column/iteration; earlier baseline adds30; algorithm runtime cannot be compared as a pure physics effect'}
manifest['data_sha256']={n:sha(ROOT/'.codex-work/review-strict-chain-20260916/data'/n) for n in ['hourly_prices_flat.csv','Ref_dict.csv','par_ref_dhd.csv']}
manifest['tooling_sha256']={n:sha(OUT/n) for n in ['prepare.py','worker.py','worker.sh']}
json.dump(manifest,open(OUT/'manifest.json','w'),indent=2)
print('prepared',len(cases),'unique component solves;',unions[-1])
