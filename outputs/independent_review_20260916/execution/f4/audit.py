#!/usr/bin/env python3
"""Replay unchanged GIRO schedules; no CG, optimization, or original-artifact mutation."""
import sys, json, csv, hashlib, subprocess, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT/'pinned/src'))
from audit_giro_known_columns import build_problem,HORIZON_MIN
from run_exact_pool_mip import validate_injected_route
spec=importlib.util.spec_from_file_location('original_replay',OUT/'compare_original_giro_charging.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvread(p):return list(csv.DictReader(open(p)))
def csvwrite(p, rows):
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
master=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv'; refs=ROOT/'.codex-work/zero-fee-terminal-cg/data'
master_rows=csvread(master)
inputs=ROOT/'outputs/chain_extension_20260913/inputs/manifest.json'
manifest=json.load(open(inputs))
assert sha(master)==manifest['source_hashes'][master.name]
all_duties=sorted({v for vs in manifest['base_duty_universe'].values() for v in vs})
(OUT/'duty_inputs').mkdir(exist_ok=True)
results=[]
for duty in all_duties:
 rows=[r for r in master_rows if r['VehicleTask']==duty and r['Identifier']=='Regular']
 rows.sort(key=lambda x:int(float(x['Ordered_Trip_ID'])))
 p=OUT/'duty_inputs'/f'{duty}.csv';csvwrite(p,rows)
 routes=m.extract_original(master_rows,rows);assert len(routes)==1
 route=routes[0]
 # Graph restriction settings match baseline defaults. Station wait uses same
 # horizon-wide setting as the historical original-schedule screen.
 problem=build_problem(p.parent,p.name,max_station_to_trip_wait_min=HORIZON_MIN,reference_data_dir=refs)
 for power in [240,350]:
  source=m.recorded_activity_check(route['source_activities'],g_kwh=240,charge_kw=power,reserve_kwh=0)
  reason=validate_injected_route(problem,route,240,power,0,HORIZON_MIN,arrival_grace_min=0,rate_grace_min=0)
  results.append({'duty':duty,'trip_count':len(rows),'battery_kwh':240,'charge_kw':power,'reserve_kwh':0,'source_valid':source['valid'],'production_valid':reason is None,'both_valid':source['valid'] and reason is None,'minimum_source_soc_kwh':source['minimum_soc_kwh'],'source_terminal_soc_kwh':source['terminal_soc_kwh'],'source_violations':source['violations'],'production_first_rejection':reason,'input_sha256':sha(p)})
 print(duty,results[-2]['both_valid'],results[-2]['production_first_rejection'],flush=True)
json.dump(results,open(OUT/'duty_replay.json','w'),indent=2)
csvwrite(OUT/'duty_replay.csv',[{**r,'source_violations':json.dumps(r['source_violations'])} for r in results])
lookup={r['duty']:r for r in results if r['charge_kw']==240}
cont_path=ROOT/'outputs/chain_extension_20260913/inputs/sources/known_duty_continuous_240_240.csv'
continuous={r['duty_id']:r for r in csvread(cont_path)}
chainrows=[]
for c,chain in manifest['chains'].items():
 order=chain['original_addition_order']+chain['added_duty_order'];assert len(order)==40
 for k in range(2,41):
  duties=order[:k]; assert len(set(duties))==k
  chainrows.append({'chain':int(c),'k':k,'unchanged_giro_schedule_valid_count':sum(lookup[d]['both_valid'] for d in duties),'unchanged_recorded_activity_valid_count':sum(lookup[d]['source_valid'] for d in duties),'unchanged_production_schedule_valid_count':sum(lookup[d]['production_valid'] for d in duties),'fixed_trip_optimized_charging_reported_valid_count':sum(continuous[d]['continuous_physical_feasible_240_240']=='True' and continuous[d]['physical_replay_status']=='validated' for d in duties),'fixed_trip_certificate_reverified_this_audit':False,'duties':';'.join(duties),'unchanged_schedule_rejected_duties':';'.join(d for d in duties if not lookup[d]['both_valid'])})
csvwrite(OUT/'chain_replay_counts.csv',chainrows)
prior=csvread(OUT/'prior350_preflight.csv')
cur350={r['duty']:r for r in results if r['charge_kw']==350}
comparison=[{'duty':r['duty'],'old_source':r['source_valid']=='True','new_source':cur350[r['duty']]['source_valid'],'old_model':r['model_valid']=='True','new_model':cur350[r['duty']]['production_valid']} for r in prior]
json.dump(comparison,open(OUT/'prior350_reproduction.json','w'),indent=2)
assert all(x['old_source']==x['new_source'] and x['old_model']==x['new_model'] for x in comparison)
provenance={'finding':'F4','execution_commit':subprocess.check_output(['git','rev-parse','a0e0bb76'],cwd=ROOT,text=True).strip(),'implementation':str(Path(__file__).relative_to(ROOT)),'implementation_sha256':sha(__file__),'source_master_sha256':sha(master),'input_manifest_sha256':sha(inputs),'known_continuous_summary_sha256':sha(cont_path),'prior350_replay_reproduced':True,'prior350_pass':sum(cur350[x['duty']]['both_valid'] for x in prior),'prior350_n':len(prior),'baseline240_pass_on_same_prior40':sum(lookup[x['duty']]['both_valid'] for x in prior),'baseline240_pass_all42_variants':sum(x['both_valid'] for x in lookup.values()),'reference_hashes':{n:sha(refs/n) for n in ['Ref_dict.csv','par_ref_dhd.csv']},'source_code_hashes':{str(p.relative_to(OUT)):sha(p) for p in [OUT/'compare_original_giro_charging.py',OUT/'pinned/src/audit_giro_known_columns.py',OUT/'pinned/src/run_exact_pool_mip.py',OUT/'pinned/src/pricing_dp_og.py',OUT/'pinned/src/config.py']},'scope':'Original unmodified windows/energies. No within-window power trace observed; no rounding repair or energy clipping. Physical replay is not a 2.5kWh/5min graph representability proof. Rejection does not prove fixed trip sequence infeasible with reoptimized charging. Each chain counts its frozen duty variants; k beyond32 has input membership only, not completed CG.'}
json.dump(provenance,open(OUT/'provenance.json','w'),indent=2)
print(json.dumps({k:v for k,v in provenance.items() if k.startswith(('prior','baseline'))},indent=2))
