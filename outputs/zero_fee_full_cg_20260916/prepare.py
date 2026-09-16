import json,hashlib,datetime
from pathlib import Path
root=Path('/home/nc437/ladder-lite/zero_fee_full_cg_20260916')
plan=json.loads(Path('/home/nc437/ladder-lite/giro_zero_start_fee_20260913/plan.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
cases=[]
for source in plan['cells']:
 if source['fee']!=0:continue
 data=Path(source['instance']).parents[3]
 assert data.name=='data',data
 paths=[source['instance'],source['tariff_path'],str(data/'Ref_dict.csv'),str(data/'par_ref_dhd.csv')]
 hashes={p:sha(p) for p in paths}
 assert hashes[source['instance']]==source['instance_sha256']
 assert hashes[source['tariff_path']]==source['tariff_sha256']
 cases.append(dict(id=source['tariff']+'_fresh',data_dir=str(data),instance=source['instance'],tariff=source['tariff_path'],input_hashes=hashes,comparator_directory=source['frontier_dir']))
m=dict(schema='zero-fee-full-terminal-cg-v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_commit='c210187bb50a2eac2022178ea21aa39d7ff6b9b1',code='/home/nc437/ladder-lite/code_pins/zero_fee_terminal_c210187b',cases=cases,validation_job=275221,
 physics=dict(battery_kwh=240,initial_kwh=240,charging_kw=350,reserve_kwh=0,shared_capacity=False,soc_step_kwh=2.5,event_minutes=5,aggregate_terminal_kwh=280.7833253),
 objective=dict(cg='100000 * buses + electricity; start fee = 0',mip_stage1='minimize fleet',mip_stage2='minimize electricity with buses <= stage1 incumbent',fleet_cap=5,coverage='cover'),
 initialization='fresh direct singletons plus explicit Phase I; no inherited or GIRO columns',
 budgets=dict(cg_seconds=14400,mip_total_seconds=3600,mip_stage1_seconds=1800),
 resources=dict(partition='default_partition',exclude='scaglione-compute-01',cpus=8,memory='48G',walltime='08:00:00',concurrency=3),
 dependencies='native validation 275221 must pass; three tariffs independent',
 restarts='distinct attempt directory per Slurm restart; restart from scratch; preserves prior journals, no Gurobi tree recovery',
 proof='CG certificate applies only to declared event graph and weighted objective. MIP proof is finite-pool only. Physical replay separate. No continuous/full-GIRO/global charging optimality claim.')
(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps({'manifest_sha256':sha(root/'manifest.json'),'cases':[c['id'] for c in cases]}))
