from pathlib import Path
import json,sys,subprocess,os
from process_support import read,save,sha,require_hash,check_code,now
b=Path(__file__).resolve().parent;m=read(b/'manifest.json');code=Path(m['code_path']);check_code(code,m['execution_commit'])
allinputs={p:h for c in m['cases'].values() for p,h in c['input_hashes'].items()}
for p,h in allinputs.items():require_hash(p,h)
for peak in ['peak08','peak12','peak18']:
 c,f=m['cases'][peak+'_cg'],m['cases'][peak+'_fixed'];assert c['input_hashes']==f['input_hashes']
 for arm in [c,f]:
  a=arm['argv'];assert a[a.index('--charge-kw')+1]=='350' and a[a.index('--reserve-kwh')+1]=='36' and a[a.index('--minimum-charge-minutes')+1]=='3' and a[a.index('--target')+1]=='280.7833253' and a[a.index('--fleet-cap')+1]=='5'
sys.path[:0]=[str(code/'src'),str(code/'tests')]
from run_terminal_energy_cg import run
from minimum_charge_pricer import MinimumChargeNetwork
from terminal_duplicate_cleanup import frontier
from validate_terminal_pool_partition import solve
from test_event_pricer_network import two_trip_problem,prices
from audit_giro_known_columns import build_problem,HORIZON_MIN
from run_exact_pool_mip import validate_injected_route
from physical_audit import audit_routes
native=subprocess.run(['/home/nc437/evsp_env/bin/python','-m','unittest','discover','-s','tests','-p','test_minimum_charge_pricer.py','-v'],cwd=code,capture_output=True,text=True);(b/'native_tests.log').write_text(native.stdout+native.stderr);assert native.returncode==0
smoke=b/'native_smoke';smoke.mkdir(exist_ok=False)
problem=two_trip_problem(145.4);net=MinimumChargeNetwork(problem,prices(),soc_step=2.5,block_min=5,g_kwh=240,charge_kw=350,reserve_kwh=36,minimum_charge_minutes=3,fixed_sequence_index=True)
cg=run(net,smoke/'cg',target=30,fleet_cap=1,cg_seconds=30,mip_seconds=2);assert cg['cg_pricing_certified'] and cg['charging_stage']['fleet']==1
rows=frontier(net,[0,1]);(smoke/'fixed').mkdir();fixed,chosen=solve(rows,problem.trips,target=30,cap=1,seconds=2,log_dir=smoke/'fixed',threads=1);assert len(chosen)==1 and fixed['exact_once_verified'];assert abs(cg['charging_stage']['cost']-fixed['charging_cost'])<1e-5
cg_audit=audit_routes(problem,read(smoke/'cg/selected_routes.json'),validate_injected_route,HORIZON_MIN);fixed_audit=audit_routes(problem,chosen,validate_injected_route,HORIZON_MIN)
save(smoke/'passed.json',dict(status='passed',cg=cg,fixed=fixed,cg_physical=cg_audit,fixed_physical=fixed_audit,scope='Synthetic two-trip validation only; not research outcome'))
example=next(iter(m['cases'].values()));problem=build_problem(Path(example['data_dir']),example['csv'],max_station_to_trip_wait_min=HORIZON_MIN);originals={}
for peak,source in m['original_comparator'].items():
 require_hash(source['path'],source['sha256']);original=read(source['path']);audit=audit_routes(problem,original['routes'],validate_injected_route,HORIZON_MIN);assert audit['buses']==5 and audit['exact_once'] and audit['aggregate_continuous_ending_kwh']>=m['terminal_target_kwh']-1e-6
 originals[peak]=dict(source_path=source['path'],source_sha256=source['sha256'],physical=audit,invoice=source['invoice'],scope='Existence of bounded-power profile within unchanged recorded windows; actual power trace unobserved. Invoice interval retained. All original positive windows>=3min; reserve36 passes.')
save(b/'original_revalidation.json',originals)
m['tooling_sha256']={name:sha(b/name) for name in ['worker.py','worker.sub','process_support.py','submit.py','physical_audit.py','validate_remote.py']};m['policy_sha256']=sha(b.parent/'SCAGLIONE_RESOURCE_POLICY.md');save(b/'manifest.json',m)
D=Path('/share/scaglione/nc437/evsp-dr')/b.name;D.mkdir(exist_ok=True);(D/'cases').mkdir(exist_ok=True)
if not (b/'cases').exists():(b/'cases').symlink_to(D/'cases')
(b/'logs').mkdir(exist_ok=True)
save(b/'validation.json',dict(status='passed',utc=now(),manifest_sha256=sha(b/'manifest.json'),cases=6,input_files_verified=len(allinputs),native_tests_sha256=sha(b/'native_tests.log'),native_smoke_sha256=sha(smoke/'passed.json'),original_revalidation_sha256=sha(b/'original_revalidation.json'),native_physics='Both optimized arms independently replayed with36kWh floor,350kW maximum and>=3min active charging. Original five-duty schedules likewise pass profile-existence checks.'))
print(json.dumps(dict(status='passed_prepared_not_submitted',cases=6,manifest_sha256=sha(b/'manifest.json'))))
