import hashlib,json
from pathlib import Path
w=Path('/home/nc437/ladder-lite/integer_columns_k15_20260921')
p=w/'manifest.json';m=json.loads(p.read_text())
(w/'source_preflight_manifest.json').write_text(p.read_text())
m['schema']='evsp-dr-integer-columns-replication-v1';m['track']='integer_columns_k15_20260921'
m['preregistered'].pop('seed',None);m['preregistered']['seeds']=[20260921]
m['preregistered']['seeds_per_case']=1
m['budget']={'shared_dive_wall_plus_solver_s':7200,'dive_wall_limit_s':5400,'mip_floor_s':0,'strict_end_to_end_cap':False,'mip_preparation_and_validation':'external measured overhead','graph':'verified cached prerequisite; original build recorded separately'}
m['resources']={'partition':'default_partition','cpus_per_task':8,'memory':'32G','allocation_wall':'03:00:00','requeue':True,'exclude':'scaglione-compute-01','independent_allocations':6,'array_throttle':None,'resource_reason':'3.38–4.86GB graph plus deserialization copies and larger pool/master;32G margin, review MaxRSS'}
m['preregistered'].update(fleet_cap=15,max_nodes=80,node_time_s=180,secondary_outcome='native15-bus dive cover',no_target_import='No GIRO, warm or witness routes')
m['launch_status']='conditional_on_native_k8_handoff_and_physical_gate'
m['arms']=['control','treatment']
for c in m['cases'].values():
 c['historical_pilot_budget']=c.pop('budget')
m['preregistration_sha256']=hashlib.sha256((w/'PREREGISTRATION.md').read_bytes()).hexdigest()
m['wrapper_sha256']=hashlib.sha256((w/'run.sub').read_bytes()).hexdigest()
p.write_text(json.dumps(m,indent=1))
print('Prepared conditional6 allocations; actual end-to-end overhead explicitly separate.')
