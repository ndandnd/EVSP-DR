"""Audit the saved morning recovery and update a copy of the frozen 80-cell CSV."""
from collections import Counter
from pathlib import Path
import copy
import csv
import hashlib
import json

OUT=Path(__file__).resolve().parent
SNAP=OUT/'snapshot.json'
BEFORE=OUT.parent/'endpoint_summary'/'expansion_cells.csv'
S=json.loads(SNAP.read_text())
checks=[]
def check(condition,label):
    assert condition,label
    checks.append(label)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def close(a,b): return abs(a-b)<1e-6

before_hash=digest(BEFORE)
with BEFORE.open() as stream: rows=list(csv.DictReader(stream))
old_rows=copy.deepcopy(rows)
check(len(rows)==80,'80 frozen rows')
row=next(r for r in rows if r['cell']=='mix1_two_price_split')
files={Path(f['path']).name:f for f in S['attempts'][0]['files']}
a=files['attempt.json']['data']; d=files['recovery_design.json']['data']
r=files['summary.json']['data']; routes=files['selected_routes.json']['data']
source=S['source_summary']; post=S['original_post']; sub=S['submission']['data']
for record in [S['submission'],post,source]+list(files.values()):
    raw=(json.dumps(record['data'],indent=2)+'\n').encode()
    check(hashlib.sha256(raw).hexdigest()==record['sha256'],'embedded JSON artifact SHA reproduced '+record['path'])
expected='684886c6dd791357cab107bc5eb951f951c91b21bf8b53a2171b3073d5551aac'
check(expected==source['sha256']==a['source_sha256']==d['source_sha256']==r['source_sha256']==row['original_summary_sha256'],
      'original fresh-CG source SHA matches recovery, design, summary and frozen CSV')
check(r['source']==d['source']==source['path']==row['original_summary_path'],'original source paths agree')
check(post['sha256']==row['post_receipt_sha256'],'original failed post receipt unchanged')
check(post['data']['cleanup_returncode']==1 and post['data']['state']=='failed','original guard failure preserved')
check(sub['replaces_failed_post']==a['recovery_of_post']=='498962','recovery replaces failed post 498962')
check(sub['job_id']==a['slurm_job_id']=='520378','recovery job 520378')
check(sub['new_treatment'] is False and a['source_is_fallback'] is False,'guard-only direct-source recovery')
check(a['returncode']==0 and a['state']=='finished' and int(a['slurm_restart'])==0,'successful unrestarted attempt')
for name in ['summary.json','selected_routes.json']:
    check(a['output_hashes'][name]==files[name]['sha256'],'output hash binding '+name)
check(a['selected_source_sha256']==d['selected_source_sha256']==r['selected_source_sha256'],
      'selected original route-set hash binding')
for key in ['instance_sha256','tariff_sha256','execution_commit']:
    check(a[key]==row[key]==post['data'][key],'unchanged '+key)
check(a['input_hashes']==source['data']['input_hashes'],'source input hashes unchanged')
check(a['physics']==source['data']['physics'],'source physics unchanged')
check(a['recovery_guard']==d['recovery_guard']==11 and d['original_guard']==10,'only recorded guard raised 10 to 11')
check(a['original_module_sha256']==d['original_module_sha256'] and a['patched_module_sha256']==d['patched_module_sha256'],
      'guard module lineage hashes agree')
check(a['wrapper_sha256']==sub['files']['cleanup_case11.py'] and a['worker_sha256']==sub['files']['recovery_worker.py'],
      'submission wrapper and worker hashes agree')
check(d['duplicated_trip_counts_by_route']==[10,1,0,2,11],'per-route duplicate counts')
check(sum(2**n for n in d['duplicated_trip_counts_by_route'])==d['subsequence_upper_count']==r['generated_sequences']==3079,
      'same exhaustive 3079-subsequence enumeration')
check(len(routes)==r['fleet']==5 and r['fleet_status']==2 and close(r['fleet_bound'],5),'five selected routes and finite-pool fleet proof')
check(r['charging_status']==2 and close(r['charging_cost'],r['charging_bound']),'finite cleanup-pool charging proof')
check(r['exact_once_verified'] and r['individual_replay_verified'] and not r['shared_capacity_validated'],
      'reported exact-once/replay pass; capacity remains unvalidated')
trips=[t for route in routes for t in route['trips']]
coverage=Counter(trips)
check(set(coverage)==set(range(111)) and all(n==1 for n in coverage.values()),'all 111 source trip indices exactly once')
check(d['unique_trips']==len(coverage)==len(trips),'design trip universe equals selected coverage')
route_checks=[]
for i,route in enumerate(routes,1):
    cr=route['continuous_realization']; pr=route['physical_realization']
    check([n for n in route['route_nodes'] if isinstance(n,int)]==route['trips'],f'route {i} integer node sequence matches trips')
    for field,key in [('trips','trip_sequence_sha256'),('route_nodes','route_nodes_sha256')]:
        check(hashlib.sha256(canonical(route[field])).hexdigest()==cr[key],f'route {i} canonical {field} hash')
    block_bytes=canonical(route['continuous_realized_charging_blocks'])
    check(hashlib.sha256(block_bytes).hexdigest()==pr['continuous_realized_charging_blocks_sha256'],f'route {i} block hash')
    check(len(block_bytes)==route['continuous_realized_charging_blocks_json_bytes'],f'route {i} block byte length')
    check(pr['status']=='valid_event_time_realized',f'route {i} saved replay status')
    check(not route['continuous_cost_pricing_certified'] and not pr['continuous_cost_pricing_certified'],
          f'route {i} no continuous-cost pricing certificate')
    check(close(route['terminal_energy_kwh'],cr['expanded_grid_terminal_soc_kwh']),f'route {i} grid terminal field')
    check(close(route['continuous_terminal_energy_kwh'],cr['continuous_terminal_soc_kwh']),f'route {i} continuous terminal field')
    grid_cost=route['expanded_grid_cost']-100000
    continuous_cost=route['continuous_realized_cost']-100000
    blocks_cost=sum(b['realized_kwh']*b['price_per_kwh'] for b in route['continuous_realized_charging_blocks'])
    grid_blocks_cost=sum(b['expanded_grid_kwh']*b['price_per_kwh'] for b in route['continuous_realized_charging_blocks'])
    check(close(continuous_cost,blocks_cost),f'route {i} independently summed continuous charging cost')
    check(close(grid_cost,grid_blocks_cost),f'route {i} independently summed grid charging cost')
    route_checks.append(dict(route=i,trips=len(route['trips']),grid_charging_cost=grid_cost,continuous_charging_cost=continuous_cost,
      grid_terminal_kwh=route['terminal_energy_kwh'],continuous_terminal_kwh=route['continuous_terminal_energy_kwh']))
grid_cost=sum(x['grid_charging_cost'] for x in route_checks)
continuous_cost=sum(x['continuous_charging_cost'] for x in route_checks)
grid_terminal=sum(x['grid_terminal_kwh'] for x in route_checks)
continuous_terminal=sum(x['continuous_terminal_kwh'] for x in route_checks)
check(close(grid_cost,r['charging_cost']),'five-route grid-cost sum matches summary')
check(close(continuous_cost,r['continuous_charging_cost']),'five-route continuous-cost sum matches summary')
check(close(grid_terminal,r['terminal_kwh']),'summary terminal_kwh is grid aggregate')
check(grid_terminal>=a['target_kwh'] and continuous_terminal>=a['target_kwh'],'both energy accountings meet aggregate floor')

accounting=[line.split('|') for line in S['accounting']['stdout'].splitlines()]
job=next(j for j in accounting if j[0]=='520378')
check(job[2:5]==['COMPLETED','0:0','00:45:57'] and job[7]=='0','scheduler completion elapsed and no restarts')
check(job[6]=='2026-09-26T05:08:07','scheduler end 05:08:07 EDT')
check(S['queue']['returncode']==0 and not S['queue']['stdout'].strip(),'captured user queue empty')
scoped=[j for j in accounting if j[1].startswith(('sx_','st_'))]
check(all(j[2] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY') for j in scoped),'all scoped accounting jobs terminal')

for cell in rows:
    cell['effective_cleanup_state']='finished'
    cell['effective_cleanup_job_id']=cell['post_job_id']
    cell['recovery_attempt_path']='';cell['recovery_attempt_sha256']=''
    cell['cleanup_grid_terminal_kwh']='';cell['cleanup_continuous_terminal_kwh']=''
    cell['original_failed_post_preserved']=''
row.update(cleanup_returncode=0,cleanup_summary_present=True,cleanup_fleet_status='OPTIMAL',cleanup_fleet=r['fleet'],
 cleanup_finite_pool_fleet_bound=r['fleet_bound'],cleanup_charging_status='OPTIMAL',cleanup_charging_grid_cost=r['charging_cost'],
 cleanup_charging_grid_bound=r['charging_bound'],cleanup_continuous_cost=r['continuous_charging_cost'],
 cleanup_exact_once_verified=True,cleanup_individual_replay_verified=True,cleanup_shared_capacity_validated=False,
 cleanup_proof_scope=r['proof_scope'],cleanup_input_duplicate_trip_count=r['duplicate_trip_count'],
 cleanup_generated_sequences=r['generated_sequences'],cleanup_summary_path=files['summary.json']['path'],
 cleanup_summary_sha256=files['summary.json']['sha256'],attempt_receipt_count=int(row['attempt_receipt_count'])+1,
 effective_cleanup_job_id='520378',recovery_attempt_path=files['attempt.json']['path'],
 recovery_attempt_sha256=files['attempt.json']['sha256'],original_failed_post_preserved='498962',
 cleanup_grid_terminal_kwh=grid_terminal,cleanup_continuous_terminal_kwh=continuous_terminal)
for old,new in zip(old_rows,rows):
    if old['cell']!=row['cell']:
        check(all(new[k]==v for k,v in old.items()),'unchanged prior endpoint '+old['cell'])
check(sum(str(x['cleanup_summary_present'])=='True' for x in rows)==80,'80 effective cleanup summaries')
check(Counter(x['post_branch'] for x in rows)=={'cleanup':55,'fallback_then_cleanup':25},'55 direct and 25 fallback')
check(Counter(x['cleanup_charging_status'] for x in rows)=={'OPTIMAL':61,'TIME_LIMIT':19},'61 optimal and 19 timed charging stages')
check(sum(x['original_cg_pricing_certified']=='True' for x in rows)==61,'original CG certificate count unchanged 61')
with (OUT/'expansion_cells.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
check(digest(BEFORE)==before_hash,'frozen 04:12 CSV unchanged')
verification=dict(snapshot_utc=S['collected_utc'],snapshot_sha256=digest(SNAP),frozen_csv_sha256=before_hash,
 reducer_sha256=digest(Path(__file__)),updated_csv_sha256=digest(OUT/'expansion_cells.csv'),
 passed=True,checks_passed=len(checks),checks=checks,recovery_job='520378',source_summary_sha256=expected,
 summary_sha256=files['summary.json']['sha256'],selected_routes_sha256=files['selected_routes.json']['sha256'],
 source_hash_scope='All seven embedded artifact hashes reproduced from indent-2 JSON plus final newline; source/output cross-bindings and canonical route/block hashes independently checked. Absent original-selected and repair-route payloads are not rehashed.',
 recovery_route_checks=route_checks,trip_count=len(trips),unique_trip_count=len(coverage),grid_charging_cost=grid_cost,
 continuous_charging_cost=continuous_cost,grid_terminal_kwh=grid_terminal,continuous_terminal_kwh=continuous_terminal,
 aggregate_floor_kwh=a['target_kwh'],expansion_cleanups=80,direct=55,fallback=25,charging_optimal=61,
 charging_time_limit=19,original_cg_certified=61,shared_capacity_validated=False,
 scheduler_recovery=job,scoped_queue_empty=True,scoped_terminal_jobs=len(scoped),
 heartbeat_status='PAUSED',heartbeat_evidence='Root task reports automation tool confirmation; not independently contained in this local snapshot.',
 scope='Finite generated cleanup pool proof; no new CG certificate, no continuous-cost proof, no full operational/GIRO dispatch claim. MIX family and absent shared-capacity caveats remain.')
(OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps({k:verification[k] for k in ['passed','checks_passed','trip_count','grid_charging_cost','continuous_charging_cost','grid_terminal_kwh','continuous_terminal_kwh','expansion_cleanups','charging_optimal','charging_time_limit']}))
