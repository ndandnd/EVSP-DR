"""Read-only continuous replay and service/empty-traversal assignment overlay.
Run under the execution environment on Unicorn; request JSON comes on stdin.
No optimization, cache rebuild, or remote write is performed.
"""
import sys,json,hashlib,time,collections,datetime
from pathlib import Path
CODE=Path('/home/nc437/ladder-lite/execution/871d057e1067411f09581e37d78f7c1ca43f68bb')
sys.path.insert(0,str(CODE/'src'))
from audit_giro_known_columns import build_problem,HORIZON_MIN
from run_exact_pool_mip import validate_injected_route

def sha(b):return hashlib.sha256(b).hexdigest()
def csha(x):return sha(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
expected={'run_exact_pool_mip.py':'bcb5a6b76040ff6ddfa932433d296a1f0f72207b28cbba738b1b4dd39f1eaac7','audit_giro_known_columns.py':'690b66ff3047894d5a142caddf7604d8df153b86838aeadac40859f5f5b6762f'}
for f,h in expected.items():assert sha((CODE/'src'/f).read_bytes())==h
requests=json.loads(sys.stdin.read());results=[];last_cid=None;problem=None
# Both original and longer searches of each case share one graph build.
requests.sort(key=lambda q:(q['case_id'],q['cohort']))
for q in requests:
 started=time.monotonic();p=Path(q['path']);raw=p.read_bytes();assert sha(raw)==q['sha256'];r=json.loads(raw);phys=r['physics']
 assert phys['g_kwh']==phys['charge_kw']==240 and phys['min_soc_frac']==0
 data=Path(q['cg_path']).parents[2]/'code/data'
 # Manifest case input path is the source used by this campaign.
 manifest=json.loads((data.parents[1]/'manifest.json').read_text())
 case=manifest['cases'][q['case_id']];csv_name=case['csv'];inp=data/csv_name
 assert sha(inp.read_bytes())==q['input_sha256']==r['physical_pool_audit']['input_hashes']['instance_sha256']
 if last_cid!=q['case_id']:
  problem=build_problem(data,csv_name,reference_data_dir=data)
  arc={(u,v):(travel,dh) for u,arcs in problem.adjacency.items() for v,travel,dh,_ in arcs}
  last_cid=q['case_id']
 original=r['selected_routes'];physical_hash_before=csha(original);seen=set();events=[];rr=[];failures=[]
 for i,route in enumerate(original):
  reason=validate_injected_route(problem,route,240,240,0,HORIZON_MIN,arc_map=arc)
  if reason is not None:failures.append(dict(route=i,reason=reason))
  assert route['trips']==[n for n in route['route_nodes'] if isinstance(n,int)]
  served=[];empty=[]
  for pos,node in enumerate(route['route_nodes']):
   if not isinstance(node,int):continue
   assigned=node not in seen;seen.add(node)
   (served if assigned else empty).append(node)
   events.append(dict(route_index=i,node_position=pos,local_trip_id=node,ordered_trip_id=q['ordered_ids'][str(node)],role='service' if assigned else 'empty_drive_same_trip_path',start_min=problem.start_min[node],end_min=problem.end_min[node],energy_kwh=problem.trip_energy[node],physical_node_retained=True))
  rr.append(dict(route_index=i,service_trip_ids=served,empty_drive_trip_ids=empty,continuous_replay_valid=reason is None,replay_reason=reason,original_physical_route_sha256=csha(route),dispatch_physical_route_sha256=csha(route),charging_stops_sha256=csha(route.get('charging_stops')),charging_blocks_sha256=csha(route.get('continuous_realized_charging_blocks')),grid_cost=route.get('expanded_grid_cost',route['cost']),continuous_cost=route.get('continuous_realized_cost'),recorded_terminal_soc_kwh=route.get('physical_realization',{}).get('continuous_terminal_soc_kwh')))
 assert seen==set(problem.trips)
 services=collections.Counter(e['local_trip_id'] for e in events if e['role']=='service');assert set(services)==seen and set(services.values())=={1}
 assert csha(original)==physical_hash_before
 results.append(dict(cohort=q['cohort'],case_id=q['case_id'],target=q['target'],buses=r['buses'],source=str(p),source_sha256=q['sha256'],input_sha256=q['input_sha256'],selected_route_set_sha256=r['selected_route_set_sha256'],input_trips=len(seen),service_occurrences=len(services),empty_drive_occurrences=sum(e['role']!='service' for e in events),continuous_replay_valid=not failures,failures=failures,physical_routes_unchanged=True,physical_routes_sha256=physical_hash_before,grid_total_cost=sum(v['grid_cost'] for v in rr),continuous_total_cost=sum(v['continuous_cost'] for v in rr),cost_preserved_by_identical_physical_schedule=True,shared_capacity_validated=False,finite_pool_partition_membership_verified=False,routes=rr,dispatch_ledger=events,replay_seconds=time.monotonic()-started))
 print(q['case_id']+' '+q['cohort']+' replay='+str(not failures),file=sys.stderr,flush=True)
print(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),execution_commit='871d057e1067411f09581e37d78f7c1ca43f68bb',module_sha256=expected,horizon_min=HORIZON_MIN,results=results)))
