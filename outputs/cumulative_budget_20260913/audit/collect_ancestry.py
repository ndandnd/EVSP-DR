"""Read-only remote ancestry audit. Never reads route-column journals."""
from pathlib import Path
import hashlib,json,subprocess,datetime,re
BASE=Path('/home/nc437/ladder-lite/full_pool_recovery_20260912')
def sha(b):return hashlib.sha256(b).hexdigest()
nodes={};missing=[]
def visit(path):
 p=Path(path); key=str(p)
 if key in nodes:return
 try:b=p.read_bytes();d=json.loads(b)
 except (OSError,ValueError) as e:missing.append({'path':key,'error':str(e)});return
 inh=d.get('inherited_event_pool_audit') or {};net=d.get('network_metrics') or {};prov=d.get('provenance') or {}
 keep=['csv','prices_csv','soc_step','block_min','g_kwh','charge_kw','min_soc_frac','master_sense','master_backend','initial_pool','time_model','columns_per_iter','column_selection','column_diversity_weight','column_candidate_multiplier','iterations','certified_rc_optimal','final','wall_s','attempt_wall_s','peak_rss_mb','stop_reason','resume_parent']
 row={'path':key,'resolved_path':str(p.resolve()),'sha256':sha(b),'mtime':p.stat().st_mtime,'size_bytes':len(b),**{k:d.get(k) for k in keep},'provenance':prov,'network_metrics':net,'inherited_event_pool_audit':inh}
 row['k']=int(re.search(r'_k(\d+)',Path(d['csv']).name).group(1)) if re.search(r'_k(\d+)',Path(d.get('csv','')).name) else None
 cache=net.get('cache_path')
 if cache:
  mp=Path(cache+'.manifest.json')
  try:mb=mp.read_bytes();row['cache_manifest']={'path':str(mp),'sha256':sha(mb),'value':json.loads(mb)}
  except (OSError,ValueError) as e:row['cache_manifest_error']=str(e)
 if row['k'] in (5,8,10,15):
  inp=BASE/'code/data'/d['csv']
  row['input_file_verified']={'path':str(inp),'sha256':sha(inp.read_bytes())}
  mip=p.parent/'mip_result.json' if 'full_pool_recovery_20260912' in key else p.parent.parent/'mip'/(p.stem+'__1h2stage.json')
  try:
   mb=mip.read_bytes();mv=json.loads(mb)
   row['warm_mip']={'path':str(mip),'sha256':sha(mb),**{f:mv.get(f) for f in ('instance','buses','fleet_bound','fleet_proven','status_name','optimal_scope','physical_replay_validated','duplicate_trip_removal_validated','cross_route_charger_capacity_validated','runtime_s','pool_columns','source_cg_iterations','source_cg_wall_s','pricer_provenance','provenance')}}
  except (OSError,ValueError) as e:row['warm_mip_error']=str(e)
 nodes[key]=row
 parent=inh.get('source_status')
 if parent:
  visit(parent)
  row['parent_current_hash_matches_consumed']=nodes.get(parent,{}).get('sha256')==inh.get('source_status_sha256')
 # Resume state is not a second independent CG budget; wall_s is cumulative.
roots={str(c):str(BASE/'cases'/f'w{c}_k15'/'cg.json') for c in range(1,7)}
for p in roots.values():visit(p)
print(json.dumps({'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roots':roots,'nodes':nodes,'missing':missing},indent=2))
