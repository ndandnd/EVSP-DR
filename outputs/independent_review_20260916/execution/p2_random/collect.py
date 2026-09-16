"""Run on Unicorn after deployment; write compact per-stage results to stdout.
Accounts all recorded worker attempts, including unsuccessful/preempted attempts.
Missing attempt end times remain explicit; Slurm accounting must fill those gaps.
"""
from pathlib import Path
import json,hashlib,datetime
B=Path('/home/nc437/ladder-lite/random_trip_groups_c1_20260916')
v=json.loads((B/'manifest.json').read_text());out=[];cum={'cache':0.,'cg':0.,'mip':0.};missing=0
for cid in v['warm_chains']['1']:
 c=v['cases'][cid];root=B/'cases'/cid;row={'case_id':cid,'stage_index_not_fleet_target':c['stage'],'trip_count':c['trip_count'],'giro_fleet_reference':15 if c['stage']==15 else None,'input_sha256':c['input_sha256']};attempts=[]
 for mode in ['cache','cg','mip']:
  duration=0.;incomplete=0
  for p in sorted((root/mode).glob('*/execution.json')):
   b=p.read_bytes();a=json.loads(b);d=a.get('ended_epoch',0)-a.get('started_epoch',0) if a.get('ended_epoch') else None
   if d is None:incomplete+=1;missing+=1
   else:duration+=d
   attempts.append({'mode':mode,'path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'attempt':a['attempt'],'status':a['status'],'recorded_process_elapsed_s':d})
  cum[mode]+=duration;row[mode+'_all_attempts_recorded_elapsed_s']=duration;row[mode+'_attempts_missing_end']=incomplete;row['cumulative_'+mode+'_recorded_elapsed_s']=cum[mode]
 for file,prefix in [('cg.json','cg'),('mip_result.json','mip')]:
  p=root/file
  if p.exists():
   b=p.read_bytes();r=json.loads(b);row[prefix+'_result_path']=str(p);row[prefix+'_result_sha256']=hashlib.sha256(b).hexdigest()
   if prefix=='cg':row.update(cg_solver_reported_wall_s=r.get('wall_s'),cg_stop_reason=r.get('stop_reason'),cg_pricing_certificate=r.get('certified_rc_optimal'),cg_iterations=r.get('iterations'),cg_fractional_route_weight=r.get('final',{}).get('route_weight'),cg_weighted_objective=r.get('final',{}).get('lp_obj'),inherited_pool_audit=r.get('inherited_event_pool_audit'))
   else:row.update(integer_buses=r.get('buses'),finite_pool_fleet_bound=r.get('fleet_bound'),fleet_proved_in_pool=r.get('fleet_proven'),individual_route_replay=r.get('physical_replay_validated'),duplicate_removal_validated=r.get('duplicate_trip_removal_validated'),shared_capacity_validated=r.get('cross_route_charger_capacity_validated'))
 row['cumulative_recorded_process_elapsed_all_stages_s']=sum(cum.values());row['cumulative_time_incomplete']=missing>0;row['attempts']=attempts;out.append(row)
print(json.dumps({'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':hashlib.sha256((B/'manifest.json').read_bytes()).hexdigest(),'rows':out,'time_semantics':'Process elapsed sum, not parallel calendar makespan, allocated CPU-hours or solver-only CG time. Missing ends require scheduler accounting. Report graph, initialization/CG and MIP separately.'},indent=2))
