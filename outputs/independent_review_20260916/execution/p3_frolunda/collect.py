"""Read-only status extraction for the one-job FDL ladder."""
from pathlib import Path
import json,hashlib,datetime
B=Path('/home/nc437/ladder-lite/review_frolunda_20260916');v=json.loads((B/'manifest.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows=[]
for cid,c in v['cases'].items():
 r={'case':cid,'reference_duties':c['k'],'input_trips':c['trip_count'],'input_sha256':c['input_sha256'],'attempts':[]}
 for stage in ['cache','cg','mip']:
  root=B/'cases'/cid;done=root/(stage+'_done.json');r[stage+'_completed']=done.exists()
  for p in sorted((root/stage).glob('*/execution.json')):
   a=json.loads(p.read_text());r['attempts'].append({'stage':stage,'path':str(p),'sha256':sha(p),'status':a['status'],'attempt':a['attempt'],'started_epoch':a.get('started_epoch'),'ended_epoch':a.get('ended_epoch'),'returncode':a.get('returncode')})
  if done.exists():
   x=json.loads(done.read_text());r[stage+'_receipt']=x;r[stage+'_receipt_sha256']=sha(done)
   if stage in ['cg','mip']:
    p=Path(x['result']);z=json.loads(p.read_text());r[stage+'_result_path']=str(p);r[stage+'_result_sha256']=sha(p)
    if stage=='cg':r.update(cg_pricing_certificate=z.get('certified_rc_optimal'),cg_stop_reason=z.get('stop_reason'),cg_wall_s=z.get('wall_s'),cg_iterations=z.get('iterations'),cg_final=z.get('final'),cg_final_lp_source=z.get('final_lp_source'))
    else:r.update(buses=z.get('buses'),pool_fleet_bound=z.get('fleet_bound'),fleet_proved_in_pool=z.get('fleet_proven'),individual_route_replay=z.get('physical_replay_validated'),duplicate_removal_validated=z.get('duplicate_trip_removal_validated'),shared_capacity_validated=z.get('cross_route_charger_capacity_validated'))
 rows.append(r)
print(json.dumps({'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest_sha256':sha(B/'manifest.json'),'job':json.loads((B/'jobs.json').read_text()),'complete':(B/'ladder_complete.json').exists(),'rows':rows},indent=2))
