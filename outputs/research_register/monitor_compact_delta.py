"""Compare collector evidence on disk; never print route lists or dual vectors."""
from pathlib import Path
import json,hashlib,subprocess,collections
meta=json.loads(Path('/tmp/evsp-monitor-current.json').read_text())
if meta['returncode']:print(json.dumps(meta));raise SystemExit(1)
new=json.loads(Path(meta['snapshot']).read_text());old=json.loads(Path(meta['previous']).read_text())
def flatten(d):
 out={}
 for campaign,c in d.get('campaigns',{}).items():
  for stage in ['mip','cg','records','comparisons','rejected_mip_outputs']:
   for i,r in enumerate(c.get(stage,[])):out[campaign,stage,r.get('path',str(i))]=r
 return out
def digest(v):return v.get('sha256') or hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
a,b=flatten(old),flatten(new);changed=[]
for key,r in b.items():
 prior=a.get(key)
 if prior is None or digest(r)!=digest(prior):
  d=r.get('result',r);s={k:d.get(k) for k in ['buses','fleet_bound','fleet_proven','physical_replay_validated','certified_rc_optimal','stop_reason','wall_s','runtime_s','status_name'] if k in d}
  s['final']={k:v for k,v in (d.get('final')or{}).items() if not isinstance(v,(list,dict))}
  changed.append({'campaign':key[0],'stage':key[1],'path':key[2],'new_path':prior is None,'summary':s,'sha256':r.get('sha256'),'payload_sha256':digest(r)})
prev={r['attempt_key']:r for r in old.get('mip_preemption_study',{}).get('attempts',[])};states=[]
for r in new.get('mip_preemption_study',{}).get('attempts',[]):
 if r.get('State')!=prev.get(r['attempt_key'],{}).get('State'):states.append({k:r.get(k) for k in ['JobID','case_id','cohort','State','ExitCode','confirmed_preemption','result_exists']})
def queue(d):
 result={}
 for line in d.get('squeue',{}).get('stdout','').splitlines():
  p=line.split('|',3)
  if len(p)==4:result[p[0]]={'state':p[1],'reason':p[3]}
 return result
qa,qb=queue(old),queue(new)
qd=[{'job':j,'before':qa.get(j),'after':qb.get(j)} for j in sorted(qa.keys()|qb.keys()) if qa.get(j)!=qb.get(j)]
delta={'previous':meta['previous'],'snapshot':meta['snapshot'],'timestamp_utc':new['timestamp_utc'],'changed_results':changed,'mip_scheduler_transitions':states,'queue_transitions':qd}
p=Path(meta['snapshot']).with_name(Path(meta['snapshot']).stem+'_delta.json');p.write_text(json.dumps(delta,indent=2)+'\n')
print('Delta:',p)
print('Changed rows:',dict(collections.Counter((r['campaign'],r['stage']) for r in changed)))
for r in changed:
 s=r['summary'];parts=Path(r['path']).parts;cid=parts[parts.index('cases')+1] if 'cases' in parts else Path(r['path']).stem
 if r['stage']=='mip':print(r['campaign'],cid,'MIP',s.get('buses'),'bound',s.get('fleet_bound'),'proved',s.get('fleet_proven'),'replay',s.get('physical_replay_validated'))
 elif r['stage']=='cg' and s.get('certified_rc_optimal'):print(r['campaign'],cid,'CG certified',round((s.get('wall_s') or 0)/60,1),'min')
print('Scheduler changes:',len(states),'problems:',[x for x in states if x['State'] in ['FAILED','PREEMPTED','TIMEOUT','OUT_OF_MEMORY']])
print('Queue transitions:',len(qd),'invalid:',[j for j,r in qb.items() if 'NeverSatisfied' in r['reason']])
subprocess.run(['python3','outputs/research_register/preemption_study/refresh.py',meta['snapshot']],check=True)
