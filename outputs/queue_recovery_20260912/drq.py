#!/usr/bin/env python3
"""Read-only EVSP–DR queue summary; never submits or cancels jobs."""
from pathlib import Path
from collections import Counter,defaultdict
import subprocess,json,sys,datetime
B=Path('/home/nc437/ladder-lite');S='/usr/local/slurm/slurm-25.05.5/bin/';known={}
for directory,family in [('full_pool_recovery_20260912','Full-pool chains'),('graph_recovery_20260912','Decomposition recovery'),('graph_recovery_retry2_20260912','Decomposition recovery')]:
 p=B/directory/'jobs.json'
 if p.exists():
  for r in json.loads(p.read_text()):known[r['job_id']]=(family,r['case_id'],r['mode'])
p=B/'queue_recovery_20260912/ready_mips_submission.json'
if p.exists():
 x=json.loads(p.read_text());manifest=json.loads((p.parent/'ready_mips_manifest.json').read_text())
 for c in manifest['cases']:known[f"{x['job_id']}_{c['index']}"]=('Recovered ready MIPs',c['case_id'],'mip')
for j,c,m in [('949691','w1_k15','cg'),('949692','w1_k15','mip'),('37583','w2_k14','mip'),('37584','w2_k15','cg'),('37585','w2_k15','mip')]:known[j]=('Earlier bounded chains',c,m)
p=B/'controlled_comparison_20260913/jobs.json'
if p.exists():
 for entry in json.loads(p.read_text()).get('jobs',[]):
  if entry.get('job_id'):known[entry['job_id']]=('Controlled comparisons',entry['pair_id'],'paired CG/MIP')
r=subprocess.run([S+'squeue','--me','-r','-h','-o','%i|%j|%T|%M|%E|%R'],capture_output=True,text=True)
if r.returncode:raise SystemExit(r.stderr)
summary=defaultdict(Counter);rows=[]
for line in r.stdout.splitlines():
 j,name,state,elapsed,deps,reason=line.split('|',5)
 family,case,mode=known.get(j,('Held historical jobs' if j.startswith('537227_') else 'V2G jobs (untouched)' if name.startswith('V2G') else 'Other or unregistered',name,''))
 category='running' if state=='RUNNING' else 'held' if 'Held' in reason else 'invalid dependency' if 'DependencyNeverSatisfied' in reason else 'waiting for prerequisite' if 'Dependency' in reason else 'waiting for resources' if state=='PENDING' else state.lower()
 summary[family][category]+=1;rows.append({'job_id':j,'family':family,'case':case,'stage':mode,'state':state,'elapsed':elapsed,'dependencies':deps,'reason':reason})
if '--json' in sys.argv:print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'summary':dict(summary),'jobs':rows},indent=2));raise SystemExit
print('EVSP–DR queue | '+datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z'))
print(f"{'Work':28} {'Running':>8} {'Dependencies':>13} {'Resources':>10} {'Invalid':>8} {'Held':>6}")
for group,c in summary.items():print(f"{group:28} {c['running']:8} {c['waiting for prerequisite']:13} {c['waiting for resources']:10} {c['invalid dependency']:8} {c['held']:6}")
print('\nDependencies mean required data are not ready; these are not CPU limits.')
print('Full-pool CG waits for previous-k CG. MIP waits for its CG. Parent32 CG also waits for shared graph and component MIPs.')
if '--details' in sys.argv:
 print('\nJob        Case       Stage   State      Elapsed    Dependency / node')
 for x in rows:
  if x['family']=='Held historical jobs':continue
  print(f"{x['job_id']:10} {x['case']:10} {x['stage']:7} {x['state']:10} {x['elapsed']:10} {x['dependencies'] if x['state']=='PENDING' else x['reason']}")
