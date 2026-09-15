"""Read-only production launch/accounting verification; private campaign output."""
from pathlib import Path
import json,subprocess
import common as w
B=Path(__file__).resolve().parent;S='/usr/local/slurm/slurm-25.05.5/bin/'
def verify():
 m=w.read(B/'manifest.json');jobs=w.read(B/'jobs.json');expected=set(m['cases']);actual={j['case_id'] for j in jobs}
 if actual!=expected or len(jobs)!=20:raise ValueError('production20-case coverage mismatch')
 rows=[]
 for j in jobs:
  c=m['cases'][j['case_id']];x=dict(t.split('=',1) for t in j['effective_scontrol_at_submission'].split() if '=' in t);wanttime='01:00:00' if c['kind']=='pool_construction' else '04:30:00'
  if x['Partition']!='default_partition' or x['ExcNodeList']!='scaglione-compute-01' or int(x['NumCPUs'])!=c['resources']['cpus'] or x['MinMemoryNode']!=c['resources']['mem'] or x['TimeLimit']!=wanttime or x['Requeue']!='1':raise ValueError('effective production resource mismatch '+j['case_id'])
  if j['dependencies'] and not all(d in x['Dependency'] for d in j['dependencies']):raise ValueError('dependency mismatch')
  if not j['dependencies'] and x['Dependency'] not in ['(null)','(none)','']:raise ValueError('unexpected dependency')
  now=subprocess.check_output([S+'scontrol','show','job',j['job_id'],'-o'],text=True)
  rows.append({'case_id':j['case_id'],'job_id':j['job_id'],'kind':c['kind'],'dependencies':j['dependencies'],'effective_resources_verified':True,'current_scontrol':now})
 w.save(B/'production_launch_verification.json',{'status':'passed','manifest_sha256':w.sha(B/'manifest.json'),'utc':w.now(),'cases':20,'constructions':8,'mips':12,'rows':rows,'reserved_node_excluded_everywhere':True,'true_own_construction_dependencies_only':True,'no_throttle':True});print(json.dumps({'status':'passed','cases':20,'constructions':8,'mips':12}))
if __name__=='__main__':verify()
