#!/usr/bin/env python3
"""Submit a campaign wave and record unambiguous Slurm job IDs."""
import argparse,datetime as dt,hashlib,json,subprocess
from pathlib import Path
def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--wave',choices=('construction','fixture','mip'),required=True);a=p.parse_args()
root=Path(a.root).resolve();m=json.loads((root/'manifest.json').read_text());jobs={};prior=root/'jobs.json'
if prior.exists():jobs=json.loads(prior.read_text()).get('jobs',{})
ids=m['construction_case_ids'] if a.wave=='construction' else [m['native_fixture_case_id']] if a.wave=='fixture' else m['production_mip_case_ids']
for cid in ids:
 c=m['cases'][cid]
 if c['kind']=='mip':
  for src in c['source_partitions']:
   if not (root/'cases'/src/'completion.json').exists():raise SystemExit(f'missing completion: {src}')
 (root/'logs').mkdir(exist_ok=True);args=['/usr/local/slurm/slurm-25.05.5/bin/sbatch','--parsable','--no-requeue','--job-name','du_'+cid,'--cpus-per-task',str(c['resources']['cpus']),'--mem',str(c['resources']['mem_gb'])+'G','--time',c['resources']['allocation'],'--output',str(root/'logs'/(cid+'_%j.out')),'--error',str(root/'logs'/(cid+'_%j.err')),'--export',f'ALL,CAMPAIGN_ROOT={root},CASE_ID={cid}',str(root/'worker.sub')]
 r=subprocess.run(args,text=True,capture_output=True)
 if r.returncode:raise SystemExit(f'sbatch failed {cid}: {r.stderr}')
 jid=r.stdout.strip().split(';')[0]
 if not jid.isdigit():raise SystemExit(f'ambiguous sbatch response {cid}: {r.stdout!r}')
 jobs[cid]={'job_id':jid,'wave':a.wave,'submitted_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'argv':args,'stdout':r.stdout.strip()};save(root/'jobs.json',{'schema':'evsp-decomposition-pool-union-jobs-v1','manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),'jobs':jobs});print(cid,jid)
