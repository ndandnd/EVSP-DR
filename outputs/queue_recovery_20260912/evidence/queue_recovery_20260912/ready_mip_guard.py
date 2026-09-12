from pathlib import Path
import hashlib,json,os,sys
B=Path('/home/nc437/ladder-lite/queue_recovery_20260912')
v=json.loads((B/'ready_mips_manifest.json').read_text());idx=int(os.environ['SLURM_ARRAY_TASK_ID']);c=next(x for x in v['cases'] if x['index']==idx)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for f in v['common_files']+c['required_files']:
 if sha(f['path'])!=f['sha256']:raise RuntimeError('Required input changed: '+f['path'])
print('Verified completed parent',c['parent_job_id'],'case',c['case_id'],flush=True)
p='/home/nc437/evsp_env/bin/python'
os.execv(p,[p,'-u','/home/nc437/ladder-lite/overnight_extension_20260912/code/scripts/overnight_worker.py','mip',c['case_id']])
