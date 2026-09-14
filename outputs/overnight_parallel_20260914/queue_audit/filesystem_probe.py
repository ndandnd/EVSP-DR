import json,pathlib,time,datetime
roots=['chain_extension_20260913','chain_extension_20260914'];out={'sampled_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'campaigns':{}}
def read(p):return json.loads(p.read_text())
def tail(p):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-200000)); lines=f.read().decode().splitlines()
 for x in reversed(lines):
  try:return json.loads(x)
  except Exception:pass
for name in roots:
 b=pathlib.Path('/home/nc437/ladder-lite')/name;m=read(b/'manifest.json');cases={}
 for cid,c in m['cases'].items():
  d=b/'cases'/cid;row={'trips':c['trip_count'],'parent_status':c['parent_status'],'cg_published':(d/'cg.json').exists(),'mip_published':(d/'mip_result.json').exists()}
  for stage in ['cache','cg','mip']:
   paths=sorted((d/stage).glob('*/execution.json'),key=lambda p:p.stat().st_mtime)
   if paths:
    p=paths[-1];v=read(p);row[stage]={k:v.get(k) for k in ['attempt','status','started_epoch','ended_epoch','returncode','watchdog_time_limit','signals']};row[stage]['path']=str(p)
  p=d/'cache_result.json'
  if p.exists():row['cache_result']=read(p)
  ps=sorted((d/'cache').glob('*/progress.jsonl'),key=lambda p:p.stat().st_mtime)
  if ps:
   row['progress']=tail(ps[-1]);row['progress_age_s']=time.time()-ps[-1].stat().st_mtime
  cases[cid]=row
 out['campaigns'][name]={'cases':cases,'resources':m['resources']}
print(json.dumps(out))
