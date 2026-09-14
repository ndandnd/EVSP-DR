from pathlib import Path
import subprocess,json,hashlib,datetime
b=Path('/home/nc437/ladder-lite/capacity_pricing_boundary_20260914/results/k1_13407_flat_capacity_prefix_memo/189169_r0')
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case':'k1_13407_flat_capacity_prefix_memo','job_id':'189169'}
out['sacct']=subprocess.run(['/usr/local/slurm/slurm-25.05.5/bin/sacct','-n','-D','-X','-P','-j','189169','--starttime=2026-09-14T00:00:00','-o','JobID,State,Start,End,ElapsedRaw,ExitCode,Reason,Restarts'],capture_output=True,text=True,check=True).stdout
out['files']={}
for name in ['worker_status.json','commands.json','allocation.json','pool.jsonl','cg.json','mip.json','cg.stdout.log','cg.stderr.log']:
 p=b/name;v={'path':str(p),'exists':p.exists()}
 if p.exists():
  raw=p.read_bytes();v.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
  if name in ['worker_status.json','commands.json']:v['contents']=json.loads(raw)
  if name.endswith('.log'):v['last_lines']=raw.decode().splitlines()[-12:]
 out['files'][name]=v
print(json.dumps(out))
