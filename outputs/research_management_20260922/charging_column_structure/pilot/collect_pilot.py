#!/usr/bin/env python3
"""Read-only snapshot collector; never launches a job or optimizer."""
from pathlib import Path
import argparse,base64,datetime,hashlib,json,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('--job',default='729675');p.add_argument('--remote',default='/share/scaglione/nc437/evsp-dr/charging_column_structure_20260922/pilot');p.add_argument('--out',type=Path,default=Path(__file__).parent/'collections');a=p.parse_args();assert a.job.isdigit()
code='''import pathlib,json,base64,subprocess
root=pathlib.Path(REMOTE)
files={}
paths=list(root.glob('attempts/JOB_r*/*'))+list(root.glob('attempts/JOB_r*/*/*'))+[root/'capacity_repr_JOB.out']
for f in paths:
 if f.is_file():
  try:files[str(f.relative_to(root))]=base64.b64encode(f.read_bytes()).decode()
  except FileNotFoundError:pass
cmds={'squeue':['/usr/local/slurm/current/bin/squeue','-j','JOB','-h','-o','%i|%T|%R|%M'], 'sacct':['/usr/local/slurm/current/bin/sacct','-j','JOB','--parsable2','--noheader','--format=JobID,State,ExitCode,Elapsed,MaxRSS,NodeList,AllocCPUS'], 'scontrol':['/usr/local/slurm/current/bin/scontrol','show','job','JOB','-o']}
status={}
for k,cmd in cmds.items():
 r=subprocess.run(cmd,capture_output=True,text=True);status[k]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps({'files':files,'scheduler':status}))
'''.replace('REMOTE',repr(a.remote)).replace('JOB',a.job)
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','unicorn','python3 -c '+shlex.quote(code)],check=True,capture_output=True,text=True);d=json.loads(r.stdout);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=a.out/stamp;out.mkdir(parents=True,exist_ok=False);hashes={}
for name,b in d['files'].items():
 rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
 f=out/rel;f.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(b);f.write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
record=dict(collected_utc=stamp,job_id=a.job,remote_root=a.remote,scheduler=d['scheduler'],files_sha256=hashes,scope='Read-only point-in-time copy; incomplete running logs are not final solver proofs')
(out/'collection.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(out=str(out),job_id=a.job,files=len(hashes),squeue=d['scheduler']['squeue']['stdout'],sacct=d['scheduler']['sacct']['stdout']),indent=2))
