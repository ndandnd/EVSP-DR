"""Idempotent recovery of only canceled cells; preserve healthy original trials."""
from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent
script=r'''
from pathlib import Path
import json,subprocess,time,fcntl,sys,hashlib
root=Path('/home/nc437/ladder-lite/mip_structure_20260922');sys.path.insert(0,str(root/'code_v3'));from launch import command,atomic
with (root/'recovery_v3.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 benchmark=json.loads((root/'loader_benchmark_v3.json').read_text());assert benchmark['passed']
 manifest=json.loads((root/'manifest_v3.json').read_text())
 for f,h in manifest['tooling_sha256'].items():assert hashlib.sha256((root/'code_v3'/f).read_bytes()).hexdigest()==h
 old=json.loads((root/'loader_failure_stop.json').read_text());receipt=root/'recovery_v3.json';d=json.loads(receipt.read_text()) if receipt.exists() else {'created_unix':time.time(),'code_commit':manifest['code_commit'],'jobs':{},'preserved_healthy':old['protected'],'replaced':old['cancelled'],'benchmark':benchmark}
 def submit(key,cmd):
  if key in d['jobs']:return d['jobs'][key]['job_id']
  intent=root/('recovery_v3_'+key+'_INTENT.json');assert not intent.exists(),'Unresolved submit intent: '+str(intent)
  intent.write_text(json.dumps({'argv':cmd,'created_unix':time.time()}));r=subprocess.run(cmd,text=True,capture_output=True,check=True);jid=r.stdout.strip().split(';')[0];assert jid.isdigit()
  control=subprocess.run(['scontrol','show','job','-o',jid],text=True,capture_output=True);d['jobs'][key]={'job_id':jid,'argv':cmd,'stdout':r.stdout,'stderr':r.stderr,'scontrol_at_submission':control.stdout,'submitted_unix':time.time()};atomic(receipt,d);return jid
 def cmd(case,mode,dep=None,arm=None):
  c=command(root,case,mode,dep,arm);c[c.index(str(root/'code/worker.sh'))]=str(root/'code_v3/worker_recovery_v3.sh');c=[v+'_v3' if v.startswith('--job-name=') else v for v in c];return c
 prep=submit('c3_k15_fresh__prepare',cmd('c3_k15_fresh','prepare'))
 for row in old['cancelled']:
  key=row['key'];case,arm=key.split('__');dependency=prep if case=='c3_k15_fresh' else None
  if dependency is None:
   gate=json.loads((root/'prepared'/f'{case}.json').read_text());assert hashlib.sha256(Path(gate['complete']).read_bytes()).hexdigest()==gate['sha256']
  submit(key,cmd(case,'trial',dependency,arm))
 print(json.dumps(d))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn',"bash -lc 'python3 -'"],input=script,text=True,capture_output=True);print(r.stderr) if r.returncode else None;r.check_returncode();(P/'recovery_v3.json').write_text(r.stdout);d=json.loads(r.stdout);print(json.dumps({k:v['job_id'] for k,v in d['jobs'].items()},indent=2))
