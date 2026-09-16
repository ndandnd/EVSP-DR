#!/usr/bin/env python3
"""Bounded review-only monitoring; no submissions, cancellations or old collector."""
import argparse,datetime,hashlib,json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--stamp');args=parser.parse_args()
stamp=args.stamp or datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
if len(stamp)!=16 or not stamp.endswith('Z'):raise ValueError('Use UTC stamp YYYYMMDDTHHMMSSZ')
out=HERE/'monitor'/stamp;out.mkdir(parents=True,exist_ok=False)
ssh=['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu']
script=HERE/'collect_review_remote.py';start=time.monotonic()
try:
 # Read-only over science artifacts; new remote monitor reports are the sole writes.
 proc=subprocess.run(ssh+['/home/nc437/evsp_env/bin/python - '+stamp+' '+hashlib.sha256(script.read_bytes()).hexdigest()],input=script.read_bytes(),capture_output=True,timeout=135)
 if proc.returncode:raise RuntimeError('Remote collector failed: '+proc.stderr.decode()[-2500:])
 d=json.loads(proc.stdout)
 rows=d['campaigns'].get('dr',{}).get('data',{}).get('rows',[])
 assert all(r['tariff'] in ['peak08','peak12','peak18'] for r in rows),'Refusing non-synthetic DR results'
 (out/'snapshot.json').write_bytes(proc.stdout+b'\n')
 for name,v in d['scheduler'].items():
  if v.get('ok'):(out/(name+'.txt')).write_text(v['text'])
 receipt={'status':('collected' if all(v['collection_ok'] for v in d['campaigns'].values()) and all(v.get('ok') for v in d['scheduler'].values()) else 'partial_collection'),'stamp':stamp,'elapsed_s':time.monotonic()-start,'snapshot_sha256':hashlib.sha256((out/'snapshot.json').read_bytes()).hexdigest(),'local_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'remote_script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'campaign_collector_success':{k:v['collection_ok'] for k,v in d['campaigns'].items()},'scheduler_query_success':{k:v.get('ok') for k,v in d['scheduler'].items()}}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(HERE/'monitor/latest_path.txt').write_text(str(out)+'\n');print(json.dumps(receipt))
except Exception as exc:
 (out/'failure.json').write_text(json.dumps({'error':str(exc),'elapsed_s':time.monotonic()-start},indent=2));raise
