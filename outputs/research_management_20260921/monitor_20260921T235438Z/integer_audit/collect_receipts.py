"""Read-only retrieval of selected small receipts/logs; reuse identical archived local copies."""
from pathlib import Path
import json,subprocess,hashlib,shutil,tarfile,io
P=Path(__file__).resolve().parent;OLD=P.parents[1]/'integer_columns'
files=json.loads((P/'remote_inventory.json').read_text());need=[];receipt=[]
for f in files:
 src=OLD/f['relative'];dst=P/f['relative'];dst.parent.mkdir(parents=True,exist_ok=True)
 if src.exists() and hashlib.sha256(src.read_bytes()).hexdigest()==f['sha256']:
  shutil.copyfile(src,dst);mode='reuse_identical_prior_local'
 else:need.append(f);mode='new_read_only_remote_fetch'
 receipt.append(dict(f,retrieval=mode,local=str(dst)))
if need:
 script="import tarfile,sys\npaths="+repr([f['relative'] for f in need])+"\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:\n for p in paths:t.add('/home/nc437/ladder-lite/integer_columns_20260921/'+p,arcname=p)\n"
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','unicorn','python3','-'],input=script.encode(),capture_output=True,check=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout),mode='r:') as t:
  for f in need:
   raw=t.extractfile(f['relative']).read();(P/f['relative']).write_bytes(raw)
for f in files:assert hashlib.sha256((P/f['relative']).read_bytes()).hexdigest()==f['sha256'],f
(P/'retrieval_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n');print('verified',len(files),'new downloads',len(need),'reused',len(files)-len(need))
# Snapshot text transport may normalize CSV CRLF; retrieve exact-byte small root receipts.
script="""from pathlib import Path
import json,base64,hashlib
P=Path('/home/nc437/ladder-lite/integer_columns_20260921')
print(json.dumps({n:{'sha256':hashlib.sha256((P/n).read_bytes()).hexdigest(),'data':base64.b64encode((P/n).read_bytes()).decode()} for n in ['manifest.json','jobs.tsv','evidence_latest.json','evidence_latest.csv','scheduler_latest.txt']}))"""
import base64
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','unicorn','python3','-'],input=script,text=True,capture_output=True,check=True)
root_files=json.loads(r.stdout)
for name,v in root_files.items():
 raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256'];(P/name).write_bytes(raw)
(P/'root_receipts_hashes.json').write_text(json.dumps({k:v['sha256'] for k,v in root_files.items()},indent=2)+'\n')
