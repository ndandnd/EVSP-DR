"""Single read-only SSH collection of102 archived final LP supports."""
import base64,csv,datetime,gzip,hashlib,json,subprocess
from pathlib import Path
B=Path(__file__).resolve().parent;SOURCE=B.parent/'audited_chain_results.csv';rows=list(csv.DictReader(SOURCE.open()))
script=(B/'extract_remote.py').read_text()+'\nextract('+repr(rows)+')\n'
p=subprocess.run(['ssh','-S','/Users/nadan/.ssh/evsp-unicorn.sock','-o','BatchMode=yes','-o','ConnectTimeout=8','nc437@unicorn-login-01.coecis.cornell.edu','/home/nc437/evsp_env/bin/python -'],input=script,text=True,capture_output=True,timeout=120)
if p.returncode:raise RuntimeError(p.stderr)
payload=base64.b64decode(p.stdout.strip());d=json.loads(gzip.decompress(payload));assert len(d)==102
(B/'saved_final_supports.json.gz').write_bytes(payload)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt={'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records':len(d),'source_table_sha256':sha(SOURCE),'remote_reader_sha256':sha(B/'extract_remote.py'),'archive_sha256':sha(B/'saved_final_supports.json.gz'),'solver_invocations':0,'scheduler_mutations':0,'missing_final_lp':sum(not r.get('final_lp') for r in d),'source_errors':[r for r in d if 'error'in r]}
(B/'collection_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
