from pathlib import Path
import json,csv,subprocess,hashlib,base64,datetime
P=Path(__file__).resolve().parent;rows=list(csv.DictReader((P.parent/'chain_comparison_mip_times/mip_stage_times.csv').open()));payload=[dict(case_id=r['case_id'],arm=r['arm'],remote_source=r['remote_source']) for r in rows]
remote='rows='+repr(payload)+r'''
from pathlib import Path
import json,hashlib,base64
out=[]
for row in rows:
 p=Path(row['remote_source']);result=json.loads(p.read_text());a=dict(row,result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),files={})
 for name in ['gurobi.log','state.json']:
  f=p.parent/name
  if f.exists():
   b=f.read_bytes();a['files'][name]={'remote':str(f),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()}
 cg=Path(result['source_result']);d=json.loads(cg.read_text());a['source_cg_metadata']={k:v for k,v in d.items() if len(json.dumps(v))<1500};a['source_cg_trip_count']=len(d.get('trip_ids',[]));a['source_cg_sha256']=hashlib.sha256(cg.read_bytes()).hexdigest();a['journal_bytes']=Path(result['source_journal']).stat().st_size
 out.append(a)
print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn','python3 -'],input=remote,text=True,capture_output=True,check=True);out=json.loads(r.stdout)
for row in out:
 for name,f in row['files'].items():
  b=base64.b64decode(f.pop('data'));assert hashlib.sha256(b).hexdigest()==f['sha256'];loc='sources/'+row['case_id']+'_'+row['arm']+'_'+name;(P/loc).write_bytes(b);f['local']=loc
(P/'collection.json').write_text(json.dumps({'collected_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'endpoints':out},indent=2)+'\n')
print(json.dumps({'endpoints':len(out),'log_count':sum('gurobi.log' in r['files'] for r in out),'total_small_file_bytes':sum(f['bytes'] for r in out for f in r['files'].values()),'CG_fields':list(out[0]['source_cg_metadata']),'journal_bytes_all48':sum(r['journal_bytes'] for r in out)}))
