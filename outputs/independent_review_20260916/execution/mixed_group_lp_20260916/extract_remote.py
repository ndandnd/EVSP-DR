"""No-solver reader: canonical CG files only; no mutation or large graph/journal load."""
import base64,gzip,hashlib,json
from pathlib import Path

def canonical_sha(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def extract(rows):
 out=[]
 for row in rows:
  p=Path(row['cg_path']);result={'case_id':row['case_id'],'path':str(p)}
  try:
   raw=p.read_bytes();d=json.loads(raw)
   payload={k:v for k,v in d.items() if k not in ['routes','columns','selected_routes','iterations','history','iteration_log'] and not isinstance(v,list)};payload['path']=str(p)
   result.update(file_sha256=hashlib.sha256(raw).hexdigest(),canonical_payload_sha256=canonical_sha(payload),final_lp=d.get('final_lp'),final=d.get('final'),final_lp_source=d.get('final_lp_source'),trip_ids=d.get('trip_ids'),columns=d.get('columns'),iterations=d.get('iterations'),certified_rc_optimal=d.get('certified_rc_optimal'),stop_reason=d.get('stop_reason'),csv=d.get('csv'),instance_sha256=d.get('provenance',{}).get('instance_sha256'),execution_commit=d.get('provenance',{}).get('git_commit'))
  except Exception as exc:result['error']=type(exc).__name__+': '+str(exc)
  out.append(result)
 data=json.dumps(out,separators=(',',':')).encode();print(base64.b64encode(gzip.compress(data,mtime=0)).decode())
