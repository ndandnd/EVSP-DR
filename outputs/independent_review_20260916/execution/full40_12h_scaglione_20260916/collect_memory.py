import sys,json,hashlib
from pathlib import Path
out=[]
for row in json.load(sys.stdin):
 p=Path(row['status_path']); raw=p.read_bytes(); d=json.loads(raw); assert hashlib.sha256(raw).hexdigest()==row['status_sha256'];out.append({'chain':row['chain'],'source_status':str(p),'status_sha256':row['status_sha256'],'trips':len(d['trip_ids']),'columns':d['columns'],'peak_rss_mb':d['peak_rss_mb'],'network':d.get('event_network') or d.get('network'),'graph_first_telemetry':row['telemetry_first'][-1],'inheritance_workers':d.get('inherited_event_pool_audit',{}).get('replay_workers')})
print(json.dumps(out,indent=2))