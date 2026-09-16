import sys,json,hashlib
from pathlib import Path
out=[]
for row in json.load(sys.stdin):
 p=Path(row['status_path']); status=json.loads(p.read_text()); t=Path(row['telemetry_path']); raw=t.read_bytes(); assert hashlib.sha256(raw).hexdigest()==row['telemetry_sha256']; events=[json.loads(l) for l in raw.splitlines() if l.strip()]; phases=[r for r in events if r.get('details',{}).get('purpose')=='final_resolve']; out.append({'chain':row['chain'],'status_path':str(p),'status_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'stop_reason':status.get('stop_reason'),'termination_signal':status.get('termination_signal'),'wall_s':status.get('wall_s'),'telemetry_path':str(t),'telemetry_sha256':row['telemetry_sha256'],'telemetry_records':len(events),'final_resolve_phases':phases,'total_final_resolve_s':sum(r['duration_s'] for r in phases)})
print(json.dumps(out,indent=2))