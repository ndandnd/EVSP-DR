"""Export the preemption-study section of a full collector snapshot."""
from pathlib import Path
import sys,json,csv,hashlib
source=Path(sys.argv[1]);raw=source.read_bytes();d=json.loads(raw);study=d['mip_preemption_study']
if 'collection_error' in study: raise RuntimeError(study['collection_error'])
root=Path(__file__).resolve().parent;samples=root/'snapshots';samples.mkdir(exist_ok=True)
name=study['timestamp_utc'].replace(':','').replace('+','_')+'.json';target=samples/name
body=json.dumps(study,indent=2)+'\n'
if target.exists() and target.read_text()!=body:raise FileExistsError(target)
target.write_text(body)
rows=study['attempts'];keys=sorted({k for r in rows for k in r})
with (root/'attempts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
(root/'summary.json').write_text(json.dumps(study['summary'],indent=2)+'\n')
(root/'latest.json').write_text(json.dumps({'source_snapshot':str(source.resolve()),'source_sha256':hashlib.sha256(raw).hexdigest(),'study_snapshot':str(target),'timestamp_utc':study['timestamp_utc']},indent=2)+'\n')
print(json.dumps({'attempt_rows':len(rows),'study_snapshot':str(target)}))
