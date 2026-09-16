import csv,json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent
r=json.load(open(p/'fixed_duty_rerun.json'))['results'];by={x['duty']:x for x in r}
c=list(csv.DictReader(open(p/'chain_replay_counts.csv')))
for x in c:
 ds=x['duties'].split(';');x['fixed_trip_certificate_reverified_this_audit']=all(by[t]['feasible'] and by[t]['replay']=='validated' and by[t]['certified'] for t in ds)
 x['fixed_trip_optimized_charging_reverified_valid_count']=sum(by[t]['feasible'] and by[t]['replay']=='validated' for t in ds)
with open(p/'chain_replay_counts.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=c[0]);w.writeheader();w.writerows(c)
hashes={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and 'pinned' not in f.parts and '__pycache__' not in f.parts and f.name!='SHA256SUMS.json'}
json.dump(hashes,open(p/'SHA256SUMS.json','w'),indent=2,sort_keys=True)
