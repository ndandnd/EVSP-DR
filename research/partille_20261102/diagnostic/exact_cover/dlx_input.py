"""Read nested real JSON, find peak sets by a one-minute sweep, and emit unique eligible trip sets."""
import csv, hashlib, json, sys
from pathlib import Path
from collections import Counter
instance, pool, out=map(Path,sys.argv[1:4])
assert hashlib.sha256(instance.read_bytes()).hexdigest()=='c1dbe1dd15d20979799c68d01430a5fd0eddc2ca3ced3f46bd8522123e6c480d'
def minute(v):
    h,m=map(int,v.split(':'))
    return h*60+m
rows=list(csv.DictReader(instance.open()))
by_minute=[set() for _ in range(max(minute(r['End1']) for r in rows))]
for i,r in enumerate(rows):
    for m in range(minute(r['Start1']),minute(r['End1'])):
        by_minute[m].add(i)
peak=max(map(len,by_minute))
groups={frozenset(s) for s in by_minute if len(s)==peak}
eligible=set()
digest=hashlib.sha256()
n=retained=0
with pool.open('rb') as f:
    for line in f:
        n+=1
        digest.update(line)
        r=json.loads(line)
        assert len(r['trips'])==len(set(r['trips'])) and all(0<=t<len(rows) for t in r['trips'])
        trips=frozenset(r['trips'])
        hits={len(g&trips) for g in groups}
        assert max(hits)<=1
        if hits=={1}:
            retained+=1
            eligible.add(tuple(sorted(trips)))
assert digest.hexdigest()=='5744645fa8fc62e3c79e3880a12b1388dda8e0342fe46bb6c700d61e98c86d0d'
out.write_text(''.join(' '.join(map(str,ts))+chr(10) for ts in sorted(eligible)))
print(json.dumps(dict(pool_sha256=digest.hexdigest(), peak=peak, total=n, eligible_columns=retained,
                      unique_trip_sets=len(eligible), peak_sets=len(groups), trips=len(rows),
                      input_sha256=hashlib.sha256(out.read_bytes()).hexdigest())))
