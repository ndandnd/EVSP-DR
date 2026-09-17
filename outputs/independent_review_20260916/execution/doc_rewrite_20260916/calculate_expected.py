from pathlib import Path
import json,csv
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
E=Path('outputs/independent_review_20260916/execution');O=E/'doc_rewrite_20260916';tz=ZoneInfo('America/New_York')
lines=(O/'slurm_timing.txt').read_text().splitlines();s={}
for line in lines[1:]:
 r=line.split('|')
 if len(r)!=7 or r[4].count(':') and r[4][:4]!='2026':continue
 if not r[0].isdigit() and '_' not in r[0]:continue
 if r[3][:4]!='2026' and r[3] not in ('Unknown','None'):continue
 s[r[0]]=dict(zip(('job','name','state','start','end','limit','elapsed'),r))
def dur(v):
 days=0
 if '-' in v: a,v=v.split('-');days=int(a)
 h,m,ss=map(int,v.split(':'));return timedelta(days=days,hours=h,minutes=m,seconds=ss)
def dt(x):return datetime.fromisoformat(x).replace(tzinfo=tz)
records={}
files=['p1/jobs.json','p2/dr_mincharge/jobs.json','p2_strict/jobs.json','p2_random/jobs.json','p3_frolunda/jobs.json','advisor_followup_20260916/f6_k5/jobs.json']
for file in files:
 d=json.loads((E/file).read_text())
 if isinstance(d,dict):d=d.get('jobs',d)
 if isinstance(d,dict):d=list(d.values()) if 'job_id' not in d else [d]
 for r in d:
  records[r['job_id']]={**r,'source':file}
cache={}
def finish(job):
 if job in cache:return cache[job]
 r=s.get(job)
 if r and r['state']=='COMPLETED':out=(dt(r['end']),'actual completion')
 elif r and r['state']=='RUNNING':out=(dt(r['start'])+dur(r['limit']),'allocation deadline')
 else:
  rec=records.get(job,{})
  deps=rec.get('dependencies',[])
  if not deps:return None
  ends=[finish(str(d)) for d in deps]
  if not all(ends):return None
  args=rec.get('argv',rec.get('command',[]));v=next(x.split('=',1)[1] for x in args if x.startswith('--time='))
  out=(max(x[0] for x in ends)+dur(v),'full allocation dependency projection; zero queue delay')
 cache[job]=out;return out
rows=[]
for job,rec in records.items():
 if job=='341352':continue
 out=finish(job);r=s.get(job,{})
 rows.append({'job':job,'source':rec['source'],'state':r.get('state','unknown'),'start':r.get('start',''), 'end_or_projection':out[0].isoformat() if out else '', 'basis':out[1] if out else 'no scheduled start'})
with (O/'expected_by_job.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for label,ids in [('fresh',[r['job'] for r in rows if r['source']=='p1/jobs.json' and s.get(r['job'],{}).get('name','').startswith('drP_fresh')]),('C5',['340968']),('k32',[str(x) for x in range(340969,340977)]),('F6',[str(x) for x in range(341682,341688)]),('DR',[str(x) for x in range(341152,341200)]),('strict',[r['job'] for r in rows if r['source']=='p2_strict/jobs.json']),('random',[r['job'] for r in rows if r['source']=='p2_random/jobs.json']),('frolunda',['341413'])]:
 e=[finish(j) for j in ids];valid=[x[0] for x in e if x];print(label, len(ids), min(valid) if valid else None,max(valid) if valid else None, 'unknown',sum(x is None for x in e))
