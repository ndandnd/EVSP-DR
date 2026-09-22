"""No-solver exact matrix compression of saved strict-physics capacity pool."""
from pathlib import Path
import json,csv,math,hashlib,collections,random
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
SRC=ROOT/'outputs/meeting_20260910/giro_k23_capacity_duals/results/e1_short_k3/pool.jsonl'
routes=[json.loads(s) for s in SRC.read_text().splitlines()];counts={'2190L':1,'4808':1,'3127L':2,'7880C':1,'JON_A':1};grid=collections.defaultdict(set);route_intervals=[]
for j,r in enumerate(routes):
 bins=collections.defaultdict(set)
 for a in r['actions']:
  if a.get('kind')!='charge' or a['station'] not in counts:continue
  start,end=float(a['setup_start_min']),float(a['connection_end_min'])
  for k in range(math.floor(start),math.ceil(end)):
   if k+1e-9<end and k+1>start+1e-9:bins[a['station']].add(k)
 for station,ks in bins.items():
  for k in ks:grid[station,k].add(j)
  run=[]
  for k in sorted(ks):
   if run and k!=run[-1]+1:route_intervals.append(dict(route_index=j,station=station,start_min=run[0],end_min=run[-1]+1));run=[]
   run.append(k)
  if run:route_intervals.append(dict(route_index=j,station=station,start_min=run[0],end_min=run[-1]+1))
rows=[dict(station=s,minute=k,capacity=counts[s],support=tuple(sorted(v))) for (s,k),v in sorted(grid.items())]
unique={};mapping=[]
for r in rows:
 key=(r['station'],r['capacity'],r['support'])
 if key not in unique:unique[key]=len(unique)
 mapping.append(dict(station=r['station'],minute=r['minute'],compressed_row=unique[key]))
compressed=[dict(index=i,station=s,capacity=c,support=rr) for (s,c,rr),i in unique.items()]
D=[];runrows=[];aux_vars=0;eq_nnz=0;original_b_nnz=sum(len(r['support']) for r in rows)
for s in sorted({r['station'] for r in route_intervals}):
 intervals=[r for r in route_intervals if r['station']==s];events=sorted({t for r in intervals for t in [r['start_min'],r['end_min']]});prev=set()
 for ii,t in enumerate(events):
  now={r['route_index'] for r in intervals if r['start_min']<=t<r['end_min']};plus=now-prev;minus=prev-now
  D.append(dict(station=s,event_min=t,plus=sorted(plus),minus=sorted(minus)))
  for k in range(t,events[ii+1] if ii+1<len(events) else t):assert now==grid.get((s,k),set())
  if ii+1<len(events):runrows.append(dict(station=s,start_min=t,end_min=events[ii+1],capacity=counts[s],support=tuple(sorted(now))))
  aux_vars+=1;eq_nnz+=len(plus)+len(minus)+1+(ii>0);prev=now
 assert not prev
# Exact coefficient reconstruction, not just sampled objective/feasibility comparison.
for s in counts:
 cur=set()
 for d in [x for x in D if x['station']==s]:
  assert not (set(d['plus'])&cur);assert set(d['minus'])<=cur;cur-=set(d['minus']);cur|=set(d['plus'])
  assert cur==grid.get((s,d['event_min']),set())
 assert not cur
# Independent rational selection test using integer-scaled weights.
random.seed(22)
for _ in range(1000):
 x=[random.randrange(5) for r in routes]
 original=max([sum(x[j] for j in r['support'])-4*r['capacity'] for r in rows],default=0)
 compact=max([sum(x[j] for j in r['support'])-4*r['capacity'] for r in compressed],default=0);assert original==compact
trips=sorted({t for r in routes for t in r['trips']});annz=sum(len(r['trips']) for r in routes)
summary=dict(cohort='e1_short_k3',source=str(SRC.relative_to(ROOT)),source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),routes=len(routes),trip_rows=len(trips),trip_nnz=annz,capacity_rows_original_active=len(rows),capacity_nnz_original=original_b_nnz,capacity_rows_unique=len(compressed),capacity_nnz_unique=sum(len(r['support']) for r in compressed),capacity_event_segments=len(runrows),capacity_event_segments_nonempty=sum(bool(r['support']) for r in runrows),capacity_event_segment_nnz=sum(len(r['support']) for r in runrows),rounded_union_intervals=len(route_intervals),difference_endpoint_rows=len(D),difference_route_nnz=sum(len(d['plus'])+len(d['minus']) for d in D),difference_aux_variables=aux_vars,difference_full_equalities_nnz=eq_nnz,full_original_rows=len(trips)+len(rows),full_original_nnz=annz+original_b_nnz,full_unique_rows=len(trips)+len(compressed),full_unique_nnz=annz+sum(len(r['support']) for r in compressed),full_difference_rows=len(trips)+len(D),full_difference_variables=len(routes)+aux_vars,full_difference_nnz=annz+eq_nnz,coefficient_equivalence_verified=True,rational_selections_verified=1000,semantics='Preserved original any-overlap conservative one-minute plug occupancy, including setup-to-disconnection; no physical endpoint relaxation',objective='sum selected route variables, original fleet objective',solver_calls=0)
def write(name,rr):
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
write('native_grid_to_unique_rows.csv',mapping);write('native_compressed_rows.csv',[{**r,'support':';'.join(map(str,r['support']))} for r in compressed]);write('native_rounded_intervals.csv',route_intervals);write('native_endpoint_difference.csv',[{**d,'plus':';'.join(map(str,d['plus'])),'minus':';'.join(map(str,d['minus']))} for d in D]);(P/'native_compression_results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
