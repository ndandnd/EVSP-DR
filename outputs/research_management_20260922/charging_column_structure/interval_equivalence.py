"""Exact-rational fixed-interval equivalence checks. No solver, no license."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import csv,json,random,hashlib
P=Path(__file__).resolve().parent
routes=[dict(route='r1',trips=[1],sessions=[('S',F(0),F(2))]),dict(route='r2',trips=[2],sessions=[('S',F(1),F(3))]),dict(route='r3',trips=[3],sessions=[('S',F(2),F(4))]),dict(route='r4',trips=[2],sessions=[('S',F(4),F(5))]),dict(route='r5',trips=[1,3],sessions=[('S',F(0),F(1)),('S',F(3),F(5))])]
def matrix(rs):
 stations=sorted({s for r in rs for s,a,b in r['sessions']});rows=[];difference=[]
 for station in stations:
  events=sorted({t for r in rs for s,a,b in r['sessions'] if s==station for t in [a,b]});prev=[0]*len(rs)
  for a,b in zip(events,events[1:]):
   support=[int(any(s==station and left<=a<right for s,left,right in r['sessions'])) for r in rs]
   rows.append((station,a,b,support));difference.append((station,a,[x-y for x,y in zip(support,prev)]));prev=support
  difference.append((station,events[-1],[-x for x in prev]))
 return rows,difference
B,D=matrix(routes);A=[[int(t in r['trips']) for r in routes] for t in [1,2,3]]
def dot(a,x):return sum(F(v)*z for v,z in zip(a,x))
def check(rs,assignments):
 b,d=matrix(rs);cases=0
 for x in assignments:
  cumulative={};live={};exact=[]
  for s,t,row in d:
   cumulative[s]=cumulative.get(s,F(0))+dot(row,x);live[s,t]=cumulative[s]
  for s,left,right,row in b:
   direct=sum(x[i] for i,r in enumerate(rs) if any(st==s and a<=(left+right)/2<end for st,a,end in r['sessions']))
   assert direct==dot(row,x)==live[s,left]
   exact.append(direct<=1)
  assert all(exact)==all(v<=1 for v in live.values())
  assert all(v==0 for v in cumulative.values());cases+=1
 return cases
binary=check(routes,product([F(0),F(1)],repeat=5));rational=check(routes,product([F(0),F(1,2),F(1)],repeat=5))
def write(name,rr):
 with (P/name).open('w') as f:
  w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
write('toy_routes.csv',[dict(route=r['route'],trips=';'.join(map(str,r['trips'])),station=s,start_min=str(a),end_min=str(b),kwh_if_60kw=str(b-a)) for r in routes for s,a,b in r['sessions']])
write('toy_trip_A.csv',[dict(trip=t,**dict(zip([r['route'] for r in routes],row))) for t,row in zip([1,2,3],A)])
write('toy_capacity_B.csv',[dict(station=s,start_min=str(a),end_min=str(b),capacity=1,**dict(zip([r['route'] for r in routes],row))) for s,a,b,row in B])
write('toy_endpoint_D.csv',[dict(station=s,event_min=str(t),**dict(zip([r['route'] for r in routes],row))) for s,t,row in D])
random.seed(22);random_cases=0
for _ in range(100):
 rs=[]
 for i in range(5):
  sessions=[]
  for j in range(random.randrange(1,4)):
   start=F(random.randrange(0,20),4);end=start+F(random.randrange(1,10),4);sessions.append((random.choice(['S','T']),start,end))
  rs.append(dict(route=str(i),trips=[i],sessions=sessions))
 # Arbitrary overlaps are unioned per route/site, matching binary occupancy semantics.
 random_cases+=check(rs,[tuple(F(random.randrange(0,5),4) for _ in rs) for _ in range(20)])
# Minute conservative rounding vs true half-open overlap: adjacent disjoint subminute sessions.
rs=[dict(route='a',trips=[1],sessions=[('S',F(1,10),F(4,10))]),dict(route='b',trips=[2],sessions=[('S',F(6,10),F(9,10))])]
b,d=matrix(rs);true_peak=max(sum(row) for _,_,_,row in b);rounded=[[int(any(s=='S' and a<1 and end>0 for s,a,end in r['sessions'])) for r in rs]];rounded_peak=max(map(sum,rounded));assert true_peak==1 and rounded_peak==2
# Long sessions show potential exact grid row compression, while preserving original rounding.
long=[dict(route='a',trips=[1],sessions=[('S',F(0),F(100))]),dict(route='b',trips=[2],sessions=[('S',F(20),F(80))])]
grid=[[int(any(s=='S' and a<k+1 and end>k for s,a,end in r['sessions'])) for r in long] for k in range(100)]
compressed=[]
for row in grid:
 if not compressed or row!=compressed[-1]:compressed.append(row)
assert len(compressed)==3
for x in product([F(0),F(1,2),F(1)],repeat=2):assert all(dot(r,x)<=1 for r in grid)==all(dot(r,x)<=1 for r in compressed)
summary=dict(binary_selections_tested=binary,rational_selections_tested=rational,random_rational_cases_tested=random_cases,exact_equivalence_passed=True,toy_A_shape=[len(A),5],toy_B_shape=[len(B),5],toy_D_shape=[len(D),5],toy_B_nnz=sum(sum(row) for s,a,b,row in B),toy_D_nnz=sum(sum(v!=0 for v in row) for s,t,row in D),conservative_bin_counterexample=dict(exact_peak=true_peak,one_minute_overlap_row_peak=rounded_peak),long_interval_grid_rows=100,long_interval_identical_adjacent_runs=3,scope='Exact rational algebra and enumeration; no optimizer calls; fixed intervals only')
(P/'test_results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
