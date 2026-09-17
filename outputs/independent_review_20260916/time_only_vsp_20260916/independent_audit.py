"""Independent read-only verification of frozen time-only certificates and F4 linkage.
Uses Dijkstra instead of generator Floyd-Warshall; never runs a solver/submits jobs.
"""
import csv,gzip,hashlib,heapq,json,time
from fractions import Fraction
from pathlib import Path
B=Path(__file__).resolve().parent;ROOT=B.parents[2];started=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 with open(p) as f:return list(csv.DictReader(f))
def norm(x):
 x=str(x).strip();return None if x.lower() in ('','nan','none') else x[:-2] if x.endswith('.0') else x
def base(x):
 y=x.rsplit('_',1);return y[0] if len(y)==2 and y[1].isdigit() else x
def clock(x):
 h,m=x.split(':');return int(h)*60+int(m)
summary=json.load(open(B/'summary.json'));certs=json.load(gzip.open(B/'certificates.json.gz','rt'));assert len(certs)==612
master=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv';assert sha(master)==summary['master_sha256'];mr={int(r['Ordered_Trip_ID']):r for r in rows(master) if r['Identifier']=='Regular'}
for p,h in summary['input_hashes'].items():assert sha(p)==h
for name,key in [('Ref_dict.csv','ref_sha256'),('par_ref_dhd.csv','deadhead_sha256')]:assert sha(B/'inputs'/name)==summary[key]
loc={norm(r['Location']):norm(r['Ref']) for r in rows(B/'inputs/Ref_dict.csv') if norm(r['Location']) and norm(r['Ref'])}
for x,y in list(loc.items()):loc.setdefault(base(x),y)
edges={}
for r in rows(B/'inputs/par_ref_dhd.csv'):
 a,b=norm(r['Start Place']),norm(r['End Place'])
 if not a or not b or a==b or not norm(r['Base Duration']) or not norm(r['Energy used']):continue
 t=Fraction(r['Base Duration']);assert t>=0
 for key in [(a,b),(b,a)]:edges[key]=min(edges.get(key,t),t)
refs=sorted({a for a,b in edges});INF=Fraction(10**12);short={}
for origin in refs:
 dist={x:INF for x in refs};dist[origin]=Fraction(0);q=[(Fraction(0),origin)]
 while q:
  value,node=heapq.heappop(q)
  if value!=dist[node]:continue
  for dest in refs:
   weight=edges.get((node,dest),INF)
   if value+weight<dist[dest]:dist[dest]=value+weight;heapq.heappush(q,(dist[dest],dest))
 short.update({(origin,dest):value for dest,value in dist.items()})
def resolve(v):
 raw=str(v).strip()
 for candidate in [raw,base(raw),norm(raw),norm(base(raw))]:
  if candidate in loc:return loc[candidate]
  if candidate in refs:return candidate
 raise AssertionError(v)
inputs={Path(p).stem:(Path(p),rows(p)) for p in summary['input_hashes']};f4=B.parent/'execution/f4';ledger={r['duty']:r for r in json.load(open(f4/'fixed_duty_rerun.json'))['results']};upper={};checked_duties=set()
for cid,(p,records) in inputs.items():
 duties={mr[int(r['Ordered_Trip_ID'])]['VehicleTask'] for r in records};mapping={}
 for duty in duties:
  dp=f4/'duty_inputs'/f'{duty}.csv';w=f4/'optimized_fixed_duties'/f'{duty}.json';d=json.load(open(w));assert sha(w)==ledger[duty]['result_sha256'];assert d['source_input_sha256']==sha(dp)==d['certificate']['declared_instance_sha256'];assert d['feasible'] and d['physical_replay']['ok'] and d['physical_replay_status']=='validated';phys=d['certificate']['physics'];assert phys['g_kwh']==240 and phys['charge_kw']==240 and phys['reserve_kwh']==0 and d['terminal_soc_policy']=='free'
  dr=rows(dp);want={int(r['Ordered_Trip_ID']) for r in records if mr[int(r['Ordered_Trip_ID'])]['VehicleTask']==duty};got={int(r['Ordered_Trip_ID']) for r in dr};assert want==got
  # Input positions may differ, but each physical trip's defining fields must match.
  byid={int(r['Ordered_Trip_ID']):r for r in records}
  for r in dr:
   assert all(norm(r[k])==norm(byid[int(r['Ordered_Trip_ID'])][k]) for k in ['From1','To1','Start1','End1','Usage kWh'])
  assert set(d['trip_sequence'])==set(range(len(dr)));checked_duties.add(duty)
  mapping[duty]=len(dr)
 upper[cid]={'continuous_feasible_fleet':len(duties),'duties':sorted(duties),'group_counts':{g:sum(x.startswith(prefix) for x in duties) for g,prefix in [('18E1','134'),('18E2','133')]},'event_lattice_upper_bound':'UNRESOLVED'}
verified=[];nontransitive=0
for c in certs:
 cid=c['case_id'];p,records=inputs[cid];assert sha(p)==c['input_sha256'];records=[r for r in records if c['group']=='mixed' or ('18E1' if mr[int(r['Ordered_Trip_ID'])]['VehicleTask'].startswith('134') else '18E2')==c['group']]
 assert [int(r['Ordered_Trip_ID']) for r in records]==c['ordered_trip_ids'];n=len(records)
 ss=[clock(r['Start1']) for r in records];ee=[clock(r['End1']) for r in records];assert all(e>s for s,e in zip(ss,ee));fr=[resolve(r['From1']) for r in records];to=[resolve(r['To1']) for r in records]
 dist=short if c['graph_kind']=='closure_relaxation' else edges
 # Integer scale yields exact rational comparisons without slow Fraction inner loops.
 scale=summary['scale_units_per_minute'];dt={(a,b):int((Fraction(0) if a==b else dist.get((a,b),INF))*scale) for a in refs for b in refs}
 adj=[[j for j in range(n) if i!=j and ee[i]*scale+dt[to[i],fr[j]]<=ss[j]*scale] for i in range(n)]
 digest=hashlib.sha256(json.dumps(adj,separators=(',',':')).encode()).hexdigest();assert digest==c['graph_sha256'];assert sum(map(len,adj))==c['edge_count']
 matches=c['matching'];cl=set(c['minimum_vertex_cover_left']);cr=set(c['minimum_vertex_cover_right']);assert len({u for u,v in matches})==len(matches)==len({v for u,v in matches});assert all(v in adj[u] for u,v in matches);assert all(u in cl or v in cr for u,vs in enumerate(adj) for v in vs);assert len(cl)+len(cr)==len(matches)
 paths=c['paths'];assert sorted(x for path in paths for x in path)==list(range(n));assert all(v in adj[u] for path in paths for u,v in zip(path,path[1:]));assert len(paths)==n-len(matches)==c['minimum_path_cover']
 anti=c['antichain']
 if c['graph_kind']=='closure_relaxation':
  assert anti is not None and len(set(anti))==len(anti)==len(paths)
  reach=[0]*n
  for u in sorted(range(n),key=lambda i:ss[i],reverse=True):
   for v in adj[u]:assert ss[v]>ss[u];reach[u]|=(1<<v)|reach[v]
  bits=sum(1<<u for u in anti);assert all(reach[u]&bits==0 for u in anti)
  if any(reach[u]!=sum(1<<v for v in adj[u]) for u in range(n)):nontransitive+=1
 verified.append({'case_id':cid,'group':c['group'],'graph_kind':c['graph_kind'],'minimum':len(paths),'graph_sha256':digest})
assert {(x['case_id'],x['group'],x['graph_kind']) for x in verified}=={(case,group,kind) for case in inputs for group in ['18E1','18E2','mixed'] for kind in ['direct','closure_relaxation']}
closure_min={(x['case_id'],x['group']):x['minimum'] for x in verified if x['graph_kind']=='closure_relaxation'}
from collections import Counter
mixed_gaps=[]
for case,u in upper.items():
 assert all(closure_min[case,g]==u['group_counts'][g] for g in ['18E1','18E2'])
 mixed_gaps.append(u['continuous_feasible_fleet']-closure_min[case,'mixed'])
assert sha(B.parent/'execution/audited_chain_results.csv')==summary['source_table_sha256']
assert sha(B/'time_only_vsp.py')==summary['script_sha256']
result={'separated_lower_equals_continuous_upper_cases':len(upper),'mixed_upper_minus_lower_counts':dict(Counter(mixed_gaps)),'status':'VERIFIED_WITH_EVENT_UPPER_BOUND_QUALIFICATION','certificates_verified':len(verified),'closure_antichains_verified_by_reachability':sum(c['graph_kind']=='closure_relaxation' for c in certs),'closure_graphs_not_transitive':nontransitive,'input_hashes_verified':len(inputs),'fixed_duty_witnesses_linked':len(checked_duties),'all_case_continuous_upper_bounds':upper,'certificates':verified,'source_hashes':{name:sha(B/name) for name in ['time_only_vsp.py','summary.json','certificates.json.gz','per_case.csv']},'verifier_sha256':sha(__file__),'elapsed_seconds':time.monotonic()-started,'qualifications':['Antichains checked by full DAG reachability, not assumed from reference triangle closure.','F4 upper bound is continuous physical charging, not event-grid representability.','No new solver calls and no cluster submissions.']}
(B/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['all_case_continuous_upper_bounds','certificates']}))
