"""Exact combinatorial time-only VSP audit; local only, no external solver."""
from pathlib import Path
from fractions import Fraction
from collections import deque,Counter
import csv,gzip,hashlib,json,math,time
B=Path(__file__).resolve().parent;ROOT=B.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def rows(p):
 with Path(p).open() as f:return list(csv.DictReader(f))
def normal(v):
 v=str(v).strip()
 if v.lower() in {'','nan','none'}:return None
 return v[:-2] if v.endswith('.0') else v

def base(v):
 v=str(v).strip();a=v.rsplit('_',1)
 return a[0] if len(a)==2 and a[1].isdigit() else v

def minutes(v):
 h,m=map(int,v.split(':'));assert h>=0 and 0<=m<60;return 60*h+m

def travel_data():
 loc={}
 for r in rows(B/'inputs/Ref_dict.csv'):
  l,ref=normal(r['Location']),normal(r['Ref'])
  if l is not None and ref is not None:loc[l]=ref
 for l,ref in list(loc.items()):loc.setdefault(base(l),ref)
 pairs={}
 for r in rows(B/'inputs/par_ref_dhd.csv'):
  a,b=normal(r['Start Place']),normal(r['End Place'])
  if a is None or b is None or a==b or not normal(r['Base Duration']) or not normal(r['Energy used']):continue
  t=Fraction(r['Base Duration']);assert t>=0;key=tuple(sorted([a,b]))
  if key not in pairs or t<pairs[key]:pairs[key]=t
 refs=sorted({x for p in pairs for x in p});scale=math.lcm(*(x.denominator for x in pairs.values()));index={r:i for i,r in enumerate(refs)};INF=10**12;n=len(refs)
 direct=[[INF]*n for _ in range(n)]
 for i in range(n):direct[i][i]=0
 for (a,b),value in pairs.items():direct[index[a]][index[b]]=direct[index[b]][index[a]]=int(value*scale)
 closure=[r[:] for r in direct]
 for k in range(n):
  for i in range(n):
   for j in range(n):closure[i][j]=min(closure[i][j],closure[i][k]+closure[k][j])
 def resolve(v):
  raw=normal(v)
  for candidate in [raw,base(raw),normal(raw),normal(base(raw))]:
   if candidate in loc:return index[loc[candidate]]
   if candidate in index:return index[candidate]
  raise ValueError('Unresolved production location '+str(v))
 return refs,scale,direct,closure,resolve

def maximum_matching(adj):
 n=len(adj);left=[-1]*n;right=[-1]*n
 # Deterministic augmenting-path matching. Each augment increases size by one.
 for origin in range(n):
  seenL={origin};seenR=set();parent={};q=deque([origin]);free=None
  while q and free is None:
   u=q.popleft()
   for v in adj[u]:
    if v in seenR:continue
    seenR.add(v);parent[v]=u
    if right[v]<0:free=v;break
    nxt=right[v]
    if nxt not in seenL:seenL.add(nxt);q.append(nxt)
  if free is not None:
   v=free
   while v>=0:
    u=parent[v];prev=left[u];left[u]=v;right[v]=u;v=prev
 # Alternating reachability yields a minimum bipartite vertex cover.
 zl={u for u in range(n) if left[u]<0};zr=set();q=deque(zl)
 while q:
  u=q.popleft()
  for v in adj[u]:
   if left[u]==v or v in zr:continue
   zr.add(v)
   if right[v]>=0 and right[v] not in zl:zl.add(right[v]);q.append(right[v])
 coverL=set(range(n))-zl;coverR=zr;matching=[(u,v) for u,v in enumerate(left) if v>=0]
 assert len({v for _,v in matching})==len(matching)
 assert all(v in adj[u] for u,v in matching)
 assert len(coverL)+len(coverR)==len(matching)
 assert all(u in coverL or v in coverR for u,vs in enumerate(adj) for v in vs)
 paths=[]
 for start in range(n):
  if right[start]>=0:continue
  path=[];u=start
  while u>=0:path.append(u);u=left[u];assert len(path)<=n
  paths.append(path)
 assert sorted(t for p in paths for t in p)==list(range(n));assert len(paths)==n-len(matching)
 antichain=zl-zr
 anti_valid=len(antichain)==len(paths) and all(not(set(adj[u])&antichain) for u in antichain)
 return dict(matching=matching,minimum_vertex_cover_left=sorted(coverL),minimum_vertex_cover_right=sorted(coverR),paths=paths,antichain=sorted(antichain) if anti_valid else None,minimum_path_cover=n-len(matching),verified_matching_cover_equality=True)

def overlap(ts):
 events=[]
 for t in ts:events.extend([(t['start'],1),(t['end'],-1)])
 n=best=0
 for _,delta in sorted(events):n+=delta;best=max(n,best)
 return best

def graph(ts,matrix,scale):
 return [[j for j,b in enumerate(ts) if i!=j and (a['end']*scale+matrix[a['end_ref']][b['start_ref']]<=b['start']*scale)] for i,a in enumerate(ts)]

def main():
 started=time.monotonic();table=rows(B.parent/'execution/audited_chain_results.csv');master=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv';duty={int(r['Ordered_Trip_ID']):r['VehicleTask'] for r in rows(master) if r['Identifier']=='Regular'}
 refs,scale,direct,closure,resolve=travel_data();assert sha(B/'inputs/Ref_dict.csv')=='7bda0e1f439dc8bf5081499566eb2c6a0314190ef27294707f1403fd2c13e3a0';assert sha(B/'inputs/par_ref_dhd.csv')=='5993e922c671f053611635578b32a1be13bab87b3b5fd8c02b699b81fe0eb66c'
 # Every allowed production deadhead is >= its unrounded duration, and closure
 # is <= every same-matrix path. Verify all source arcs and charger detours.
 assert all(closure[i][j]<=direct[i][j] for i in range(len(refs)) for j in range(len(refs)))
 stations=['2190L_0','4808_0','PARX_1','3127L_0','7880C_0','JON_A_0'];stationrefs=[resolve(s) for s in stations]
 assert all(closure[i][j]<=direct[i][s]+direct[s][j] for i in range(len(refs)) for j in range(len(refs)) for s in stationrefs)
 out=[];certs=[];inputhashes={}
 for row in table:
  cid=row['case_id'];p=ROOT/'outputs'/row['campaign']/'inputs'/(cid+'.csv');assert sha(p)==row['input_sha256'];inputhashes[str(p)]=sha(p);ts=[]
  for r in rows(p):
   ordered=int(r['Ordered_Trip_ID']);d=duty[ordered];assert d.startswith(('133','134'));s,e=minutes(r['Start1']),minutes(r['End1']);assert e>s
   ts.append(dict(ordered=ordered,local=int(r['count_trip_id']),duty=d,group='18E1' if d.startswith('134') else '18E2',start=s,end=e,start_ref=resolve(r['From1']),end_ref=resolve(r['To1'])))
  assert len(ts)==int(row['trip_count']);baseout={k:row[k] for k in ['case_id','chain','target_buses','trip_count','fractional_route_weight','cg_pricing_certificate','input_sha256']};baseout['cohort']='k_minus_1' if abs(float(row['fractional_route_weight'])-int(row['target_buses'])+1)<1e-6 else 'k'
  for group in ['18E1','18E2','mixed']:
   sub=[t for t in ts if group=='mixed' or t['group']==group];baseout[group+'_giro_duties']=len({t['duty'] for t in sub});baseout[group+'_service_overlap']=overlap(sub)
   for kind,matrix in [('direct',direct),('closure_relaxation',closure)]:
    adj=graph(sub,matrix,scale);c=maximum_matching(adj);assert c['minimum_path_cover']>=baseout[group+'_service_overlap']
    if kind=='closure_relaxation':assert c['antichain'] is not None,'Closure graph must certify fractional path-cover bound'
    else:c['antichain']=None  # Direct graph need not be transitive; no LP dual claim.
    c.update(case_id=cid,group=group,graph_kind=kind,ordered_trip_ids=[t['ordered'] for t in sub],graph_sha256=digest(adj),edge_count=sum(map(len,adj)),input_sha256=row['input_sha256']);certs.append(c);baseout[group+'_'+kind+'_minimum']=c['minimum_path_cover'];baseout[group+'_'+kind+'_edges']=c['edge_count']
  baseout['segregated_closure_lp_fleet_lower_bound']=sum(baseout[g+'_closure_relaxation_minimum'] for g in ['18E1','18E2']);baseout['segregated_direct_minimum']=sum(baseout[g+'_direct_minimum'] for g in ['18E1','18E2']);baseout['group_separation_closure_premium']=baseout['segregated_closure_lp_fleet_lower_bound']-baseout['mixed_closure_relaxation_minimum'];baseout['segregated_closure_excludes_k_minus_1']=baseout['segregated_closure_lp_fleet_lower_bound']>int(row['target_buses'])-1;baseout['mixed_recorded_lp_weight_minus_time_lower_bound']=float(row['fractional_route_weight'])-baseout['mixed_closure_relaxation_minimum'];out.append(baseout)
 with (B/'per_case.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
 (B/'certificates.json.gz').write_bytes(gzip.compress(json.dumps(certs,separators=(',',':')).encode(),mtime=0));summaries=[]
 for group in ['18E1','18E2','mixed']:
  counts=Counter(r[group+'_giro_duties']-r[group+'_closure_relaxation_minimum'] for r in out)
  for gap,n in sorted(counts.items()):summaries.append(dict(group=group,giro_duties_minus_time_only_relaxation=gap,cases=n))
 with (B/'gap_distribution.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
 proof={'method':'Exact maximum bipartite matching and equal-size minimum vertex cover certify minimum vertex-disjoint DAG path cover n−matching. Closure graphs additionally supply an equally large antichain, certifying any nonnegative path-cover LP route weight.','production_inclusion':'Source location aliases, symmetric pairs and minimum duplicate duration match production build_problem. Closure over all reference locations uses exact rational unrounded durations. Every production direct/charger path consists of same reference-pair arcs with nonnegative waiting/charging; production stored arc travel uses ceil(duration). Closure never exceeds that path travel. Removing energy, depot, charging and maximum-wait restrictions only enlarges the route space. Every feasible production route maps to a path in closure time DAG. Therefore closure antichain/path-cover value is a valid lower bound on the production fleet and path-cover LP, within the homogeneous or group-separated model respectively.','direct_scope':'Direct-matrix minimum is an exact answer only for the explicitly direct-transition time-only model. Missing direct arcs or shorter charger detours mean it is NOT automatically a lower bound on production EV routes.','rounding':'Fraction arithmetic source durations scaled by exact common denominator; comparisons integer exact, no floating tolerances. Integer-minute trip times retain hours beyond24. No depot pull-out/return constraint imposed.','nonenergy_scope':'Gaps against EV results are descriptive; energy, charger/depot/idle restrictions, omitted waypoints, objective differences and incomplete EV pricing are not isolated.'}
 metadata=dict(cases=len(out),certified_graphs=len(certs),wall_seconds=time.monotonic()-started,external_solver_calls=0,cluster_submissions=0,scale_units_per_minute=scale,source_table_sha256=sha(B.parent/'execution/audited_chain_results.csv'),master_sha256=sha(master),ref_sha256=sha(B/'inputs/Ref_dict.csv'),deadhead_sha256=sha(B/'inputs/par_ref_dhd.csv'),script_sha256=sha(__file__),input_hashes=inputhashes,proof=proof,source_code={'audit_giro_known_columns.py':sha(B.parent/'execution/f4/pinned/src/audit_giro_known_columns.py'),'pricing_dp_og.py':sha(B.parent/'execution/f4/pinned/src/pricing_dp_og.py')},nine_cases=[r for r in out if r['cohort']=='k_minus_1'],direct_closure_different_cells=sum(r[g+'_direct_minimum']!=r[g+'_closure_relaxation_minimum'] for r in out for g in ['18E1','18E2','mixed']),segregated_bound_equals_k=sum(r['segregated_closure_lp_fleet_lower_bound']==int(r['target_buses']) for r in out),gap_distribution=summaries)
 (B/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({k:v for k,v in metadata.items() if k in ['cases','certified_graphs','wall_seconds','direct_closure_different_cells','segregated_bound_equals_k','gap_distribution']}));print(json.dumps([(r['case_id'],r['18E1_closure_relaxation_minimum'],r['18E2_closure_relaxation_minimum'],r['mixed_closure_relaxation_minimum'],r['segregated_closure_excludes_k_minus_1']) for r in metadata['nine_cases']]))
if __name__=='__main__':main()
