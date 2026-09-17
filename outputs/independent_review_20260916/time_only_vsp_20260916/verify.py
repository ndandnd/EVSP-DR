"""Independent certificate verification and inclusion of every archived positive LP route."""
import csv,gzip,json,hashlib
from pathlib import Path
from time_only_vsp import travel_data,rows,minutes,graph,digest,sha,B,ROOT

def main():
 refs,scale,direct,closure,resolve=travel_data();n=len(refs)
 assert all(closure[i][j]<=closure[i][k]+closure[k][j] for i in range(n) for j in range(n) for k in range(n))
 table={r['case_id']:r for r in rows(B.parent/'execution/audited_chain_results.csv')};master=ROOT/'outputs/chain_extension_20260913/inputs/sources/Par_VehicleDetails_Updated.csv';duty={int(r['Ordered_Trip_ID']):r['VehicleTask'] for r in rows(master) if r['Identifier']=='Regular'};instances={}
 for cid,r in table.items():
  p=ROOT/'outputs'/r['campaign']/'inputs'/(cid+'.csv');assert sha(p)==r['input_sha256'];ts=[]
  for i,v in enumerate(rows(p)):
   assert i==int(v['count_trip_id']);d=duty[int(v['Ordered_Trip_ID'])]
   ts.append(dict(ordered=int(v['Ordered_Trip_ID']),start=minutes(v['Start1']),end=minutes(v['End1']),start_ref=resolve(v['From1']),end_ref=resolve(v['To1']),group='18E1' if d.startswith('134') else '18E2'))
  assert all(t['end']>t['start'] for t in ts);instances[cid]=ts
 certificates=json.loads(gzip.decompress((B/'certificates.json.gz').read_bytes()));tested=0;antichains=0
 for c in certificates:
  ts=[t for t in instances[c['case_id']] if c['group']=='mixed' or t['group']==c['group']];matrix=closure if c['graph_kind']=='closure_relaxation' else direct;adj=graph(ts,matrix,scale);assert digest(adj)==c['graph_sha256'];assert [t['ordered'] for t in ts]==c['ordered_trip_ids'];nv=len(ts)
  match=c['matching'];cl=set(c['minimum_vertex_cover_left']);cr=set(c['minimum_vertex_cover_right']);assert len({a for a,b in match})==len({b for a,b in match})==len(match);assert all(b in adj[a] for a,b in match);assert all(a in cl or b in cr for a,vs in enumerate(adj) for b in vs);assert len(cl)+len(cr)==len(match);assert nv-len(match)==c['minimum_path_cover'];assert sorted(i for p in c['paths'] for i in p)==list(range(nv));assert len(c['paths'])==c['minimum_path_cover'];assert all(b in adj[a] for p in c['paths'] for a,b in zip(p,p[1:]))
  if c['graph_kind']=='closure_relaxation':
   anti=set(c['antichain']);assert len(anti)==c['minimum_path_cover']
   # Check pairwise non-reachability, not merely absence of direct edges.
   for origin in anti:
    reach=set();pending=list(adj[origin])
    while pending:
     v=pending.pop()
     if v in reach:continue
     reach.add(v);pending.extend(w for w in adj[v] if w not in reach)
    assert not (reach&anti),(c['case_id'],c['group'],origin)
   antichains+=1
  tested+=1
 archive=B.parent/'execution/mixed_group_lp_20260916/saved_final_supports.json.gz';support=json.loads(gzip.decompress(archive.read_bytes()));route_count=pair_count=0;case_rows=[];violations=[]
 for record in support:
  cid=record['case_id'];assert record['file_sha256']==table[cid]['bound_source_sha256'];ts=instances[cid];local_routes=local_pairs=0
  for ordinal,r in enumerate(record['final_lp']['positive_routes']):
   assert r['value']>0;local_routes+=1
   for a,b in zip(r['trips'],r['trips'][1:]):
    left,right=ts[a],ts[b];gap=(right['start']-left['end'])*scale;needed=closure[left['end_ref']][right['start_ref']];local_pairs+=1
    if needed>gap:violations.append(dict(case_id=cid,route_ordinal=ordinal,trip_pair=[a,b],lambda_value=r['value'],needed_scaled=needed,gap_scaled=gap))
  route_count+=local_routes;pair_count+=local_pairs;case_rows.append(dict(case_id=cid,positive_routes_checked=local_routes,adjacent_connections_checked=local_pairs,source_file_sha256=record['file_sha256']))
 assert not violations,violations[:10]
 report=dict(status='verified',certificate_graphs_checked=tested,closure_antichains_checked_for_pairwise_nonreachability=antichains,saved_lp_cases=len(support),saved_positive_routes=route_count,saved_route_adjacent_connections=pair_count,connection_violations=violations,certificate_sha256=sha(B/'certificates.json.gz'),support_archive_sha256=sha(archive),script_sha256=sha(__file__),per_case=case_rows)
 (B/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='per_case'}))
if __name__=='__main__':main()
