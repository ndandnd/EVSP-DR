#!/usr/bin/env python3
"""Blind adjacency recombination: only archived priced-pool trip sequences and canonical trip times."""
import argparse,collections,csv,gzip,hashlib,json,pathlib,time

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x): pathlib.Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
 a=argparse.ArgumentParser();a.add_argument('--pool',required=True);a.add_argument('--instance',required=True);a.add_argument('--out',required=True);z=a.parse_args();t0=time.monotonic();out=pathlib.Path(z.out)
 rows=list(csv.DictReader(open(z.instance))); assert sha(z.instance)=='c1dbe1dd15d20979799c68d01430a5fd0eddc2ca3ced3f46bd8522123e6c480d'
 n=len(rows); assert n==173
 def minute(x): h,m=map(int,x.split(':'));return 60*h+m
 starts=[minute(x['Start1']) for x in rows];ends=[minute(x['End1']) for x in rows]
 arcs=collections.Counter();sequences=set();full=hashlib.sha256();statuses=collections.Counter();route_count=0
 with gzip.open(z.pool,'rb') as f:
  for line in f:
   full.update(line);r=json.loads(line);seq=tuple(r['trips']);route_count+=1
   assert len(seq)==len(set(seq)) and set(seq)<=set(range(n))
   statuses[r.get('physical_realization',{}).get('status','missing')]+=1
   assert r.get('physical_realization',{}).get('status')=='valid_event_time_realized'
   assert r['continuous_realization']['charge_kw']==240 and r['continuous_realization']['station_kw']=={'PARX':60}
   sequences.add(seq)
   for u,v in zip(seq,seq[1:]):
    assert ends[u]<=starts[v], (u,v)
    arcs[u,v]+=1
 assert full.hexdigest()=='5744645fa8fc62e3c79e3880a12b1388dda8e0342fe46bb6c700d61e98c86d0d' and route_count==39390
 # DFS augmenting paths are exact for bipartite maximum matching; fixed frequency/id tie break.
 adj={u:sorted((v for x,v in arcs if x==u),key=lambda v:(-arcs[u,v],v)) for u in range(n)}
 right={}
 def augment(u,seen):
  for v in adj[u]:
   if v in seen:continue
   seen.add(v)
   if v not in right or augment(right[v],seen):right[v]=u;return True
  return False
 for u in sorted(range(n),key=lambda u:(starts[u],u)):augment(u,set())
 left={u:v for v,u in right.items()}
 # Konig vertex cover gives a separately checkable optimality certificate.
 zl=set(range(n))-set(left);zr=set();todo=list(zl)
 while todo:
  u=todo.pop()
  for v in adj[u]:
   if left.get(u)==v or v in zr:continue
   zr.add(v)
   if v in right and right[v] not in zl:zl.add(right[v]);todo.append(right[v])
 coverl=set(range(n))-zl;coverr=zr
 assert len(coverl)+len(coverr)==len(right)
 assert all(u in coverl or v in coverr for u,v in arcs)
 paths=[]
 for s in sorted(set(range(n))-set(right),key=lambda u:(starts[u],u)):
  seq=[s]
  while seq[-1] in left:seq.append(left[seq[-1]])
  paths.append(seq)
 assert sorted(t for p in paths for t in p)==list(range(n)) and len(paths)==n-len(right)
 new=[tuple(p) not in sequences for p in paths]
 write(out/'paths.json',paths)
 write(out/'graph_certificate.json',{'edges':[[u,v,c] for (u,v),c in sorted(arcs.items())],'matching':sorted(map(list,left.items())),'min_vertex_cover_left':sorted(coverl),'min_vertex_cover_right':sorted(coverr)})
 result=dict(schema='blind-priced-adjacency-path-cover-v1',source_pool=str(pathlib.Path(z.pool).resolve()),source_pool_gzip_sha256=sha(z.pool),source_pool_uncompressed_sha256=full.hexdigest(),source_instance=str(pathlib.Path(z.instance).resolve()),source_instance_sha256=sha(z.instance),pool_routes=route_count,unique_pool_sequences=len(sequences),trip_count=n,edge_count=len(arcs),source_physical_status_counts=statuses,dag_checked_by_nonoverlapping_service_times=True,matching_size=len(right),minimum_paths=len(paths),maximum_matching_certified=True,min_vertex_cover_size=len(coverl)+len(coverr),path_lengths=list(map(len,paths)),paths_new_to_original_pool=sum(new),path_is_new_sequence=new,algorithm='exact augmenting-path bipartite matching; successor frequency descending then id; left order by start time then id',scope='minimum trip-path cover over observed passenger adjacency DAG; battery/charging/shared-capacity feasibility NOT inferred',code_sha256=sha(__file__),paths_sha256=sha(out/'paths.json'),graph_certificate_sha256=sha(out/'graph_certificate.json'),compute_seconds=time.monotonic()-t0,witness_data_read=False)
 write(out/'receipt.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
