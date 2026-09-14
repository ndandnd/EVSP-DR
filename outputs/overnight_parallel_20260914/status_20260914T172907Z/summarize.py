from pathlib import Path
import json,collections,re,datetime
b=Path(__file__).resolve().parent;q=json.load(open(b/'queue.json'));fs=json.load(open(b/'filesystem.json'));rows=[]
for line in q['stdout'].splitlines():
 r=dict(zip(['job_id','name','state','reason','partition','nodes','dependency','cpus','memory'],line.split('|')))
 n=r['name'];r['group']='held_historical' if r['job_id'].startswith('537227') else 'seed_mip' if n.startswith('drSeed_') and n.endswith('_mip') else 'seed_cg' if n.startswith('drSeed_') else 'capacity_boundary' if n.startswith('cpb_') else 'extension_graph' if n=='drX_graphs' else 'extension_mip' if n.startswith('drX_') and n.endswith('_mip') else 'extension_cg' if n.startswith('drX_') else 'prior_mip' if n.startswith('drD_') else 'other_project'
 rows.append(r)
ids={r['job_id']:r for r in rows};edges=[]
for r in rows:
 for parent,condition in re.findall(r'afterok:(\d+(?:_\d+)?)\(([^)]+)\)',r['dependency']):edges.append(dict(child=r['job_id'],parent=parent,condition=condition,parent_state=ids.get(parent,{}).get('state')))
evsp=[r for r in rows if r['group'] not in ('held_historical','other_project')];running=[r for r in evsp if r['state']=='RUNNING']
graphs=[]
for cid,c in fs['campaigns']['chain_extension_20260914']['cases'].items():
 p=c.get('progress',{});ratio=p.get('finished_sources',0)/max(1,p.get('total_sources',0))
 graphs.append(dict(case_id=cid,fraction=ratio,elapsed_h=p.get('elapsed_s',0)/3600,linear_total_h=p.get('elapsed_s',0)/3600/ratio if ratio else None,progress_age_s=c.get('progress_age_s'),peak_rss_gib=p.get('peak_rss_kib',0)/1024**2))
def mem_g(r):
 s=r['memory'];v=float(s[:-1]);return v if s[-1]=='G' else v/1024 if s[-1]=='M' else v*1024 if s[-1]=='T' else None
s=dict(sampled_utc=q['sampled_utc'],directory_timestamp_is_parent_snapshot_not_live_sample=True,evsp_counts=dict(collections.Counter(r['state'] for r in evsp)),groups={g:dict(collections.Counter(r['state'] for r in rows if r['group']==g)) for g in sorted({r['group'] for r in rows})},allocated_running_cpus=sum(int(r['cpus']) for r in running),allocated_running_memory_gib=sum(mem_g(r) for r in running),memory_scope='Slurm minimum requested memory for running allocations (per-node suffix absent), not measured RSS.',dependency_edges=len(edges),unfulfilled_parent_absent=[e for e in edges if e['condition']=='unfulfilled' and e['parent_state'] is None],invalid_dependency_reasons=[r for r in evsp if r['reason']=='DependencyNeverSatisfied'],dependency_edges_detail=edges,graphs=graphs,rows=rows)
(b/'status.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({k:v for k,v in s.items() if k not in ['rows','graphs','dependency_edges_detail']},indent=2));print('graphs',min(g['fraction'] for g in graphs),max(g['fraction'] for g in graphs),min(g['linear_total_h'] for g in graphs),max(g['linear_total_h'] for g in graphs),max(g['progress_age_s'] for g in graphs))
