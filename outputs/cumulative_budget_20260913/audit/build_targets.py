from pathlib import Path
import json,math,hashlib,csv,io
P=Path(__file__).resolve().parent
m=json.loads((P/'ancestry.json').read_text());nodes=m['nodes'];targets={};table=[]
account=json.loads((P/'accounting.json').read_text())
allocation={r['JobID']:r for r in csv.DictReader(io.StringIO(account['sacct_stdout']),delimiter='|') if '.' not in r['JobID']}
assert len(allocation)==84

def seconds(v):
 days=0
 if '-' in v:
  d,v=v.split('-',1);days=int(d)
 parts=[float(x) for x in v.split(':')]
 return days*86400+sum(n*60**i for i,n in enumerate(parts[::-1]))
for chain,root in m['roots'].items():
 ancestry=[];p=root
 while p:
  assert p not in ancestry
  ancestry.append(p);p=nodes[p]['inherited_event_pool_audit'].get('source_status')
 ancestry.reverse()
 assert [nodes[p]['k'] for p in ancestry]==list(range(2,16))
 for k in [5,8,10,15]:
  paths=ancestry[:k-1];t=nodes[paths[-1]];native=sum(nodes[p]['wall_s'] for p in paths)
  allgraphs=sum(nodes[p]['network_metrics'].get('cache_original_build_s',0) for p in paths if nodes[p]['network_metrics'].get('cache_hit'))
  targetgraph=t['network_metrics'].get('cache_original_build_s',0);extra=allgraphs-targetgraph
  assert all(nodes[p]['network_metrics'].get('cache_hit') is True for p in paths)
  meta=t['cache_manifest'];cv=meta['value'];cid=f'c{chain}_k{k:02d}'
  targets[cid]={'id':cid,'chain':int(chain),'k':k,'status_path':t['path'],'status_sha256':t['sha256'],'csv':t['csv'],'input_remote_path':'/home/nc437/ladder-lite/full_pool_recovery_20260912/code/data/'+t['csv'],'input_sha256':t['provenance']['instance_sha256'],'cache_path':t['network_metrics']['cache_path'],'cache_sha256':cv['pickle_sha256'],'cache_hash_scope':'recorded producer manifest, not rehashed multi-GB pickle in this audit','cache_manifest_path':meta['path'],'cache_manifest_sha256':meta['sha256'],'cache_identity':cv['identity'],'ancestry_paths':paths,'ancestry_sha256':[nodes[p]['sha256'] for p in paths],'distinct_stage_count':len(paths),'native_cumulative_wall_s':native,'fresh_primary_budget_s':math.ceil(native),'ancestor_external_graph_build_s':extra,'target_external_graph_build_s':targetgraph,'all_external_graph_build_s':allgraphs,'fresh_graph_sensitivity_budget_s':math.ceil(native+extra),'sensitivity_scope':'extra ancestor graph build only; common target graph excluded','cumulative_import_s':sum((nodes[p]['inherited_event_pool_audit'] or {}).get('total_import_s',0) for p in paths),'intermediate_mip_budget_s':0,'actual_cpu_s':None,'allocated_cpu_s':None,'cpu_evidence_scope':'not inferred from wall_s; awaiting scheduler accounting','source_commits':sorted({nodes[p]['provenance']['git_commit'] for p in paths}),'target_certified_rc_optimal':t['certified_rc_optimal'],'target_settings':{x:t[x] for x in ['soc_step','block_min','g_kwh','charge_kw','min_soc_frac','master_sense','master_backend','time_model','columns_per_iter']}}
  accounting=[]
  for path in paths:
   stage=f"c{chain}_k{nodes[path]['k']:02d}";job=account['job_mapping'][stage]['job_id'];record=allocation[job]
   assert record['State']=='COMPLETED' and record['Start'].startswith('2026-09')
   accounting.append({'status_path':path,'job_id':job,'allocation':record})
  targets[cid].update(actual_cpu_s=sum(seconds(r['allocation']['TotalCPU']) for r in accounting),allocated_cpu_s=sum(int(r['allocation']['CPUTimeRAW']) for r in accounting),cpu_evidence_scope='Slurm allocation-level TotalCPU actual versus CPUTimeRAW allocated CPU-seconds, CG jobs only; external graph preparation excluded; do not sum .batch/.extern again',cg_accounting=accounting,warm_target_mip=t.get('warm_mip'),warm_target_mip_missing=t.get('warm_mip_error'))
  assert t['input_file_verified']['sha256']==targets[cid]['input_sha256']
  table.append({x:targets[cid][x] for x in ['id','chain','k','distinct_stage_count','native_cumulative_wall_s','fresh_primary_budget_s','ancestor_external_graph_build_s','target_external_graph_build_s','fresh_graph_sensitivity_budget_s','cumulative_import_s','target_certified_rc_optimal','actual_cpu_s','allocated_cpu_s']})
(P/'targets.json').write_text(json.dumps({'schema':'evsp-cumulative-budget-targets-v1','ancestry_evidence_sha256':hashlib.sha256((P/'ancestry.json').read_bytes()).hexdigest(),'panel_policy':'All six existing chains at k5,8,10,15 fixed by chain and size; no outcome filtering','targets':targets},indent=2)+'\n')
with (P/'budgets.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
print('24 targets; primary total h',sum(x['fresh_primary_budget_s'] for x in table)/3600,'sensitivity total h',sum(x['fresh_graph_sensitivity_budget_s'] for x in table)/3600)
