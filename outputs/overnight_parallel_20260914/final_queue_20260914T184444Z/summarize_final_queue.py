from pathlib import Path
import argparse,json,collections,re

def main(folder):
 b=Path(folder);v=json.loads((b/'snapshot.json').read_text());roster=v['roster'];rows=[]
 for line in v['queue_raw'].splitlines():
  r=dict(zip(['job_id','name','state','reason','partition','nodes','dependency','cpus','memory'],line.split('|')))
  known=roster.get(r['job_id'])
  if known:r.update(known)
  elif r['job_id'].startswith('537227'):r.update(family='held_historical',kind='held')
  elif r['name'].startswith('drD_'):r.update(family='prior_diagnostics',kind='mip' if r['name'].endswith('_mip') else 'cg')
  elif r['name'].startswith('cpb_'):r.update(family='capacity_boundary',kind='combined_workflow')
  else:r.update(family='other_project',kind='unknown')
  rows.append(r)
 active={r['job_id']:r for r in rows};accounting={}
 for line in v['sacct_raw'].splitlines():
  x=line.split('|')
  if len(x)>=8:accounting[x[0]]=dict(zip(['job_id','raw_job_id','state','exit_code','elapsed_s','cpus','memory','nodes'],x[:8]))
 evsp=[r for r in rows if r['family'] not in ('held_historical','other_project')]
 edges=[]
 for r in evsp:
  for typ,parent,condition in re.findall(r'(afterok|afterany):(\d+(?:_\d+)?)\(([^)]+)\)',r['dependency']):
   edges.append(dict(child=r['job_id'],type=typ,parent=parent,condition=condition,parent_queue_state=active.get(parent,{}).get('state'),parent_accounting_state=accounting.get(parent,{}).get('state')))
 missing=[e for e in edges if e['condition']=='unfulfilled' and not e['parent_queue_state'] and not e['parent_accounting_state']]
 invalid=[r for r in evsp if r['reason']=='DependencyNeverSatisfied']
 reserved=[r for r in evsp if 'scaglione-compute-01' in r['nodes'] or re.search(r'scaglione-compute-\[0?1(?:,|-|\])',r['nodes'])]
 family_counts={f:dict(collections.Counter((r['kind']+'_'+r['state']) for r in rows if r['family']==f)) for f in sorted({r['family'] for r in rows})}
 old=json.loads((b.parent/'status_20260914T172907Z/filesystem.json').read_text())['campaigns']['chain_extension_20260914']['cases']
 graphs=[]
 for cid,r in v['filesystem']['chain_extension_20260914'].items():
  p=r.get('latest_progress') or {};prior=old[cid].get('progress',{});n=p.get('finished_sources')
  graphs.append(dict(case_id=cid,published=r['graph_published'],phase=p.get('phase'),progress_mtime_utc=r.get('progress_mtime_utc'),age_s=r.get('progress_age_s'),finished_sources=n,total_sources=p.get('total_sources'),advanced_since_1746=bool(n is not None and n>prior.get('finished_sources',0)),elapsed_h=p.get('elapsed_s',0)/3600,peak_rss_gib=p.get('peak_rss_kib',0)/1024**2))
 s=dict(sampled_utc=v['sampled_utc'],evsp_queue_counts=dict(collections.Counter(r['state'] for r in evsp)),running_by_type=dict(collections.Counter(r['kind'] for r in evsp if r['state']=='RUNNING')),family_counts=family_counts,pending_by_reason=dict(collections.Counter(r['reason'] for r in evsp if r['state']=='PENDING')),held_historical_rows=[r for r in rows if r['family']=='held_historical'],other_project_rows=[r for r in rows if r['family']=='other_project'],reserved_node_violations=reserved,invalid_dependency_reasons=invalid,unknown_dependency_parents=missing,dependency_edges=edges,graphs=graphs,rows=rows,accounting=accounting,mutations_performed=False)
 phase_path=b/'capacity_phase.json'
 if phase_path.exists():
  phase=json.loads(phase_path.read_text());phasejobs={r['attempt'].split('_r')[0]:r for r in phase['rows']}
  for row in rows:
   if row['family']=='capacity_boundary' and row['state']=='RUNNING':
    evidence=phasejobs[row['job_id']]
    if evidence['cg_log_started'] and not evidence['mip_log_started']:row['kind']='cg';row['stage_evidence']=str(phase_path)
 s['family_counts']={f:dict(collections.Counter(r['kind']+'_'+r['state'] for r in rows if r['family']==f)) for f in sorted({r['family'] for r in rows})}
 run=[r for r in evsp if r['state']=='RUNNING'];s['running_by_type']=dict(collections.Counter(r['kind'] for r in run));s['running_allocated_cpus']=sum(int(r['cpus']) for r in run)
 def gib(r):
  match=re.fullmatch(r'([0-9.]+)([KMGT])([cn]?)',r['memory']);assert match,r
  val=float(match[1])*{'K':1/1024**2,'M':1/1024,'G':1,'T':1024}[match[2]]
  return val*int(r['cpus']) if match[3]=='c' else val
 s['running_requested_memory_gib']=sum(gib(r) for r in run);s['memory_scope']='Requested memory of active Slurm allocations, not measured RSS.'
 s['research_solver_dependency_waits']=sum(r['state']=='PENDING' and r['kind'] in ('cg','mip') for r in evsp);s['operational_gate_waits']=sum(r['state']=='PENDING' and r['kind']=='graph_recovery_gate' for r in evsp)
 s['accounting_by_campaign']={}
 for family in ['seed','prefix','decomposition_first','decomposition_second','graph_gates_v2']:
  counts=collections.Counter();validation=collections.Counter()
  for job,r in roster.items():
   if r['family']!=family:continue
   target=validation if r['case_id'].startswith('native_fixture') else counts
   target[r['kind']+'|'+accounting.get(job,{}).get('state','not_reported')]+=1
  s['accounting_by_campaign'][family]={'production':dict(counts),'validation':dict(validation)}
 (b/'summary.json').write_text(json.dumps(s,indent=2)+'\n');(b/'squeue.txt').write_text(v['queue_raw']);(b/'sacct.txt').write_text(v['sacct_raw']);print(json.dumps({k:s[k] for k in ['sampled_utc','evsp_queue_counts','running_by_type','family_counts','pending_by_reason','reserved_node_violations','invalid_dependency_reasons','unknown_dependency_parents']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');main(p.parse_args().folder)
