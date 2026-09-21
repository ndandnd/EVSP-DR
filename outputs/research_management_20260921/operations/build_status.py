from pathlib import Path
import json,csv,hashlib,re,collections,sys
r=Path(__file__).parent;s=Path(sys.argv[1]) if len(sys.argv)>1 else sorted(r.glob('snapshot_*'))[-1];d=json.load(open(s/'snapshot.json'));q={x.split('|')[0]:x.split('|') for x in d['squeue']['stdout'].splitlines()};jobs=json.load(open(r.parents[1]/'week_20260921/chain_extension_40/case_jobs.json'))
sc={};checks=[]
for line in d['scontrol']['stdout'].splitlines():
 v=dict(re.findall(r'(\w+)=(\S+)',line));j=v.get('JobId')
 if v.get('ArrayJobId') and '-' not in v.get('ArrayTaskId',''):j=v['ArrayJobId']+'_'+v['ArrayTaskId']
 sc[j]=v
progress={}
for x in d['graph_files']:
 if x['path'].endswith('progress.jsonl'):
  case=x['path'].split('/cases/')[1].split('/')[0]
  lines=[]
  for line in x.get('tail','').splitlines():
   try:lines.append(json.loads(line))
   except json.JSONDecodeError:pass
  if lines and (case not in progress or x['modified_unix']>progress[case]['modified_unix']):progress[case]=dict(lines[-1],modified_unix=x['modified_unix'],source_path=x['path'])
rows=[]
for case,j in jobs.items():
 chain,k=case.split('_k');k=int(k);deps=[j['cache']]+([jobs[f'{chain}_k{k-1}']['cg']] if k>33 else [])
 for stage in ['cache','cg','mip']:
  v=sc[j[stage]];assert v['ExcNodeList']=='scaglione-compute-01';assert v['Partition']=='default_partition'
  if stage=='cache':assert v['ArrayTaskThrottle']=='44'
  else:
   expected=deps if stage=='cg' else [j['cg']]
   assert all(f'afterok:{x}(' in v['Dependency'] for x in expected),(case,stage,expected,v['Dependency'])
 checks.append({'case':case,'dependencies_verified':True,'resources_policy_verified':True})
 rows.append({'case':case,**{f'{stage}_job':j[stage] for stage in j},**{f'{stage}_state':q[j[stage]][2] for stage in j},'pricing_certificate':'not yet available','pool_fleet_proof':'not yet available','physical_validation':'not yet available','target_attainment':'not yet available'})
with open(s/'case_status.csv','w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
preempted=[line.split('|') for line in d['sacct']['stdout'].splitlines() if '|PREEMPTED|' in line and '.batch' not in line and '.extern' not in line]
seconds=sum(sum(int(a)*b for a,b in zip(x[3].split(':'),[3600,60,1])) for x in preempted)
summary={'collected_utc':d['collected_utc'],'verified_case_count':len(checks),'unique_graphs':len({j['cache'] for j in jobs.values()}),'preempted_attempts':preempted,'preemption_lost_wall_s':seconds,'current_graph_progress':progress,'checks':checks,'all_active_dependencies_preserved':True,'all_cpu_jobs_exclude_reserved_node':True,'graph_endpoints_available':sum(x['path'].endswith('network.pkl') for x in d['graph_files'])}
(s/'campaign_audit.json').write_text(json.dumps(summary,indent=2));print({'cases':len(checks),'preemptions':len(preempted),'lost_minutes':seconds/60,'graph_peak_rss_range_GiB':[min(v.get('peak_rss_kib',0) for v in progress.values())/1048576,max(v.get('peak_rss_kib',0) for v in progress.values())/1048576]})
