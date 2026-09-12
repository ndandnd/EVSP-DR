from pathlib import Path
import sys,os,json,subprocess,hashlib,csv,time,shutil
BASE=Path('/home/nc437/ladder-lite/overnight_extension_20260912')
CODE=BASE/'code';PY='/home/nc437/evsp_env/bin/python'
MIP=Path('/home/nc437/ladder-lite/execution/871d057e1067411f09581e37d78f7c1ca43f68bb')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,p)
def mappings(csv_rel):
 rows=list(csv.DictReader((CODE/'data'/csv_rel).open()));return {int(r['count_trip_id']):int(float(r['Ordered_Trip_ID'])) for r in rows}
def run(args):
 print('RUN',args,flush=True);subprocess.run(args,cwd=CODE,check=True)
def main():
 mode,case_id=sys.argv[1:3];manifest=json.loads((BASE/'manifest.json').read_text());case=manifest['cases'][case_id]
 assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==manifest['execution_commit']
 assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'],text=True).strip()
 root=BASE/'cases'/case_id;root.mkdir(parents=True,exist_ok=True);status=root/'cg.json'
 assert sha(CODE/'data'/case['csv'])==case['input_sha256']
 attempt=f"{os.environ.get('SLURM_JOB_ID','manual')}_r{os.environ.get('SLURM_RESTART_COUNT','0')}"
 save(root/f'{mode}_{attempt}_start.json',{'mode':mode,'case':case,'job':attempt,'started_epoch':time.time(),'commit':manifest['execution_commit'],'partition':os.environ.get('SLURM_JOB_PARTITION'),'resources':{k:os.environ.get(k) for k in ['SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE']}})
 if mode=='mip':
  out=root/'mip'/attempt/'result.json';out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():raise RuntimeError('Refusing to overwrite an attempt')
  os.environ['EVSP_EXPECTED_COMMIT']=MIP.name
  os.environ['EVSP_REQUIRE_DETACHED']='1'
  os.environ['EVSP_MIP_EXPECTED_RESULT_SHA256']=sha(status)
  os.environ['EVSP_MIP_EXPECTED_JOURNAL_SHA256']=sha(json.loads(status.read_text())['columns_journal'])
  run([PY,str(MIP/'src/run_exact_pool_mip.py'),'--result',str(status),'--data-dir',str(CODE/'data'),'--reference-data-dir',str(CODE/'data'),'--cover','--two-stage','--timelimit','3600','--stage1-timelimit','1800','--threads','8','--mipgap','0.0001','--gurobi-log',str(out.with_suffix('.gurobi.log')),'--out',str(out)])
  result=json.loads(out.read_text());save(root/'mip_result.json',result);save(root/'mip_provenance.json',{'result_path':str(out),'result_sha256':sha(out),'status_sha256':sha(status),'execution_commit':'871d057e1067411f09581e37d78f7c1ca43f68bb','data_execution_commit':manifest['execution_commit']});return
 inherit=case.get('parent_status')
 if mode=='join':
  # Sequence-only import descriptor. It is NOT a solved master or a certified bound.
  global_stable_to_local={v:k for k,v in mappings(case['csv']).items()};seqs={};pieces=[];total=0
  for child in case['children']:
   child_root=BASE/'cases'/child;cs=json.loads((child_root/'cg.json').read_text());mi=json.loads((child_root/'mip_result.json').read_text());mapping=mappings(cs['csv'])
   count=mi.get('buses');assert isinstance(count,(int,float));total+=count
   pieces.append({'case':child,'fleet':count,'mip_sha256':sha(child_root/'mip_result.json'),'input_sha256':manifest['cases'][child]['input_sha256'],'local_to_global':{str(k):global_stable_to_local[v] for k,v in mapping.items()},'physical_replay_validated':mi.get('physical_replay_validated'),'selected_routes_source':str(child_root/'mip_result.json')})
   for line in Path(cs['columns_journal']).open():
    r=json.loads(line);trips=[global_stable_to_local[mapping[t]] for t in r['trips']];key=tuple(trips)
    if key not in seqs or r['cost']<seqs[key]['cost']:seqs[key]={'trips':trips,'cost':r['cost']}
  assert all(x['physical_replay_validated'] for x in pieces)
  save(root/'decomposed_solution.json',{'fleet_sum':total,'target_duties':32,'components':pieces,'shared_capacity_enforced':False,'scope':'disjoint component schedules, covering model; no full-model optimality claim','original_duty_routes_injected':False})
  journal=root/'sequence_import.jsonl';journal.write_text(''.join(json.dumps(r)+'\n' for r in seqs.values()))
  inherit=str(root/'sequence_import.json');save(inherit,{'csv':case['csv'],'trip_ids':sorted(global_stable_to_local.values()),'columns_journal':str(journal),'provenance':{'instance_sha256':case['input_sha256']},'certificate':False,'source_kind':'mapped_component_sequences_only'})
 if status.exists():raise RuntimeError('Refusing to overwrite existing CG case')
 args=[PY,str(CODE/'src/exact_pricer_expanded.py'),'--csv',case['csv'],'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--soc-step','2.5','--block-min','5','--max-iters','50000','--columns_per_iter','30','--column-selection','reduced_cost','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s',str(case['cg_seconds']),'--checkpoint-every','25','--g-kwh','240','--charge-kw','240','--min-soc-frac','0','--phase-telemetry',str(status)+'.phase-telemetry.jsonl','--gurobi-log',str(status)+'.gurobi.log','--out',str(status)]
 if inherit:
  args+=['--inherit-event-pool-from',inherit,'--inherit-event-pool-workers','8','--inherit-max-columns','512','--inherit-time-limit-s','900']
 cache=case.get('source_cache')
 if cache and Path(cache).exists():
  # Network implementation unchanged; retain original manifest and rebind only execution identity.
  original=Path(cache+'.manifest.json');payload=json.loads(original.read_text());assert payload['identity']['instance_sha256']==case['input_sha256']
  assert sha(CODE/'src/event_pricer_network.py')==manifest['event_network_sha256']
  local=root/'network.pkl';os.link(cache,local);save(root/'source_network_manifest.json',payload);payload['identity']['git_commit']=manifest['execution_commit'];save(str(local)+'.manifest.json',payload)
  args+=['--event-network-cache',str(local),'--event-network-cache-mode','require']
 run(args)
 save(root/f'{mode}_{attempt}_end.json',{'ended_epoch':time.time(),'status_sha256':sha(status),'stop_reason':json.loads(status.read_text()).get('stop_reason')})
if __name__=='__main__':main()
