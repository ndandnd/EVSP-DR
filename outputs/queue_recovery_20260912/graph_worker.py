from pathlib import Path
import os,sys,json,hashlib,subprocess,time,csv
B=Path('/home/nc437/ladder-lite/graph_recovery_20260912');C=B/'code';O=Path('/home/nc437/ladder-lite/overnight_extension_20260912');PY='/home/nc437/evsp_env/bin/python';MIP=Path('/home/nc437/ladder-lite/execution/871d057e1067411f09581e37d78f7c1ca43f68bb')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def mapping(rel):return {int(r['count_trip_id']):int(float(r['Ordered_Trip_ID'])) for r in csv.DictReader((C/'data'/rel).open())}
def make_seed(cid,v):
 case=v['cases'][cid];root=B/'cases'/cid;root.mkdir(parents=True,exist_ok=True);global_ids=mapping(case['csv']);reverse={v:k for k,v in global_ids.items()};seqs={};pieces=[];coverage=set();fleet=0
 for child in case['children']:
  cr=B/'cases'/child if child=='d00_g3' else O/'cases'/child
  cp=cr/'cg.json';mp=cr/'mip_result.json';cg=json.loads(cp.read_text());mi=json.loads(mp.read_text());assert mi.get('physical_replay_validated') is True
  ids=mapping(cg['csv']);selected=mi['selected_routes'];assert len(selected)==mi['buses'];fleet+=mi['buses']
  childcoverage=set()
  for r in selected:
   trips=[reverse[ids[int(t)]] for t in r['trips']];assert trips
   childcoverage.update(trips);key=tuple(trips)
   if key not in seqs or r['cost']<seqs[key]['cost']:seqs[key]={'trips':trips,'cost':r['cost']}
  assert childcoverage=={reverse[t] for t in ids.values()};assert not coverage.intersection(childcoverage);coverage.update(childcoverage)
  pieces.append({'case_id':child,'fleet':mi['buses'],'status':str(cp),'status_sha256':sha(cp),'mip':str(mp),'mip_sha256':sha(mp),'physical_replay_validated':True})
 assert coverage==set(global_ids)
 journal=root/'selected_sequences.jsonl';assert not journal.exists();journal.write_text(''.join(json.dumps(r)+'\n' for r in seqs.values()))
 descriptor=root/'selected_sequences.json';save(descriptor,{'csv':case['csv'],'trip_ids':sorted(global_ids),'columns_journal':str(journal),'provenance':{'instance_sha256':case['input_sha256']},'certificate':False,'source_kind':'mapped_component_integer_solution_sequences'})
 save(root/'decomposed_solution.json',{'fleet_sum':fleet,'target_duties':32,'components':pieces,'selected_sequence_count':len(seqs),'all_parent_trips_covered':True,'shared_capacity_enforced':False,'scope':'component integer solutions combined under baseline covering physics; no parent optimality claim; fixed-sequence replay follows','original_duty_routes_injected':False})
 return descriptor

def main():
 mode,cid=sys.argv[1:3];v=json.loads((B/'manifest.json').read_text());assert subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip()==v['execution_commit'];assert not subprocess.check_output(['git','-C',str(C),'status','--porcelain','--untracked-files=no'],text=True).strip()
 if mode=='seed-only':print(make_seed(cid,v));return
 root=B/'cases'/cid;root.mkdir(parents=True,exist_ok=True);status=root/'cg.json';attempt=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');start={'case_id':cid,'mode':mode,'attempt':attempt,'execution_commit':v['execution_commit'],'started_epoch':time.time()}
 if mode in ('cache','cg'):
  key=cid if mode=='cache' else 'd00_g3' if cid=='d00_g3' else 'parent32';cache=v['cache_builds'][key];rel=cache['csv'];seconds=v['cache_seconds'] if mode=='cache' else v['cases'][cid]['cg_seconds']
  out=root/('cache_status.json' if mode=='cache' else 'cg.json');assert not out.exists()
  args=[PY,str(C/'src/exact_pricer_expanded.py'),'--csv',rel,'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--event-network-cache',cache['cache'],'--event-network-cache-mode','build-or-load' if mode=='cache' else 'require','--fixed-sequence-index','--soc-step','2.5','--block-min','5','--max-iters','50000','--columns_per_iter','30','--column-selection','reduced_cost','--column-diversity-weight','0.0','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s',str(seconds),'--checkpoint-every','25','--g-kwh','240','--charge-kw','240','--min-soc-frac','0','--phase-telemetry',str(out)+'.phase-telemetry.jsonl','--gurobi-log',str(out)+'.gurobi.log','--out',str(out)]
  if mode=='cache':
   assert args[-2]=='--out'
   args=args[:-2]+['--event-network-cache-only']
  elif cid.startswith('join'):
   descriptor=root/'selected_sequences.json'
   if not descriptor.exists():descriptor=make_seed(cid,v)
   args+=['--inherit-event-pool-from',str(descriptor),'--inherit-event-pool-workers','8','--inherit-max-columns','0','--inherit-time-limit-s','0'];start['seed_descriptor_sha256']=sha(descriptor)
 elif mode=='mip':
  cg=json.loads(status.read_text());assert cg['final']['artificials']==0 and cg['final']['iter']>0
  os.environ['EVSP_EXPECTED_COMMIT']=v['mip_execution_commit'];os.environ['EVSP_REQUIRE_DETACHED']='1';os.environ['EVSP_MIP_EXPECTED_RESULT_SHA256']=sha(status);os.environ['EVSP_MIP_EXPECTED_JOURNAL_SHA256']=sha(cg['columns_journal'])
  out=root/'mip'/attempt/'result.json';out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
  start.update({'source_status_sha256':sha(status),'source_journal_sha256':sha(cg['columns_journal'])})
  args=[PY,str(MIP/'src/run_exact_pool_mip.py'),'--result',str(status),'--data-dir',str(C/'data'),'--reference-data-dir',str(C/'data'),'--cover','--two-stage','--timelimit','3600','--stage1-timelimit','1800','--threads','8','--mipgap','0.0001','--gurobi-log',str(out.with_suffix('.gurobi.log')),'--out',str(out)]
 else:raise ValueError(mode)
 start['argv']=args;save(root/f'{mode}_{attempt}_start.json',start);print('RUN',json.dumps(args),flush=True);r=subprocess.run(args,cwd=C);start.update({'returncode':r.returncode,'ended_epoch':time.time()});save(root/f'{mode}_{attempt}_end.json',start)
 if r.returncode:raise SystemExit(r.returncode)
 if mode=='mip':
  mi=json.loads(out.read_text());assert mi.get('physical_replay_validated') is True;save(root/'mip_result.json',mi);save(root/'mip_provenance.json',{'result_path':str(out),'result_sha256':sha(out),'status_sha256':sha(status),'execution_commit':v['mip_execution_commit']})
 if mode=='cg' and cid.startswith('join'):
  cg=json.loads(status.read_text());a=cg.get('inherited_event_pool_audit',{});d=json.loads((root/'decomposed_solution.json').read_text());save(root/'seed_replay_check.json',{'selected_sequence_count':d['selected_sequence_count'],'accepted':a.get('accepted_columns'),'rejected':a.get('rejected_columns'),'all_selected_sequences_replayed':a.get('accepted_columns')==d['selected_sequence_count'] and a.get('rejected_columns')==0})
if __name__=='__main__':main()
