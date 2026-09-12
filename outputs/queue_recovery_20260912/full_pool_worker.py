from pathlib import Path
import json,hashlib,subprocess,os,sys,time
B=Path('/home/nc437/ladder-lite/full_pool_recovery_20260912');C=B/'code';PY='/home/nc437/evsp_env/bin/python';MIP=Path('/home/nc437/ladder-lite/execution/871d057e1067411f09581e37d78f7c1ca43f68bb')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
mode,cid=sys.argv[1:3];v=json.loads((B/'manifest.json').read_text());c=v['cases'][cid];root=B/'cases'/cid;status=root/'cg.json';attempt=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0')
assert subprocess.check_output(['git','-C',str(C),'rev-parse','HEAD'],text=True).strip()==v['execution_commit']
assert not subprocess.check_output(['git','-C',str(C),'status','--porcelain','--untracked-files=no'],text=True).strip()
assert sha(C/'data'/c['csv'])==c['input_sha256']
record={'case_id':cid,'mode':mode,'attempt':attempt,'execution_commit':v['execution_commit'],'started_epoch':time.time(),'source_manifest_sha256':sha(B/'manifest.json')}
if mode=='cg':
 assert not status.exists() and not Path(str(status)+'.columns.jsonl').exists(),'exclusive CG attempt already has outputs'
 parent=Path(c['parent_status']);p=json.loads(parent.read_text());assert p['final']['artificials']==0 and p['final']['iter']>0,'parent has no usable terminal LP'
 assert sha(C/'data'/p['csv'])==p['provenance']['instance_sha256']
 record.update({'parent_status':str(parent),'parent_status_sha256':sha(parent),'parent_journal':p['columns_journal'],'parent_journal_sha256':sha(p['columns_journal'])})
 initial=v['initial_parents'][str(c['chain'])]
 if str(parent)==initial['status']:
  assert record['parent_status_sha256']==initial['status_sha256'] and record['parent_journal_sha256']==initial['journal_sha256']
 assert sha(c['cache'])==c['cache_sha256'] and sha(c['cache']+'.manifest.json')==c['cache_manifest_sha256']
 args=[PY,str(C/'src/exact_pricer_expanded.py'),'--csv',c['csv'],'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--event-network-cache',c['cache'],'--event-network-cache-mode','require','--fixed-sequence-index','--soc-step','2.5','--block-min','5','--max-iters','50000','--columns_per_iter','30','--column-selection','reduced_cost','--column-diversity-weight','0.0','--column-candidate-multiplier','4','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s',str(c['cg_seconds']),'--checkpoint-every','25','--g-kwh','240','--charge-kw','240','--min-soc-frac','0','--inherit-event-pool-from',str(parent),'--inherit-event-pool-workers','8','--inherit-max-columns','0','--inherit-time-limit-s','0','--phase-telemetry',str(status)+'.phase-telemetry.jsonl','--gurobi-log',str(status)+'.gurobi.log','--out',str(status)]
elif mode=='mip':
 p=json.loads(status.read_text());assert p['final']['artificials']==0 and p['final']['iter']>0
 os.environ['EVSP_EXPECTED_COMMIT']=v['mip_execution_commit'];os.environ['EVSP_REQUIRE_DETACHED']='1';os.environ['EVSP_MIP_EXPECTED_RESULT_SHA256']=sha(status);os.environ['EVSP_MIP_EXPECTED_JOURNAL_SHA256']=sha(p['columns_journal'])
 out=root/'mip'/attempt/'result.json';out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
 record.update({'source_status':str(status),'source_status_sha256':sha(status),'source_journal_sha256':sha(p['columns_journal']),'output':str(out)})
 args=[PY,str(MIP/'src/run_exact_pool_mip.py'),'--result',str(status),'--data-dir',str(C/'data'),'--reference-data-dir',str(C/'data'),'--cover','--two-stage','--timelimit','3600','--stage1-timelimit','1800','--threads','8','--mipgap','0.0001','--gurobi-log',str(out.with_suffix('.gurobi.log')),'--out',str(out)]
else:raise ValueError(mode)
record['argv']=args;save(root/f'{mode}_{attempt}_start.json',record);print('RUN',json.dumps(args),flush=True)
r=subprocess.run(args,cwd=C);record.update({'returncode':r.returncode,'ended_epoch':time.time()});save(root/f'{mode}_{attempt}_end.json',record)
if r.returncode:raise SystemExit(r.returncode)
if mode=='mip':
 result=json.loads(out.read_text());assert result.get('physical_replay_validated') is True
 save(root/'mip_result.json',result);save(root/'mip_provenance.json',{'result_path':str(out),'result_sha256':sha(out),'status_sha256':sha(status),'execution_commit':v['mip_execution_commit']})
