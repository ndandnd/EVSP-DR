"""Fresh-start controls with measured cumulative ancestry budgets."""
from pathlib import Path
import argparse,datetime,fcntl,hashlib,json,math,os,resource,shutil,signal,socket,subprocess,sys,time
B=Path('/home/nc437/ladder-lite/cumulative_budget_20260913')
D=Path('/share/scaglione/nc437/evsp-dr/cumulative_budget_20260913')
CODE=Path('/home/nc437/ladder-lite/full_pool_recovery_20260912/code')
COMMIT='e091a4dba549510238507ef5e5367abea958bd30'
MIP_COMMIT='871d057e1067411f09581e37d78f7c1ca43f68bb'
MIP=Path('/home/nc437/ladder-lite/execution')/MIP_COMMIT
PY='/home/nc437/evsp_env/bin/python';S='/usr/local/slurm/slurm-25.05.5/bin/'

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def git(*args,code=CODE):return subprocess.check_output(['git','-C',str(code),*args],text=True).strip()
def code_check():
 for code,pin in [(CODE,COMMIT),(MIP,MIP_COMMIT)]:
  assert git('rev-parse','HEAD',code=code)==pin
  assert not git('status','--porcelain','--untracked-files=no',code=code)
  assert subprocess.run(['git','-C',str(code),'symbolic-ref','-q','HEAD'],capture_output=True).returncode!=0

def prepare():
 assert not (B/'manifest.json').exists(),'Do not overwrite a prepared campaign'
 code_check();D.mkdir(parents=True,exist_ok=True);(D/'cases').mkdir(exist_ok=True)
 (B/'cases').symlink_to(D/'cases',target_is_directory=True);(B/'logs').mkdir(exist_ok=True)
 audit=read(B/'audit/targets.json');cases={};compat=read(B/'cache_compatibility.json')
 for cid,src in sorted(audit['targets'].items()):
  c=dict(src);p=B/'inputs'/cid;p.mkdir(parents=True,exist_ok=False)
  assert sha(src['input_remote_path'])==src['input_sha256']==sha(CODE/'data'/src['csv'])
  for path,digest in zip(src['ancestry_paths'],src['ancestry_sha256']):assert sha(path)==digest
  old=Path(src['cache_path']);meta=read(src['cache_manifest_path']);assert sha(src['cache_manifest_path'])==src['cache_manifest_sha256']
  assert meta['identity']['instance_sha256']==src['input_sha256'] and meta['pickle_sha256']==src['cache_sha256']
  # Keep original producer evidence. Hardlinks cost no additional graph storage.
  shutil.copy2(src['cache_manifest_path'],p/'producer_manifest.json')
  os.link(old,p/'network.pkl');consumer=json.loads(json.dumps(meta))
  if meta['identity']['git_commit']!=COMMIT:
   producer=git('show',meta['identity']['git_commit']+':src/event_pricer_network.py')+'\n'
   assert hashlib.sha256(producer.encode()).hexdigest()==compat['reference_network_sha256']
   consumer['producer_identity']=dict(meta['identity']);consumer['identity']['git_commit']=COMMIT
   consumer['consumer_compatibility_audit']=str(B/'cache_compatibility.json')
  save(p/'network.pkl.manifest.json',consumer)
  c.update(cache=str(p/'network.pkl'),consumer_cache_manifest_sha256=sha(p/'network.pkl.manifest.json'))
  assert c['fresh_graph_sensitivity_budget_s']>=c['fresh_primary_budget_s']>0
  c['base_allocation_s']=c['fresh_primary_budget_s']+3600
  c['extra_allocation_s']=c['fresh_graph_sensitivity_budget_s']-c['fresh_primary_budget_s']+3600
  cases[cid]=c
 settings=dict(master_sense='cover',battery_kwh=240,charge_kw=240,reserve_kwh=0,terminal_floor=None,shared_station_capacity=False,tariff='flat',charge_start_fee=5,bus_coefficient=100000,soc_step_kwh=2.5,block_min=5,columns_per_iter=30,rc_epsilon=1e-4,max_iters=50000,fixed_sequence_index=True,skip_unused_incidence=False,initial_pool='singletons',inherited_columns=False,mip_s=3600,stage1_s=1800,stage2_fleet='<= validated incumbent')
 v=dict(schema='evsp-cumulative-budget-v1',prepared_utc=now(),cases=cases,execution_commit=COMMIT,mip_execution_commit=MIP_COMMIT,settings=settings,audit_sha256=sha(B/'audit/targets.json'),ancestry_sha256=sha(B/'audit/ancestry.json'),resources=dict(cg_cpus=8,cg_mem='96G',mip_cpus=8,mip_mem='24G',mip_allocation_s=7200,independent_primary_cases=len(cases),partition='default_partition',exclude='scaglione-compute-01'),tooling_sha256={n:sha(B/n) for n in ['campaign.py','worker.sub']},static_sha256={n:sha(CODE/'data'/n) for n in ['Ref_dict.csv','par_ref_dhd.csv','hourly_prices_flat.csv']},policy_sha256=sha(B.parent/'SCAGLIONE_RESOURCE_POLICY.md'),storage_root=str(D),interpretation=['Primary budget is ceil(sum native CG wall_s over actual k2..k ancestors including target).','Cache loading and inherited-route checking are inside native times. Queue waits and intermediate MIPs are not included.','Larger budget additionally credits prior-instance original graph construction; target graph is common to both methods.','Historical mixed revisions and hardware mean this is a retrospective budget control, not a randomized pure code speedup.','A primary certificate ends fresh CG early. Larger-budget endpoint then shares that result; do not duplicate its computation.','Only primary wall-limit stops continue at the same k from copied checkpoints. Resume overhead consumes the larger cumulative budget.','One identical one-hour MIP per distinct usable endpoint pool; primary MIP does not feed continuation.','Native CG time is serial accumulated elapsed time, not actual CPU usage. Measure new CPU user+system and report historical missingness.'])
 save(B/'manifest.json',v);print(json.dumps({'prepared_cases':len(cases),'primary_total_h':sum(c['fresh_primary_budget_s'] for c in cases.values())/3600,'max_primary_h':max(c['fresh_primary_budget_s'] for c in cases.values())/3600,'max_extended_h':max(c['fresh_graph_sensitivity_budget_s'] for c in cases.values())/3600}))

def cg_args(c,out,budget):
 return [PY,str(CODE/'src/exact_pricer_expanded.py'),'--csv',c['csv'],'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--event-network-cache',c['cache'],'--event-network-cache-mode','require','--fixed-sequence-index','--soc-step','2.5','--block-min','5','--max-iters','50000','--columns_per_iter','30','--column-selection','reduced_cost','--column-diversity-weight','0','--column-candidate-multiplier','4','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s',str(budget),'--checkpoint-every','25','--g-kwh','240','--charge-kw','240','--min-soc-frac','0','--phase-telemetry',str(out)+'.phases.jsonl','--gurobi-log',str(out)+'.gurobi.log','--out',str(out)]

def run_process(argv,path,limit,env=None):
 path=Path(path);path.mkdir(parents=True,exist_ok=False);record=dict(argv=argv,started_utc=now(),host=socket.getfqdn(),status='running',watchdog_s=limit)
 save(path/'execution.json',record);signals=[];proc=None
 def forward(sig,frame):
  signals.append(sig)
  if proc and proc.poll() is None:
   try:os.killpg(proc.pid,sig)
   except ProcessLookupError:pass
 previous={sig:signal.signal(sig,forward) for sig in [signal.SIGTERM,signal.SIGINT]}
 start=time.monotonic();usage=resource.getrusage(resource.RUSAGE_CHILDREN);timed=False
 try:
  with (path/'stdout.log').open('xb') as out,(path/'stderr.log').open('xb') as err:
   proc=subprocess.Popen(argv,cwd=CODE,env=env,stdout=out,stderr=err,start_new_session=True)
   try:rc=proc.wait(timeout=max(1,limit))
   except subprocess.TimeoutExpired:
    timed=True;os.killpg(proc.pid,signal.SIGTERM)
    try:rc=proc.wait(timeout=60)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
 finally:
  for sig,handler in previous.items():signal.signal(sig,handler)
 u=resource.getrusage(resource.RUSAGE_CHILDREN)
 record.update(returncode=rc,ended_utc=now(),wall_s=time.monotonic()-start,user_cpu_s=u.ru_utime-usage.ru_utime,system_cpu_s=u.ru_stime-usage.ru_stime,children_maxrss_kib=u.ru_maxrss,signals=signals,watchdog=timed,status='interrupted' if signals else 'watchdog' if timed else 'finished' if rc==0 else 'failed')
 save(path/'execution.json',record)
 if signals or timed or rc:raise RuntimeError('Solver execution did not complete: '+json.dumps(record))
 return record

def usable(v):return (v.get('final') or {}).get('artificials')==0 and (v.get('final') or {}).get('iter',0)>0

def copy_checkpoint(source,out):
 source=Path(source);v=read(source);assert (v.get('provenance') or {}).get('git_commit')==COMMIT
 journal=Path(v['columns_journal']);assert journal.is_file()
 shutil.copy2(source,out);shutil.copy2(journal,str(out)+'.columns.jsonl')
 iters=Path(str(source)+'.iters.csv')
 if iters.exists():shutil.copy2(iters,str(out)+'.iters.csv')
 return dict(source=str(source),status_sha256=sha(source),journal_sha256=sha(journal),native_elapsed_s=v['wall_s'])

def publish(stage,result):
 save(stage/'completion.json',result)

def register_mip(cid,arm,out):
 registry=B.parent/'mip_preemption_study_20260911/registry.json'
 with registry.with_suffix('.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);v=read(registry);job=os.environ['SLURM_JOB_ID']
  if job not in {x['job_id'] for x in v['cases']}:
   v['cases'].append(dict(job_id=job,case_id=cid+'_'+arm,cohort='default_cumulative_fresh_3600',solver_budget_s=3600,result_path=str(out),attempt_tag='cumulative_budget_20260913',requeue=True));save(registry,v)

def worker(mode,cid):
 v=read(B/'manifest.json');c=v['cases'][cid];root=B/'cases'/cid;stage=root/mode
 token=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');a=stage/token;a.mkdir(parents=True,exist_ok=False)
 state=dict(case_id=cid,stage=mode,attempt=token,started_utc=now(),manifest_sha256=sha(B/'manifest.json'),status='preflight')
 save(a/'state.json',state)
 try:
  code_check()
  for p,h in v['tooling_sha256'].items():assert sha(B/p)==h
  for p,h in v['static_sha256'].items():assert sha(CODE/'data'/p)==h
  assert sha(CODE/'data'/c['csv'])==c['input_sha256']
  assert sha(c['cache']+'.manifest.json')==c['consumer_cache_manifest_sha256']
  if (stage/'completion.json').exists():
   old=read(stage/'completion.json')
   if old.get('result_path'):assert sha(old['result_path'])==old['result_sha256']
   save(a/'state.json',{**state,'status':'already_complete','completion':old});return
  if mode in ['base','extra']:
   budget=c['fresh_primary_budget_s'] if mode=='base' else c['fresh_graph_sensitivity_budget_s']
   out=a/'cg.json';argv=cg_args(c,out,budget);checkpoint=None
   earlier=sorted((p for p in stage.glob('*/cg.json') if p.parent!=a),key=lambda p:p.stat().st_mtime,reverse=True)
   if earlier:checkpoint=earlier[0]
   if mode=='extra' and checkpoint is None:
    base=read(root/'base/completion.json')
    prior=read(base['result_path'])
    assert sha(base['result_path'])==base['result_sha256'] and sha(prior['columns_journal'])==base['journal_sha256']
    if prior.get('certified_rc_optimal'):
     publish(stage,{**state,'status':'shared_primary_certificate','usable':True,'result_path':base['result_path'],'result_sha256':base['result_sha256'],'journal_sha256':base['journal_sha256'],'budget_s':budget,'optimization_run':False});return
    if prior.get('stop_reason')!='wall_limit':
     publish(stage,{**state,'status':'skipped','usable':False,'reason':'primary_stopped_for_'+str(prior.get('stop_reason'))});return
    checkpoint=Path(base['result_path'])
   if checkpoint:
    state['resume']=copy_checkpoint(checkpoint,out);argv+=['--resume']
   state.update(status='cg_running',budget_s=budget);save(a/'state.json',state)
   offset=(state.get('resume') or {}).get('native_elapsed_s',0)
   execution=run_process(argv,a/'process',max(300,budget-offset+1800))
   value=read(out);assert not value.get('inherited_event_pool_status_sha256') and not value.get('validated_seed_routes_sha256')
   assert value.get('initial_pool')=='singletons';ok=usable(value)
   result={**state,'status':'finished','usable':ok,'result_path':str(out),'result_sha256':sha(out),'journal_sha256':sha(value['columns_journal']),'execution':execution,'native_wall_s':value.get('wall_s'),'budget_overshoot_s':max(0,value.get('wall_s',0)-budget),'certified':value.get('certified_rc_optimal'),'stop_reason':value.get('stop_reason'),'optimization_run':True,'ended_utc':now()}
   publish(stage,result);save(a/'state.json',result)
  elif mode in ['mip_base','mip_extra','mip_warm']:
   arm=mode.replace('mip_','')
   if arm=='warm':
    source=Path(c['status_path']);assert sha(source)==c['status_sha256'];historical=read(source)
    cg=dict(usable=usable(historical),result_path=str(source),result_sha256=c['status_sha256'],journal_sha256=sha(historical['columns_journal']))
   else:cg=read(root/arm/'completion.json')
   if not cg.get('usable'):
    publish(stage,{**state,'status':'skipped','reason':cg.get('reason','no_usable_pool'),'optimization_run':False});return
   if arm=='extra' and cg.get('status')=='shared_primary_certificate':
    base=read(root/'mip_base/completion.json');publish(stage,{**state,'status':'shared_primary_mip','optimization_run':False,'shared_completion':str(root/'mip_base/completion.json'),'result_path':base.get('result_path'),'result_sha256':base.get('result_sha256')});return
   source=Path(cg['result_path']);value=read(source);assert sha(source)==cg['result_sha256'] and sha(value['columns_journal'])==cg['journal_sha256']
   out=a/'result.json';register_mip(cid,arm,stage/'result.json')
   env=os.environ.copy();env.update(EVSP_EXPECTED_COMMIT=MIP_COMMIT,EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=cg['result_sha256'],EVSP_MIP_EXPECTED_JOURNAL_SHA256=cg['journal_sha256'])
   argv=[PY,str(MIP/'src/run_exact_pool_mip.py'),'--result',str(source),'--data-dir',str(CODE/'data'),'--reference-data-dir',str(CODE/'data'),'--cover','--two-stage','--timelimit','3600','--stage1-timelimit','1800','--threads','8','--mipgap','0.0001','--gurobi-log',str(a/'gurobi.log'),'--out',str(out)]
   state.update(status='mip_running',source_status_sha256=cg['result_sha256'],source_journal_sha256=cg['journal_sha256']);save(a/'state.json',state)
   execution=run_process(argv,a/'process',6300,env);result=read(out);assert result.get('physical_replay_validated') is True
   published=stage/'result.json';temporary=stage/('result.link.'+str(os.getpid()));temporary.symlink_to(out.resolve());temporary.replace(published)
   final={**state,'status':'finished','optimization_run':True,'result_path':str(out),'result_sha256':sha(out),'execution':execution,'ended_utc':now()};publish(stage,final);save(a/'state.json',final)
  else:raise ValueError(mode)
 except Exception as exc:
  state.update(status='execution_failed',error=repr(exc),ended_utc=now());save(a/'state.json',state);raise

def submit():
 v=read(B/'manifest.json');validation=read(B/'validation.json');assert validation['status']=='passed'
 assert not (B/'jobs.json').exists(),'Do not duplicate existing submissions'
 jobs=[];mapping={}
 for cid,c in v['cases'].items():
  row={}
  for mode in ['base','mip_warm','mip_base','extra','mip_extra']:
   dep=[] if mode in ['base','mip_warm'] else [row['base']] if mode in ['mip_base','extra'] else [row['extra'],row['mip_base']]
   secs=c['base_allocation_s'] if mode=='base' else c['extra_allocation_s'] if mode=='extra' else 7200
   minutes=math.ceil(secs/60);wall=f'{minutes//60}:{minutes%60:02d}:00'
   argv=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem='+('24G' if mode.startswith('mip') else '96G'),'--time='+wall,'--requeue','--kill-on-invalid-dep=yes','--job-name=cum_'+cid+'_'+mode,'--output='+str(B/'logs/%x_%j.out'),'--error='+str(B/'logs/%x_%j.err')]
   if dep:argv+=['--dependency=afterok:'+':'.join(dep)]
   argv+=[str(B/'worker.sub'),mode,cid]
   job=subprocess.check_output(argv,text=True).strip().split(';')[0];assert job.isdigit();row[mode]=job
   record=dict(job_id=job,case_id=cid,mode=mode,dependencies=dep,argv=argv,submitted_utc=now());jobs.append(record);save(B/'jobs.json',jobs)
   control=subprocess.check_output([S+'scontrol','show','job',job,'-o'],text=True);assert 'ExcNodeList=scaglione-compute-01' in control and 'Partition=default_partition' in control
   record['scontrol']=control;save(B/'jobs.json',jobs)
  mapping[cid]=row;save(B/'case_jobs.json',mapping)
 print(json.dumps({'cases':len(mapping),'independent_primary_cg':len(mapping),'submitted_stage_jobs':len(jobs)}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','submit']);p.add_argument('mode',nargs='?');p.add_argument('case',nargs='?');a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='submit':submit()
 else:worker(a.mode,a.case)
