"""One-allocation FDL ladder; persistent stages, exact input/code identity, native CG resume."""
from pathlib import Path
import json,hashlib,subprocess,os,time,signal,shutil,sys
B=Path('/home/nc437/ladder-lite/review_frolunda_20260916');PY='/home/nc437/evsp_env/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
v=read(B/'manifest.json');CG=B/'cg_code';MIP=B/'mip_code';DATA=B/'data';case_root=B/'cases'
for tag,path in [('cg',CG),('mip',MIP)]:
 assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==v['code'][tag]['commit']
 assert not subprocess.check_output(['git','-C',str(path),'status','--porcelain','--untracked-files=no'],text=True).strip()
for file,h in v['tool_hashes'].items():assert sha(B/file)==h
for file,h in v['data_hashes'].items():assert sha(DATA/file)==h
job=os.environ.get('SLURM_JOB_ID','manual');restart=os.environ.get('SLURM_RESTART_COUNT','0');attempt=job+'_r'+restart
received=[];proc=None

def forward(sig,_):
 received.append(sig)
 if proc is not None and proc.poll() is None:
  try:os.killpg(proc.pid,sig)
  except ProcessLookupError:pass
for sig in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]:signal.signal(sig,forward)

def run(stage,cid,argv,limit,a,env=None):
 global proc
 if received:raise SystemExit(128+received[-1])
 a.mkdir(parents=True,exist_ok=False)
 record={'stage':stage,'case_id':cid,'attempt':attempt,'argv':argv,'manifest_sha256':sha(B/'manifest.json'),'started_epoch':time.time(),'status':'running','env_identity':{k:(env or os.environ).get(k) for k in ['EVSP_EXPECTED_COMMIT','EVSP_MIP_EXPECTED_RESULT_SHA256','EVSP_MIP_EXPECTED_JOURNAL_SHA256']}}
 save(a/'execution.json',record)
 with (a/'stdout.log').open('w') as out,(a/'stderr.log').open('w') as err:
  proc=subprocess.Popen(argv,cwd=CG,start_new_session=True,stdout=out,stderr=err,env=env)
  try:rc=proc.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:rc=proc.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
   record['watchdog_timeout']=True
 record.update(returncode=rc,ended_epoch=time.time(),signals=received,status='interrupted' if received else 'finished' if rc==0 and not record.get('watchdog_timeout') else 'failed')
 save(a/'execution.json',record)
 if received:raise SystemExit(128+received[-1])
 if rc or record.get('watchdog_timeout'):raise SystemExit(rc or 124)
 proc=None

def done_valid(p):
 if not p.exists():return None
 x=read(p)
 assert x['manifest_sha256']==sha(B/'manifest.json')
 for f,h in x['output_hashes'].items():assert sha(f)==h,(f,'changed completed artifact')
 return x

def cg_args(c,cache,graph=False,out=None,parent=None):
 args=['--csv',c['csv'],'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--event-network-cache',str(cache),'--event-network-cache-mode','build-or-load' if graph else 'require','--fixed-sequence-index','--soc-step','2.5','--block-min','5','--max-iters','50000','--columns_per_iter','30','--column-selection','reduced_cost','--column-diversity-weight','0.0','--column-candidate-multiplier','4','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s',str(v['graph_budget_s'] if graph else v['cg_budget_s']),'--checkpoint-every','25','--g-kwh','240','--charge-kw','240','--min-soc-frac','0']
 if graph:return args+['--event-network-cache-only']
 if parent:args+=['--inherit-event-pool-from',str(parent),'--inherit-event-pool-workers','8','--inherit-max-columns','0','--inherit-time-limit-s','0']
 return args+['--out',str(out),'--phase-telemetry',str(out)+'.phase-telemetry.jsonl','--gurobi-log',str(out)+'.gurobi.log']

parent=None
for cid,c in v['cases'].items():
 root=case_root/cid;root.mkdir(parents=True,exist_ok=True)
 cache_done=root/'cache_done.json';d=done_valid(cache_done)
 if not d:
  a=root/'cache'/attempt;cache=a/'network.pkl';args=[PY,str(B/'fdl_entry.py'),str(CG),str(DATA),*cg_args(c,cache,True)]
  run('cache',cid,args,v['graph_budget_s']+600,a)
  meta=read(str(cache)+'.manifest.json');assert meta['identity']['git_commit']==v['code']['cg']['commit'] and meta['identity']['instance_sha256']==c['input_sha256'];assert sha(cache)==meta['pickle_sha256']
  d={'manifest_sha256':sha(B/'manifest.json'),'cache':str(cache),'output_hashes':{str(cache):sha(cache),str(cache)+'.manifest.json':sha(str(cache)+'.manifest.json')}};save(cache_done,d)
 cache=Path(d['cache']);cg_done=root/'cg_done.json';d=done_valid(cg_done)
 if not d:
  a=root/'cg'/attempt;out=a/'cg.json';args=[PY,str(B/'fdl_entry.py'),str(CG),str(DATA),*cg_args(c,cache,out=out,parent=parent)]
  # run() creates attempt directory. Native checkpoints are copied immediately before spawn.
  previous=sorted((p for p in (root/'cg').glob('*/cg.json') if p.parent!=a),key=lambda p:p.stat().st_mtime)
  resume=None
  for old in reversed(previous):
   x=read(old);journal=Path(str(old)+'.columns.jsonl')
   if journal.exists():
    assert x['provenance']['git_commit']==v['code']['cg']['commit'] and x['provenance']['instance_sha256']==c['input_sha256'];resume=old;break
  if resume:
   # A small launcher copies validated native checkpoint files after run creates a.
   payload={'from':str(resume),'to':str(out),'source_sha256':sha(resume),'journal_sha256':sha(str(resume)+'.columns.jsonl')};save(root/'resume_'+attempt+'.json',payload)
   args=[PY,str(B/'resume_entry.py'),json.dumps(payload),*args[1:],'--resume']
  run('cg',cid,args,v['cg_budget_s']+1800,a)
  x=read(out);assert x.get('final',{}).get('artificials')==0 and x.get('iterations',0)>0
  journal=Path(x['columns_journal']);d={'manifest_sha256':sha(B/'manifest.json'),'result':str(out),'pricing_certified':x.get('certified_rc_optimal'),'stop_reason':x.get('stop_reason'),'output_hashes':{str(out):sha(out),str(journal):sha(journal)}};save(cg_done,d)
 status=Path(d['result']);parent=status
 mip_done=root/'mip_done.json';d=done_valid(mip_done)
 if not d:
  x=read(status);env=dict(os.environ,EVSP_EXPECTED_COMMIT=v['code']['mip']['commit'],EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=sha(status),EVSP_MIP_EXPECTED_JOURNAL_SHA256=sha(x['columns_journal']))
  a=root/'mip'/attempt;out=a/'result.json';args=[PY,str(MIP/'src/run_exact_pool_mip.py'),'--result',str(status),'--data-dir',str(DATA),'--reference-data-dir',str(DATA),'--cover','--two-stage','--timelimit','3600','--stage1-timelimit','1800','--threads','8','--mipgap','0.0001','--gurobi-log',str(a/'gurobi.log'),'--out',str(out)]
  run('mip',cid,args,6300,a,env)
  x=read(out);assert x.get('physical_replay_validated') is True
  save(mip_done,{'manifest_sha256':sha(B/'manifest.json'),'result':str(out),'source_cg_sha256':sha(status),'buses':x.get('buses'),'fleet_bound':x.get('fleet_bound'),'fleet_proven':x.get('fleet_proven'),'output_hashes':{str(out):sha(out)}})
 print('COMPLETED '+cid,flush=True)
save(B/'ladder_complete.json',{'manifest_sha256':sha(B/'manifest.json'),'completed_cases':list(v['cases']),'ended_epoch':time.time()})
