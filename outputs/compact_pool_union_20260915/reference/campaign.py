"""Prepare and submit six independent compute pool constructions and six MIPs."""
from pathlib import Path
import argparse,copy,datetime,fcntl,hashlib,importlib.util,json,math,os,shutil,subprocess
B=Path('/home/nc437/ladder-lite/parallel_pool_unions_20260914')
D=Path('/share/scaglione/nc437/evsp-dr/parallel_pool_unions_20260914')
OLD=B.parent/'overnight_diagnostics_20260914';CUM=B.parent/'cumulative_budget_20260913';S='/usr/local/slurm/slurm-25.05.5/bin/'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with t.open('w') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def prepare():
 assert not (B/'manifest.json').exists()
 prior=read(OLD/'manifest.json');selection=read(B/'selection.json')
 assert sha(OLD/'worker.py')==prior['tooling_sha256']['worker.py'];shutil.copy2(OLD/'worker.py',B/'worker.py')
 D.mkdir(parents=True,exist_ok=True);(D/'cases').mkdir(exist_ok=True)
 if not (B/'cases').exists():(B/'cases').symlink_to(D/'cases',target_is_directory=True)
 assert (B/'cases').resolve()==(D/'cases').resolve();(B/'logs').mkdir(exist_ok=True)
 sp=importlib.util.spec_from_file_location('logic',B/'union_logic.py');logic=importlib.util.module_from_spec(sp);sp.loader.exec_module(logic)
 cases={};checks=[]
 for base in selection['union_inputs']:
  sources=[];values=[]
  for campaign,cgname,mipname,arm in [(CUM,base+'/base',base+'/mip_base','original'),
      (OLD,base+'_c200',base+'_c200_mip','c200'),(OLD,base+'_complementary',base+'_complementary_mip','complementary')]:
   comp=read(campaign/'cases'/cgname/'completion.json');mc=read(campaign/'cases'/mipname/'completion.json')
   status=read(comp['result_path']);mp=read(mc['result_path']);pa=mp['physical_pool_audit']
   assert sha(comp['result_path'])==comp['result_sha256'] and sha(mc['result_path'])==mc['result_sha256']
   assert mp['source_result_sha256']==comp['result_sha256'] and mp['source_journal_sha256']==comp['journal_sha256']
   assert mp['physical_replay_validated'] and pa['rejected_columns']==pa['deterministically_repaired']==0
   sources.append(dict(arm=arm,status_path=str(Path(comp['result_path']).resolve()),status_sha256=comp['result_sha256'],
    journal_path=status['columns_journal'],journal_sha256=comp['journal_sha256'],
    mip_evidence_path=str(Path(mc['result_path']).resolve()),mip_evidence_sha256=mc['result_sha256'],
    source_mip_ordered_pool_sha256=pa['mip_ordered_pool_sha256']))
   values.append(status)
  logic.identities(values)
  template=copy.deepcopy(prior['cases'][base+'_c200_mip']);template.pop('source_case',None)
  config=Path('/home/nc437/ladder-lite/full_pool_recovery_20260912/code/src/config.py')
  static={**template['static_hashes'],str(config):sha(config)}
  buildid=base+'_union_build';mipid=base+'_union_mip'
  build=dict(id=buildid,kind='pool_construction',chain=template['chain'],target_k=template['target_k'],target_duties=template['target_k'],
   original_case=base,csv=template['csv'],data_dir=template['data_dir'],input_path=template['input_path'],input_sha256=template['input_sha256'],
   sources=sources,static_hashes=static,watchdog_s=3300,resources=dict(cpus=2,mem='8G',allocation_s=3600),
   baseline_scope=dict(shared_station_capacity=False,terminal_soc_floor=False,bus_coefficient=100000,charge_start_fee=5,configuration_bound_by=logic.PRODUCER),
   treatment='deterministic_union_of_existing_columns',optimization_run=False)
  cases[buildid]=build
  template.update(id=mipid,source_case=buildid,treatment='three_way_existing_column_union',source_artifact_kind='finite_pool_union',
   solver_budget_s=12600,stage1_budget_s=10800,watchdog_s=15300,resources=dict(cpus=8,mem='24G',allocation_s=16200),
   interpretation='Finite-pool union of identical-input original/c200/complementary columns, not a CG endpoint. Same12600/10800MIPbudget as individual-treatment long controls. Native full-pool replay must have no rejected/repaired columns before publishing union equivalence.')
  a=template['argv'];a[a.index('--timelimit')+1]='12600';a[a.index('--stage1-timelimit')+1]='10800';cases[mipid]=template
  checks.append(dict(case_id=base,status='passed',source_cases=3,source_physical_audits='zero rejected and repaired',journal_rehash_scope='compute construction before/after merge; nativeMIP before solve'))
 m=dict(schema='evsp-parallel-pool-unions-v1',prepared_utc=now(),cases=cases,selection=selection,selection_sha256=sha(B/'selection.json'),
  tooling_sha256={n:sha(B/n) for n in ['union_logic.py','union_worker.py','worker.py','worker.sub']},storage_root=str(D),
  baseline_scope=next(c['baseline_scope'] for c in cases.values() if c['kind']=='pool_construction'),
  execution_budgets=dict(build_watchdog_s=3300,build_allocation_s=3600,mip_total_s=12600,mip_fleet_s=10800),
  policy_sha256=sha(B.parent/'SCAGLIONE_RESOURCE_POLICY.md'),
  interpretation='Six independent pool-construction allocations; eachMIP depends only on itsown successful construction. Constructions are notCG/LPresults; no pricing certificate is generated. Allsource and actualunionMIP physicaladmission gates required.')
 save(B/'manifest.json',m)
 # Lightweight code/data checks; pool parsing/merging belongs to compute nodes.
 sp=importlib.util.spec_from_file_location('worker',B/'worker.py');w=importlib.util.module_from_spec(sp);sp.loader.exec_module(w)
 for c in cases.values():
  w.require_hash(c['input_path'],c['input_sha256'])
  for p,h in c['static_hashes'].items():w.require_hash(p,h)
  if c['kind']=='mip':
   w.check_code(c['source_code'],c['execution_commit'])
   a=w.expand_argv(c['argv'],'/validation/result.json','/validation','/validation/pool.json')
   assert a[a.index('--threads')+1]=='8' and a[a.index('--timelimit')+1]=='12600' and a[a.index('--stage1-timelimit')+1]=='10800'
 subprocess.run(['/home/nc437/evsp_env/bin/python','-m','unittest','discover','-s',str(B),'-p','test_union.py','-v'],check=True)
 save(B/'validation.json',dict(status='passed',manifest_sha256=sha(B/'manifest.json'),checks=checks,
  unit_tests=3,source_mip_admission_bindings=18,native_full_union_replay='Mandatory execution gate before everyMIP; not claimed to have run at preparation'))
 print(json.dumps({'prepared':12,'construction':6,'mip':6,'validation':'passed'}))

def submit():
 with (B/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  m=read(B/'manifest.json');v=read(B/'validation.json');assert v['status']=='passed' and v['manifest_sha256']==sha(B/'manifest.json')
  for n,h in m['tooling_sha256'].items():assert sha(B/n)==h
  jobs=read(B/'jobs.json') if (B/'jobs.json').exists() else [];done={j['case_id']:j['job_id'] for j in jobs}
  if (B/'submission_intent.json').exists():assert read(B/'submission_intent.json')['status']=='recorded','Reconcile uncertain submission'
  for cid in sorted(m['cases'],key=lambda k:(m['cases'][k]['kind']=='mip',k)):
   if cid in done:continue
   c=m['cases'][cid];r=c['resources'];mins=math.ceil(r['allocation_s']/60);deps=[done[c['source_case']]] if c.get('source_case') else []
   argv=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task='+str(r['cpus']),
    '--mem='+r['mem'],'--time='+f'{mins//60}:{mins%60:02d}:00','--requeue','--kill-on-invalid-dep=yes','--job-name=drU_'+cid,
    '--output='+str(B/'logs/%x_%j.out'),'--error='+str(B/'logs/%x_%j.err')]
   if deps:argv+=['--dependency=afterok:'+':'.join(deps)]
   argv += [str(B/'worker.sub'),str(B),cid,c['kind']]
   intent=dict(case_id=cid,argv=argv,status='submitting',utc=now());save(B/'submission_intent.json',intent)
   job=subprocess.check_output(argv,text=True).strip().split(';')[0];assert job.isdigit()
   jobs.append(dict(case_id=cid,job_id=job,kind=c['kind'],dependencies=deps,argv=argv,submitted_utc=now()));done[cid]=job
   save(B/'jobs.json',jobs);save(B/'case_jobs.json',done);intent.update(status='recorded',job_id=job);save(B/'submission_intent.json',intent)
  rows=[]
  for j in jobs:
   raw=subprocess.check_output([S+'scontrol','show','job',j['job_id'],'-o'],text=True);x=dict(t.split('=',1) for t in raw.split() if '=' in t)
   assert x['Partition']=='default_partition' and x['ExcNodeList']=='scaglione-compute-01'
   if j['dependencies']:assert j['dependencies'][0] in x['Dependency']
   else:assert x['Dependency'] in ('(null)','(none)','')
   rows.append(dict(case_id=j['case_id'],job_id=j['job_id'],kind=j['kind'],state=x['JobState'],reason=x.get('Reason'),nodes=x.get('NodeList'),raw=raw))
  counts={s:sum(x['state']==s for x in rows) for s in sorted({x['state'] for x in rows})}
  save(B/'scheduler_verification.json',dict(status='passed',collected_utc=now(),rows=rows,counts=counts))
  print(json.dumps({'submitted':len(jobs),'counts':counts}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit']);a=p.parse_args();{'prepare':prepare,'submit':submit}[a.action]()
