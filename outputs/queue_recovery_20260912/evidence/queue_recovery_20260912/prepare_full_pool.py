from pathlib import Path
import subprocess,json,hashlib,shutil,os,datetime
Q=Path('/home/nc437/ladder-lite/queue_recovery_20260912');B=Path('/home/nc437/ladder-lite/full_pool_recovery_20260912');O=Path('/home/nc437/ladder-lite/overnight_extension_20260912');C=B/'code';pin=(Q/'full_pool_execution_pin.txt').read_text().strip();compat=json.loads((Q/'full_pool_cache_compatibility.json').read_text())
B.mkdir(exist_ok=True);(B/'logs').mkdir(exist_ok=True)
assert not (B/'manifest.json').exists()
if not C.exists():subprocess.run(['git','clone','--shared','--no-checkout','/home/nc437/ladder-lite/efficiency_validation_20260912/code-baseline',str(C)],check=True,capture_output=True)
subprocess.run(['git','-C',str(C),'fetch',str(Q/'evsp-queue-recovery-code.bundle'),'HEAD'],check=True,capture_output=True)
subprocess.run(['git','-C',str(C),'checkout','--detach',pin],check=True,capture_output=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def copy_data(rel):
 s=O/'code/data'/rel;d=C/'data'/rel;d.parent.mkdir(parents=True,exist_ok=True)
 if not d.exists():shutil.copy2(s,d)
 assert sha(s)==sha(d)
 return sha(d)
for rel in ['hourly_prices_flat.csv','Ref_dict.csv','par_ref_dhd.csv']:copy_data(rel)
original=json.loads((O/'manifest.json').read_text());cases={};parents={}
for chain,ids in original['warm_chains'].items():
 for pos,cid in enumerate(ids):
  old=original['cases'][cid];case=dict(old);k=int(cid[-2:]);case['cg_seconds']=28800 if k<=10 else 14400;case['kind']='full_pool_inheritance_indexed_recovery';case['source_original_case']=str(O/'cases'/cid);case['csv_sha256']=copy_data(case['csv']);assert case['csv_sha256']==case['input_sha256']
  parent=Path(old['parent_status']) if pos==0 else B/'cases'/ids[pos-1]/'cg.json';case['parent_status']=str(parent)
  if pos==0:
   pv=json.loads(parent.read_text());assert pv['final']['artificials']==0 and pv['final']['iter']>0
   copy_data(pv['csv']);jp=Path(pv['columns_journal']);assert jp.stat().st_size>0
   parents[chain]={'status':str(parent),'status_sha256':sha(parent),'journal':str(jp),'journal_sha256':sha(jp),'csv':pv['csv'],'csv_sha256':copy_data(pv['csv'])}
  source=Path(case['source_cache']);meta=Path(str(source)+'.manifest.json');m=json.loads(meta.read_text());actual=sha(source)
  assert actual==m['pickle_sha256'];assert m['identity']['instance_sha256']==case['input_sha256']
  # Verify the original producer used the exact audited reference graph code.
  prod=m['identity']['git_commit'];producer_source=subprocess.check_output(['git','-C',str(C),'show',prod+':src/event_pricer_network.py'])
  assert hashlib.sha256(producer_source).hexdigest()==compat['reference_network_sha256'],(cid,prod)
  case_root=B/'cases'/cid;case_root.mkdir(parents=True,exist_ok=True);dest=case_root/'network.pkl';os.link(source,dest)
  (case_root/'source_network_manifest.json').write_bytes(meta.read_bytes())
  m['producer_identity']=dict(m['identity']);m['identity']['git_commit']=pin;m['consumer_compatibility_audit']=str(Q/'full_pool_cache_compatibility.json');Path(str(dest)+'.manifest.json').write_text(json.dumps(m,indent=2)+'\n')
  case['cache']=str(dest);case['cache_sha256']=actual;case['cache_manifest_sha256']=sha(str(dest)+'.manifest.json');cases[cid]=case
v={'schema':'evsp-full-pool-recovery-v1','prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_commit':pin,'mip_execution_commit':'871d057e1067411f09581e37d78f7c1ca43f68bb','cases':cases,'warm_chains':original['warm_chains'],'initial_parents':parents,'scientific_settings':{'master_sense':'cover','battery_kwh':240,'charge_kw':240,'soc_step_kwh':2.5,'block_min':5,'reserve_kwh':0,'return_soc_floor':None,'shared_station_capacity':False,'prices':'flat','objective':'100000 + electricity +5 per charge start','columns_per_iter':30,'rc_epsilon':0.0001,'inherit_max_columns':0,'inherit_time_limit_s':0,'inherit_workers':8,'fixed_sequence_index':True,'skip_gurobi_incidence':False,'mip_seconds':3600,'stage1_seconds':1800,'stage2_fleet_sense':'<= validated incumbent'},'budget_note':'Recover original k<=10 full-pool cases with their original 8h CG budget. Extend k>=11 with 4h CG, matching bounded extension budgets. Preparation included. All prior failed and bounded results retained. Indexing changes replay traversal only; all inherited sequences are attempted unless overall application deadline binds.','borrowed_object_store':'/home/nc437/ladder-lite/efficiency_validation_20260912/code-baseline/.git/objects'}
(B/'manifest.json').write_text(json.dumps(v,indent=2)+'\n');shutil.copy2(Q/'full_pool_cache_compatibility.json',B/'cache_compatibility.json');print(json.dumps({'prepared_cases':len(cases),'chains':len(v['warm_chains']),'pin':pin}))
