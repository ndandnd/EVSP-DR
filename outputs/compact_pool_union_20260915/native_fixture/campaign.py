"""Prepare immutable plans; submit validation or explicitly authorized production."""
from pathlib import Path
import argparse,copy,fcntl,json,math,os,shutil,subprocess
import common as w
B=Path(__file__).resolve().parent
REMOTE=Path('/home/nc437/ladder-lite/compact_pool_union_20260915')
STORE=Path('/share/scaglione/nc437/evsp-dr/compact_pool_union_20260915')
CODE=Path('/home/nc437/ladder-lite/execution/871d057e1067411f09581e37d78f7c1ca43f68bb')
COMMIT=CODE.name
SLURM='/usr/local/slurm/slurm-25.05.5/bin/'
TOOLS=['common.py','pool_logic.py','worker.py','worker.sub','campaign.py']
def accounting(audit):
 cumulative=w.read(REMOTE.parent/'cumulative_budget_20260913/manifest.json');allrows={};pairs=[]
 for pair in audit['pairs']:
  refs={};s=pair['sources'][0];seed=s['source_manifest_seed'];h=seed['historical_costs'];k=pair['target_k'];chain=int(pair['pair_id'][1])
  if k==15:
   for p,d in zip(h['ancestry_paths'],h['ancestry_sha256']):refs[p]=d
  else:
   c=cumulative['cases'][f'c{chain}_k15']
   for p,d in zip(c['ancestry_paths'],c['ancestry_sha256']):refs[p]=d
   refs[c['status_path']]=c['status_sha256']
   for r in h['extension_history']:refs[r['path']]=r['sha256']
  parent=[]
  for p,d in refs.items():
   w.require_hash(p,d);v=w.read(p);r={'path':p,'sha256':d,'native_wall_s':v['wall_s'],'execution_commit':v['provenance']['git_commit']};parent.append(r);allrows[d]=r
  child=[{'path':x['status_path'],'sha256':x['status_sha256'],'native_wall_s':x['cg_wall_s'],'execution_commit':x['cg_commit']} for x in pair['sources']]
  for r in child:allrows[r['sha256']]=r
  pairs.append({'pair_id':pair['pair_id'],'shared_upstream_cg_native_wall_s':sum(r['native_wall_s'] for r in parent),'both_donor_cg_native_wall_s':sum(r['native_wall_s'] for r in child),'prior_cg_native_wall_s_total':sum(r['native_wall_s'] for r in parent+child),'shared_upstream_sources':parent,'donor_sources':child})
 return {'scope':'Native CG wall for both current-k donors plus unique shared upstream statuses through previous k. Upstream paid once per pair. Across-campaign total deduplicates statuses by SHA; scheduler CPU and graph build expenditure remain separate in source audit. Excludes unreferenced failed attempts. Not a same-computation performance comparison.','pairs':pairs,'unique_status_count':len(allrows),'unique_prior_cg_native_wall_s':sum(r['native_wall_s'] for r in allrows.values()),'cumulative_manifest_sha256':w.sha(REMOTE.parent/'cumulative_budget_20260913/manifest.json')}

def prepare():
 if B!=REMOTE:raise ValueError('prepare runs only at recorded cluster root')
 if (B/'manifest.json').exists():raise ValueError('manifest already exists; do not overwrite frozen plan')
 audit=w.read(B/'source_audit.json');w.require_hash(CODE/'src/run_exact_pool_mip.py',audit['native_script_sha256']);w.check_code(CODE,COMMIT)
 (STORE/'cases').mkdir(parents=True,exist_ok=True);(B/'cases').symlink_to(STORE/'cases',target_is_directory=True);(B/'logs').mkdir(exist_ok=True)
 cases={}
 for pair in audit['pairs']:
  base=pair['pair_id'];s=pair['sources'][0];static={}
  for x in pair['sources']:static.update(x['static_hashes'])
  static[str(CODE/'src/run_exact_pool_mip.py')]=audit['native_script_sha256'];static[str(CODE/'src/config.py')]=w.sha(CODE/'src/config.py')
  common={'pair_id':base,'chain':int(base[1]),'target_k':pair['target_k'],'input_path':s['input_path'],'input_sha256':s['input_sha256'],'data_dir':s['data_dir'],'static_hashes':static}
  build=base+'_build';cases[build]={**common,'id':build,'kind':'pool_construction','sources':pair['sources'],'watchdog_s':3300,'resources':{'cpus':2,'mem':'8G','allocation_s':3600}}
  for treatment in ['union']+(['control'] if pair['control_arm'] else []):
   cid=base+'_'+treatment;cases[cid]={**common,'id':cid,'kind':'mip','source_case':build,'treatment':treatment,'source_code':str(CODE),'execution_commit':COMMIT,'solver_budget_s':12600,'stage1_s':10800,'watchdog_s':15300,'resources':{'cpus':8,'mem':'24G','allocation_s':16200},'initialization':'native_greedy_pool_partition','supplied_start':False}
   if treatment=='control':cases[cid]['control_source']=next(x for x in pair['sources'] if x['arm']==pair['control_arm'])
 m={'schema':'evsp-compact-pool-union-controlled-v1','prepared_utc':w.now(),'cases':cases,'source_audit_sha256':w.sha(B/'source_audit.json'),'tooling_sha256':{n:w.sha(B/n) for n in TOOLS},'storage_root':str(STORE),'policy_sha256':w.sha(B.parent/'SCAGLIONE_RESOURCE_POLICY.md'),'initialization_design':'Frozen-native greedy algorithm in both arms; realized greedy start is pool-dependent. No supplied initial routes or extra columns. Donor witnesses are independent feasible upper bounds, not new-solver incumbents.','objective':'Two-stage fleet then charging/start-fee objective at most validated fleet incumbent; bus coefficient100000, charge-start fee5; cover,240kWh/240kW,no reserve/shared capacity/terminal floor','interpretation':'8 native all-column unions and4 unchanged core512 controls; original pool proofs suffice for4 fully excluding pairs. No CG or pricing certificate created. Not a same-computation performance comparison.','independent_construction_cases':8,'independent_mip_cases':12,'arbitrary_throttle':None,'production_authorized':False}
 w.save(B/'prior_cg_accounting.json',accounting(audit));m['prior_cg_accounting_sha256']=w.sha(B/'prior_cg_accounting.json')
 w.save(B/'manifest.json',m)
 w.save(B/'preflight.json',{'status':'passed','manifest_sha256':w.sha(B/'manifest.json'),'source_markers_and_native_settings_verified':16,'fixture_required':True,'production_submitted':False,'native_script_sha256':audit['native_script_sha256']})
 print(json.dumps({'prepared':len(cases),'construction':8,'mip':12,'production_submitted':False}))

def fixture():
 root=B/'native_fixture';root.mkdir();(STORE/'native_fixture/cases').mkdir(parents=True,exist_ok=True);(root/'cases').symlink_to(STORE/'native_fixture/cases',target_is_directory=True);(root/'logs').mkdir()
 for n in TOOLS:shutil.copy2(B/n,root/n)
 m=copy.deepcopy(w.read(B/'manifest.json'));m['cases']={k:c for k,c in m['cases'].items() if k in ['c1_k15_build','c1_k15_union','c2_k20_build','c2_k20_union','c2_k20_control']}
 for c in m['cases'].values():
  c['is_validation']=True;c['watchdog_s']=1700;c['resources']['allocation_s']=1800
  if c['kind']=='mip':c['solver_budget_s']=30;c['stage1_s']=20
 m['is_validation']=True;m['storage_root']=str(STORE/'native_fixture');m['parent_manifest_sha256']=w.sha(B/'manifest.json');m['tooling_sha256']={n:w.sha(root/n) for n in TOOLS};w.save(root/'manifest.json',m)
 return root

def submission_argv(root,cid,c,deps,fixture_run=False):
 r=c['resources'];mins=math.ceil(r['allocation_s']/60)
 argv=[SLURM+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task='+str(r['cpus']),'--mem='+r['mem'],'--time='+f'{mins//60}:{mins%60:02d}:00','--no-requeue' if fixture_run else '--requeue','--kill-on-invalid-dep=yes','--job-name='+('cuFix_' if fixture_run else 'cu15_')+cid,'--output='+str(root/'logs/%x_%j.out'),'--error='+str(root/'logs/%x_%j.err')]
 if deps:argv+=['--dependency=afterok:'+':'.join(deps)]
 return argv+[str(root/'worker.sub'),str(root),cid,c['kind']]

def submit(root,fixture_run=False):
 with (root/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);m=w.read(root/'manifest.json')
  for n,h in m['tooling_sha256'].items():w.require_hash(root/n,h)
  jobs=w.read(root/'jobs.json') if (root/'jobs.json').exists() else [];done={j['case_id']:j['job_id'] for j in jobs}
  if (root/'submission_intent.json').exists() and w.read(root/'submission_intent.json')['status']!='recorded':raise ValueError('reconcile uncertain submission before retry')
  for cid,c in sorted(m['cases'].items(),key=lambda x:(x[1]['kind']=='mip',x[0])):
   if cid in done:continue
   deps=[done[c['source_case']]] if c.get('source_case') else [];argv=submission_argv(root,cid,c,deps,fixture_run)
   intent={'case_id':cid,'argv':argv,'status':'submitting','utc':w.now()};w.save(root/'submission_intent.json',intent)
   job=subprocess.check_output(argv,text=True,timeout=45).strip().split(';')[0]
   if not job.isdigit():raise ValueError('ambiguous job ID')
   rec={'case_id':cid,'job_id':job,'kind':c['kind'],'dependencies':deps,'argv':argv,'submitted_utc':w.now()};jobs.append(rec);done[cid]=job;w.save(root/'jobs.json',jobs);w.save(root/'case_jobs.json',done);w.save(root/'submission_intent.json',{**intent,'status':'recorded','job_id':job})
   raw=subprocess.check_output([SLURM+'scontrol','show','job',job,'-o'],text=True);x=dict(t.split('=',1) for t in raw.split() if '=' in t)
   if x['Partition']!='default_partition' or x['ExcNodeList']!='scaglione-compute-01' or int(x['NumCPUs'])!=c['resources']['cpus'] or x['MinMemoryNode']!=c['resources']['mem']:raise ValueError('effective resources mismatch')
   if deps and not all(d in x['Dependency'] for d in deps):raise ValueError('effective dependency mismatch')
   if not deps and x['Dependency'] not in ['(null)','(none)','']:raise ValueError('unexpected dependency')
   rec['effective_scontrol_at_submission']=raw;w.save(root/'jobs.json',jobs)
  w.save(root/'scheduler_verification.json',{'status':'passed','manifest_sha256':w.sha(root/'manifest.json'),'rows':jobs,'utc':w.now(),'all_exclude_reserved_compute01':True,'independent_jobs_have_no_throttle':True});print(json.dumps(done))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fixture','submit']);p.add_argument('--production-authorized',action='store_true');a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='fixture':submit(fixture(),True)
 else:
  if not a.production_authorized:raise SystemExit('Production requires parent review; pass explicit production authorization only after review.')
  report=w.read(B/'native_validation.json')
  if report['status']!='passed' or report['production_manifest_sha256']!=w.sha(B/'manifest.json'):raise ValueError('native fixture validation missing or stale')
  submit(B)
