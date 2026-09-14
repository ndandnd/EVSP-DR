"""Hash-bound operational repeats and newly selected extension gap searches."""
from pathlib import Path
import argparse, copy, datetime, fcntl, hashlib, importlib.util, json, math, os, shutil, subprocess

B=Path('/home/nc437/ladder-lite/mip_repeatability_20260914')
D=Path('/share/scaglione/nc437/evsp-dr/mip_repeatability_20260914')
OLD=B.parent/'overnight_diagnostics_20260914'
EXT=B.parent/'chain_extension_20260913'
S='/usr/local/slurm/slurm-25.05.5/bin/'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with t.open('w') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 t.replace(p)
def worker_module():
 spec=importlib.util.spec_from_file_location('repeat_worker',B/'worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def prepare():
 assert not (B/'manifest.json').exists(),'Manifest is immutable once prepared'
 prior=read(OLD/'manifest.json');ext=read(EXT/'manifest.json');selection=read(B/'selection.json')
 for n in ['worker.py','worker.sub']:
  assert sha(OLD/n)==prior['tooling_sha256'][n]
  shutil.copy2(OLD/n,B/n)
 D.mkdir(parents=True,exist_ok=True);(D/'cases').mkdir(exist_ok=True)
 if not (B/'cases').exists():(B/'cases').symlink_to(D/'cases',target_is_directory=True)
 assert (B/'cases').resolve()==(D/'cases').resolve()
 (B/'logs').mkdir(exist_ok=True)
 cases={}
 for cid in selection['recovered_inherited']:
  parent=prior['cases'][cid];endpoint=read(OLD/'cases'/cid/'completion.json')
  assert sha(endpoint['result_path'])==endpoint['result_sha256']
  result=read(endpoint['result_path'])
  assert result['physical_replay_validated'] and result['fleet_proven'] and result['buses']==parent['target_k']
  for rep in [1,2,3]:
   c=copy.deepcopy(parent);name=cid.replace('_extension_longmip','')+'_original_budget_repeat'+str(rep)
   c.update(id=name,treatment='original_budget_repeatability',replicate=rep,solver_budget_s=3600,
    stage1_budget_s=1800,watchdog_s=6300,resources=dict(cpus=8,mem='24G',allocation_s=7200),
    comparator_case=cid,comparator_result_sha256=endpoint['result_sha256'],
    interpretation='Operational repeatability at the original one-hour total/30-minute fleet allowance; identical default Gurobi seed, eight threads, pool and initializer. Not independent input instances or a seed-effect experiment; node and parallel timing vary.',
    changed_factor='Only replicate allocation relative to original 3600/1800s settings; no pool, solver, physics, seed or initializer changes.')
   a=c['argv'];a[a.index('--timelimit')+1]='3600';a[a.index('--stage1-timelimit')+1]='1800'
   cases[name]=c
 for original in selection['extension_gaps']:
  e=ext['cases'][original];source=(EXT/'cases'/original/'cg.json').resolve();status=read(source)
  latest_path=EXT/'cases'/original/'mip_result.json';latest=read(latest_path)
  template=copy.deepcopy(prior['cases'][selection['recovered_inherited'][0]])
  name=original+'_newgap_longmip';data=EXT/'code/data'
  template.update(id=name,chain=e['chain'],target_k=e['k'],target_duties=e['k'],csv=e['csv'],
   input_path=str(data/e['csv']),input_sha256=e['input_sha256'],source_status=str(source),
   source_status_sha256=sha(source),source_journal_sha256=sha(status['columns_journal']),
   source_cg_commit=status['provenance']['git_commit'],original_case=original,
   treatment='new_snapshot_gap_longer_search',selection_snapshot=selection['snapshot'],
   latest_original_result=str(latest_path),latest_original_sha256=sha(latest_path),
   latest_original_buses=latest['buses'],latest_original_target_matched=latest['buses']<=e['k'],
   selected_as_gap_at_snapshot=True,
   interpretation='Selected unresolved at frozen082509Z snapshot; latest original endpoint separately rechecked before launch. New Gurobi tree on unchanged saved pool; finite-pool proof only.')
  template['comparator']=dict(prior_result=str(latest_path),prior_sha256=sha(latest_path))
  cases[name]=template
 assert len(cases)==30 and all(not x.get('source_case') for x in cases.values())
 manifest=dict(schema='evsp-mip-repeatability-v1',prepared_utc=now(),cases=cases,
  selection=selection,selection_sha256=sha(B/'selection.json'),
  prior_manifest_sha256=sha(OLD/'manifest.json'),policy_sha256=sha(B.parent/'SCAGLIONE_RESOURCE_POLICY.md'),
  tooling_sha256={n:sha(B/n) for n in ['worker.py','worker.sub']},storage_root=str(D),
  worker_registry_note='Exact unchanged frozen worker registers attempts under default_overnight_diagnostics_20260914; unique result_path and attempt IDs distinguish this campaign.',
  independent_allocations=30,independent_input_instances=12,physics=prior['physics'])
 save(B/'manifest.json',manifest)
 w=worker_module();checks=[]
 for cid,c in cases.items():
  source=w.preflight(B,manifest,c)
  a=w.expand_argv(c['argv'],Path('/validation')/cid/'result.json',Path('/validation')/cid,source['path'])
  assert '--cover' in a and '--two-stage' in a and a[a.index('--threads')+1]=='8'
  assert int(a[a.index('--timelimit')+1])==c['solver_budget_s']
  assert int(a[a.index('--stage1-timelimit')+1])==c['stage1_budget_s']
  checks.append(dict(case_id=cid,source_status_sha256=source['status_sha256'],source_journal_sha256=source['journal_sha256'],status='passed'))
 save(B/'validation.json',dict(status='passed',validated_utc=now(),manifest_sha256=sha(B/'manifest.json'),
  checks=checks,worker_reused_exactly=True,native_license_gate='Frozen worker.sub performs native2001-variable license gate per allocation',
  existing_native_smoke_validation=read(OLD/'validation.json')))
 print(json.dumps({'prepared':30,'validation':'passed'}))

def submit():
 with (B/'launch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  v=read(B/'manifest.json');validation=read(B/'validation.json')
  assert validation['status']=='passed' and validation['manifest_sha256']==sha(B/'manifest.json')
  for n,h in v['tooling_sha256'].items():assert sha(B/n)==h
  ledger=read(B/'jobs.json') if (B/'jobs.json').exists() else []
  done={x['case_id']:x['job_id'] for x in ledger}
  if (B/'submission_intent.json').exists():
   old=read(B/'submission_intent.json')
   assert old.get('status')=='recorded','Ambiguous prior submission: reconcile before retrying'
  for cid,c in sorted(v['cases'].items()):
   if cid in done:continue
   minutes=math.ceil(c['resources']['allocation_s']/60)
   args=[S+'sbatch','--parsable','--partition=default_partition','--exclude=scaglione-compute-01',
    '--cpus-per-task=8','--mem=24G','--time='+f'{minutes//60}:{minutes%60:02d}:00','--requeue',
    '--job-name=drR_'+cid,'--output='+str(B/'logs/%x_%j.out'),'--error='+str(B/'logs/%x_%j.err'),
    str(B/'worker.sub'),str(B),cid]
   intent=dict(case_id=cid,argv=args,status='submitting',created_utc=now());save(B/'submission_intent.json',intent)
   output=subprocess.check_output(args,text=True).strip();job=output.split(';')[0];assert job.isdigit()
   ledger.append(dict(case_id=cid,job_id=job,kind='mip',treatment=c['treatment'],dependencies=[],argv=args,submitted_utc=now()))
   save(B/'jobs.json',ledger);done[cid]=job;save(B/'case_jobs.json',done)
   intent.update(status='recorded',job_id=job);save(B/'submission_intent.json',intent)
  verify(ledger)

def verify(ledger=None):
 ledger=ledger or read(B/'jobs.json');rows=[]
 for x in ledger:
  raw=subprocess.check_output([S+'scontrol','show','job',x['job_id'],'-o'],text=True)
  tokens=dict(t.split('=',1) for t in raw.split() if '=' in t)
  assert tokens['Partition']=='default_partition' and tokens['ExcNodeList']=='scaglione-compute-01'
  assert tokens.get('Dependency') in ('(null)','(none)','')
  assert tokens['NumCPUs']=='8'
  rows.append(dict(case_id=x['case_id'],job_id=x['job_id'],state=tokens['JobState'],reason=tokens.get('Reason'),nodes=tokens.get('NodeList'),raw=raw))
 counts={s:sum(x['state']==s for x in rows) for s in sorted({x['state'] for x in rows})}
 save(B/'scheduler_verification.json',dict(status='passed',collected_utc=now(),counts=counts,rows=rows,all_independent=True))
 print(json.dumps(dict(submitted=len(rows),counts=counts)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','submit','verify']);a=p.parse_args()
 {'prepare':prepare,'submit':submit,'verify':verify}[a.action]()
