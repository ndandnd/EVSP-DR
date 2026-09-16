#!/usr/bin/env python3
"""Idempotent frozen strict-chain submission; unresolved intents stop retries."""
import fcntl,hashlib,json,subprocess,sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SLURM=Path('/usr/local/slurm/slurm-25.05.5/bin')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(tmp,path)
with open(ROOT/'submit.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 manifest=json.load(open(ROOT/'manifest.json'));ledgerpath=ROOT/'jobs.json';state=json.load(open(ledgerpath)) if ledgerpath.exists() else {'manifest_sha256':sha(ROOT/'manifest.json'),'jobs':{},'intents':{}}
 assert state['manifest_sha256']==sha(ROOT/'manifest.json')
 if state['intents']:raise SystemExit('Unresolved submission intent: reconcile scheduler before retrying; no automatic duplicate')
 smoke=json.load(open(ROOT/'smoke/native/verification.json'));assert smoke['both_certified'] and smoke['child_inherited_columns']==1
 assert smoke['commit']==manifest['execution_commit']
 code=ROOT/'code'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()==manifest['execution_commit']
 assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=code,text=True).strip()
 for name,expected in manifest['data_sha256'].items():assert sha(code/'data'/name)==expected
 for name,expected in manifest['tooling_sha256'].items():assert sha(ROOT/name)==expected
 (ROOT/'logs').mkdir(exist_ok=True)
 for case in manifest['cases']:
  assert sha(ROOT/case['input'])==case['input_sha256']
  for mode in ['cg','mip']:
   key=case['id']+'__'+mode
   if key in state['jobs']:continue
   deps=[]
   if mode=='cg' and case['previous_group_case']:deps=[state['jobs'][case['previous_group_case']+'__cg']['job_id']]
   if mode=='mip':deps=[state['jobs'][case['id']+'__cg']['job_id']]
   cmd=[str(SLURM/'sbatch'),'--parsable','--partition=default_partition','--exclude=scaglione-compute-01','--cpus-per-task=8','--mem='+('48G' if mode=='cg' else '24G'),'--time='+('06:00:00' if mode=='cg' else '01:15:00'),'--requeue','--job-name=rvS'+case['id'][4:]+mode,'--comment=EVSP_REVIEW_STRICT_'+key,'--output='+str(ROOT/'logs'/('%j_'+key+'.out')),'--error='+str(ROOT/'logs'/('%j_'+key+'.err'))]
   if deps:cmd+=['--dependency=afterok:'+':'.join(map(str,deps))]
   cmd+=['--chdir='+str(ROOT),'--export=ALL',str(ROOT/'worker.sh'),str(ROOT),case['id'],mode]
   intent={'command':cmd,'dependencies':deps,'created_unix':time.time()};state['intents'][key]=intent;save(ledgerpath,state)
   proc=subprocess.run(cmd,capture_output=True,text=True)
   if proc.returncode:
    state['intents'][key].update(error=proc.stderr,returncode=proc.returncode);save(ledgerpath,state);raise SystemExit(proc.stderr)
   job=proc.stdout.strip().split(';')[0]
   if not job.isdigit():raise SystemExit('Ambiguous sbatch output; retain intent and reconcile')
   state['jobs'][key]={**intent,'job_id':job,'submitted_unix':time.time(),'stdout':proc.stdout.strip()};del state['intents'][key];save(ledgerpath,state)
   info=subprocess.check_output([str(SLURM/'scontrol'),'show','job',job,'--oneliner'],text=True)
   if 'ExcNodeList=scaglione-compute-01' not in info:raise SystemExit('Required exclusion missing; root must inspect before further submissions')
   state['jobs'][key]['scontrol_initial']=info.strip();save(ledgerpath,state)
 print(json.dumps({'submitted_or_existing_jobs':len(state['jobs']),'root':str(ROOT)}))
