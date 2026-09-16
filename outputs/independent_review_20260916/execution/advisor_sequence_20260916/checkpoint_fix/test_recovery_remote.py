from pathlib import Path
import sys,json,os,shutil,subprocess,tempfile,time,multiprocessing,hashlib
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
import checkpoint_recovery as recovery
CODE=Path('/home/nc437/ladder-lite/review_full40_20260916/code')
SEED=Path('/tmp/evsp-checkpoint-audit-lbqv6x8r/seed/cg.json')
PY='/home/nc437/evsp_env/bin/python'
BASE=Path(tempfile.mkdtemp(prefix='evsp-checkpoint-fix-tests-'))
seed=json.loads(SEED.read_text())
argv=['--csv',seed['csv'],'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--fixed-sequence-index','--soc-step','2.5','--block-min','5','--columns_per_iter','30','--column-selection','reduced_cost','--column-diversity-weight','0.0','--column-candidate-multiplier','4','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--g-kwh','240','--charge-kw','240','--min-soc-frac','0']
results=[]
def validate(p):
 q=subprocess.run([PY,str(HERE/'checkpoint_validator.py')],input=json.dumps({'code':str(CODE),'argv':argv+['--out',str(p),'--resume']}),capture_output=True,text=True,cwd=CODE,timeout=90)
 if q.returncode:raise ValueError(q.stdout+q.stderr)
 return json.loads(q.stdout)
def clone(d):
 d.mkdir(parents=True);p=d/'cg.json'
 for suffix in recovery.SUFFIXES:shutil.copy2(str(SEED)+suffix,str(p)+suffix)
 os.utime(p,None)
 return p
def hashes(paths):return {str(p):recovery.sha(p) for d in paths for p in d.rglob('*') if p.is_file()}
def scenario(name,mutator,expected='old',copy_file=shutil.copy2):
 root=BASE/name;old=clone(root/'old');new=clone(root/'new');mutator(new);os.utime(new,ns=(time.time_ns(),time.time_ns()+1000000));attempt=root/'current';attempt.mkdir();before=hashes([old.parent,new.parent])
 out=recovery.recover(root,attempt,validate,copy_file=copy_file);r=json.loads((attempt/'resume_selection.json').read_text());assert Path(r['candidates'][-1]['source_status']).parent.name==expected,r
 assert hashes([old.parent,new.parent])==before
 assert out==attempt/'resume/cg.json' and (out.parent/'ready.json').exists()
 assert validate(out)['valid'];results.append({'test':name,'passed':True,'selection':r,'sources_unchanged':True})
def broken(p):p.write_text('{"bad":')
def incomplete(p):
 j=Path(str(p)+'.columns.jsonl');j.write_bytes(j.read_bytes().splitlines(keepends=True)[0])
def identity(p):
 d=json.loads(p.read_text());d['charge_kw']=241;p.write_text(json.dumps(d))
def ahead(p):
 d=json.loads(p.read_text());d.update(columns=0,final_lp=None,iterations=0,wall_s=0);p.write_text(json.dumps(d))
def torn(p):
 j=Path(str(p)+'.columns.jsonl');j.write_bytes(j.read_bytes()+b'{"partial":')
scenario('corrupt_latest_json_fallback',broken)
scenario('incomplete_latest_journal_fallback',incomplete)
scenario('identity_mismatch_fallback',identity)
scenario('journal_ahead_status_accepted',ahead,'new')
scenario('safe_torn_tail_repaired_privately',torn,'new')
def interrupted_copy(source,target):
 if Path(source).parent.name=='new':
  Path(target).write_bytes(b'partial');raise OSError('injected copy interruption')
 return shutil.copy2(source,target)
scenario('copy_failure_fallback',lambda p:None,copy_file=interrupted_copy)
root=BASE/'all_invalid';p=clone(root/'bad');broken(p);a=root/'current';a.mkdir()
try:recovery.recover(root,a,validate);raise AssertionError('empty restart must not occur')
except recovery.RecoveryError:pass
assert not (a/'resume').exists();results.append({'test':'all_invalid_fail_closed','passed':True,'report':json.loads((a/'resume_selection.json').read_text())})
root=BASE/'missing_status';(root/'old').mkdir(parents=True);a=root/'current';a.mkdir()
try:recovery.recover(root,a,validate);raise AssertionError('missing status must fail closed')
except recovery.RecoveryError:pass
results.append({'test':'prior_attempt_without_checkpoint_fail_closed','passed':True})
root=BASE/'first_run';a=root/'current';a.mkdir(parents=True);assert recovery.recover(root,a,validate) is None;results.append({'test':'genuine_first_run_allowed','passed':True})
root=BASE/'hard_kill';old=clone(root/'old');a=root/'killed';a.mkdir()
def exit_copy(source,target):Path(target).write_bytes(b'partial');os._exit(99)
def child():recovery.recover(root,a,validate,copy_file=exit_copy)
proc=multiprocessing.get_context('fork').Process(target=child);proc.start();proc.join(30);assert proc.exitcode==99
assert not (a/'resume').exists() and list(a.glob('.resume-stage-*'))
b=root/'next';b.mkdir();out=recovery.recover(root,b,validate);assert out and validate(out)['valid'];results.append({'test':'hard_kill_during_copy_then_requeue_fallback','passed':True,'no_partial_checkpoint_published':True,'report':json.loads((b/'resume_selection.json').read_text())})
# Resume once from a published nested copy with the unchanged native solver.
out=BASE/'journal_ahead_status_accepted/current/resume/cg.json'
env=dict(os.environ,GRB_LICENSE_FILE='/share/apps/software/gurobi/gurobi.lic',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LM_LICENSE_FILE',None)
with (BASE/'native_resume.log').open('w') as f:
 q=subprocess.run([PY,str(CODE/'src/exact_pricer_expanded.py'),*argv,'--out',str(out),'--resume','--max-iters','20','--wall-limit-s','300'],cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
assert q.returncode==0,(BASE/'native_resume.log').read_text();d=json.loads(out.read_text());assert d['certified_rc_optimal']
results.append({'test':'native_solver_resumes_published_checkpoint','passed':True,'columns':d['columns'],'certified_rc_optimal':d['certified_rc_optimal'],'stop_reason':d['stop_reason']})
print(json.dumps({'tests':results,'temp_root':str(BASE),'slurm_submissions':0,'production_modified':False,'source_code_commit':subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()},indent=2))
