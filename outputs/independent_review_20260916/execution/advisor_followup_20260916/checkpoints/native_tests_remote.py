"""No Slurm submissions. Native a0 checkpoint tests only in a new temporary directory."""
from pathlib import Path
import csv,json,hashlib,subprocess,tempfile,os,shutil,time,sys
P=Path(tempfile.mkdtemp(prefix='evsp-checkpoint-audit-'));CODE=Path('/home/nc437/ladder-lite/review_full40_20260916/code');PY='/home/nc437/evsp_env/bin/python';payload=json.loads(sys.stdin.read());(P/'fixture.csv').write_text(payload['csv']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,GRB_LICENSE_FILE='/share/apps/software/gurobi/gurobi.lic',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1');env.pop('LM_LICENSE_FILE',None)
base=[PY,str(CODE/'src/exact_pricer_expanded.py'),'--csv',str(P/'fixture.csv'),'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--fixed-sequence-index','--soc-step','2.5','--block-min','5','--columns_per_iter','30','--column-selection','reduced_cost','--column-diversity-weight','0.0','--column-candidate-multiplier','4','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--g-kwh','240','--charge-kw','240','--min-soc-frac','0','--checkpoint-every','25']
results=[]
def run(name,extra,expected_success=True):
 out=P/name/'cg.json';out.parent.mkdir(exist_ok=True);args=base+['--out',str(out)]+extra;before={s:sha(Path(str(out)+s)) for s in ['', '.columns.jsonl','.iters.csv'] if Path(str(out)+s).exists()};t=time.monotonic()
 with (out.parent/'native.log').open('w') as f:q=subprocess.run(args,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
 after={s:sha(Path(str(out)+s)) for s in before};text=(out.parent/'native.log').read_text();row={'test':name,'returncode':q.returncode,'elapsed_s':time.monotonic()-t,'before_sha256':before,'after_sha256':after,'log_sha256':sha(out.parent/'native.log'),'log_tail':text[-1600:]}
 if q.returncode==0:
  x=json.loads(out.read_text());row['result']={k:x.get(k) for k in ['certified_rc_optimal','stop_reason','iterations','attempt_iterations','columns','wall_s','attempt_wall_s','final','final_lp_source']}
 assert (q.returncode==0)==expected_success,row
 results.append(row);return row

def clone(name):
 dest=P/name;dest.mkdir()
 for suffix in ['', '.columns.jsonl','.iters.csv']:shutil.copy2(str(P/'seed/cg.json')+suffix,str(dest/'cg.json')+suffix)
 return dest/'cg.json'
seed=run('seed',['--max-iters','1','--wall-limit-s','300']);assert seed['result']['columns']>0
clone('resume_clean');r=run('resume_clean',['--resume','--max-iters','20','--wall-limit-s','300']);assert r['result']['certified_rc_optimal'] and r['result']['iterations']>seed['result']['iterations']
p=clone('torn_tail');with_tail=Path(str(p)+'.columns.jsonl');with_tail.write_bytes(with_tail.read_bytes()+b'{"interrupted":');r=run('torn_tail',['--resume','--max-iters','20','--wall-limit-s','300']);assert r['result']['certified_rc_optimal']
p=clone('interior_corruption');j=Path(str(p)+'.columns.jsonl');lines=j.read_bytes().splitlines(keepends=True);j.write_bytes(lines[0]+b'{broken interior\n'+b''.join(lines[1:])+b'{"extra":1}\n');r=run('interior_corruption',['--resume','--max-iters','20','--wall-limit-s','300'],False);assert r['before_sha256']==r['after_sha256']
p=clone('identity_mismatch');j=Path(str(p)+'.columns.jsonl');j.write_bytes(j.read_bytes()+b'{"interrupted":');r=run('identity_mismatch',['--resume','--max-iters','20','--wall-limit-s','300','--charge-kw','241'],False);assert r['before_sha256']==r['after_sha256']
p=clone('journal_ahead');x=json.loads(p.read_text());x.update(columns=0,iterations=0,attempt_iterations=0,final=None,final_lp=None,final_lp_source=None,history_tail=[],wall_s=0,attempt_wall_s=0,certified_rc_optimal=False,stop_reason='initializing');p.write_text(json.dumps(x));r=run('journal_ahead',['--resume','--max-iters','20','--wall-limit-s','300']);assert r['result']['certified_rc_optimal']
p=clone('journal_behind');j=Path(str(p)+'.columns.jsonl');j.write_bytes(j.read_bytes().splitlines(keepends=True)[0]);r=run('journal_behind',['--resume','--max-iters','20','--wall-limit-s','300'],False);assert r['before_sha256']==r['after_sha256']
p=clone('cumulative_budget');x=json.loads(p.read_text());x['wall_s']=200.;p.write_text(json.dumps(x));r=run('cumulative_budget',['--resume','--max-iters','20','--wall-limit-s','120']);assert r['result']['stop_reason']=='wall_limit' and r['result']['attempt_iterations']==0 and r['result']['wall_s']>=200
# Reproduce campaign.py's candidate loop on synthetic private-attempt copies only.
d=P/'candidate_selection';(d/'old').mkdir(parents=True);(d/'new').mkdir();old=d/'old/cg.json';new=d/'new/cg.json';old.write_text((P/'seed/cg.json').read_text());Path(str(old)+'.columns.jsonl').write_bytes(Path(str(P/'seed/cg.json')+'.columns.jsonl').read_bytes());time.sleep(.02);new.write_text('{"provenance":');Path(str(new)+'.columns.jsonl').write_text('')
try:
 for p in reversed(sorted(d.glob('*/cg.json'),key=lambda p:p.stat().st_mtime)):
  x=json.loads(p.read_text());j=Path(str(p)+'.columns.jsonl')
  if not j.is_file():continue
  break
 raise AssertionError('Expected current wrapper to fail on newest corrupt status')
except json.JSONDecodeError:
 results.append({'test':'wrapper_latest_corrupt_status','observed':'JSONDecodeError before reaching older valid checkpoint','verdict':'automatic fallback absent; reproducible restart robustness gap'})
print(json.dumps({'temp_root':str(P),'code_commit':subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip(),'production_artifacts_modified':False,'slurm_submissions':0,'fixture_sha256':sha(P/'fixture.csv'),'tests':results},indent=2))
