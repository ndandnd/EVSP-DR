"""One fresh review DR arm; immutable source, private restart outputs."""
from pathlib import Path
import argparse,json,os,sys
from process_support import read,save,sha,require_hash,check_code,run_process,now

def main(root,cid):
 b=Path(root);m=read(b/'manifest.json');c=m['cases'][cid];code=Path(m['code_path'])
 token=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');attempt=b/'cases'/cid/'attempts'/token;attempt.mkdir(parents=True,exist_ok=False)
 out=attempt/'optimization';state=dict(case_id=cid,arm=c['arm'],attempt=token,status='preflight',started_utc=now(),manifest_sha256=sha(b/'manifest.json'),publication=c['publication'],execution_commit=m['execution_commit']);save(attempt/'state.json',state)
 try:
  for f,h in m['tooling_sha256'].items():require_hash(b/f,h)
  check_code(code,m['execution_commit'])
  for f,h in c['input_hashes'].items():require_hash(f,h)
  argv=[s.replace('{out}',str(out)) for s in c['argv']];env=os.environ.copy()
  state.update(status='running',argv=argv);save(attempt/'state.json',state)
  execution=run_process(argv,attempt,code,c['watchdog_s'],env)
  result=read(out/'summary.json');selected=read(out/'selected_routes.json') if (out/'selected_routes.json').exists() else []
  for r in selected:
   n=0
   for a,z,e in zip(r['charging_stops']['cst'],r['charging_stops']['cet'],r['charging_stops']['kwh']):
    if e>1e-9:
     assert z-a>=3-1e-8 and e*60/(z-a)<=240+1e-6;n+=1
   assert n==r['active_charge_start_count']
  check_code(code,m['execution_commit'])
  state.update(status='finished',ended_utc=now(),execution=execution,result_path=str(out/'summary.json'),result_sha256=sha(out/'summary.json'),selected_routes_sha256=sha(out/'selected_routes.json') if selected else None,positive_active_minimum_verified=bool(selected),scientific_status=result.get('cg_stop',('fixed_frontiers_complete' if result.get('frontier_complete') else 'fixed_frontier_budget_exhausted')))
  save(attempt/'state.json',state);save(b/'cases'/cid/'completion.json',state)
 except BaseException as exc:
  state.update(status='failed_or_interrupted',error=repr(exc),ended_utc=now());save(attempt/'state.json',state);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();main(a.root,a.case)
