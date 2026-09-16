from pathlib import Path
import argparse,os,sys
from process_support import read,save,sha,require_hash,check_code,run_process,now

def main(root,cid):
 b=Path(root);m=read(b/'manifest.json');c=m['cases'][cid];code=Path(m['code_path']);token=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');attempt=b/'cases'/cid/'attempts'/token;attempt.mkdir(parents=True,exist_ok=False);out=attempt/'optimization'
 state=dict(case_id=cid,arm=c['arm'],attempt=token,status='preflight',started_utc=now(),manifest_sha256=sha(b/'manifest.json'),execution_commit=m['execution_commit']);save(attempt/'state.json',state)
 try:
  for f,h in m['tooling_sha256'].items():require_hash(b/f,h)
  check_code(code,m['execution_commit'])
  for f,h in c['input_hashes'].items():require_hash(f,h)
  argv=[s.replace('{out}',str(out)) for s in c['argv']];state.update(status='running',argv=argv);save(attempt/'state.json',state)
  execution=run_process(argv,attempt,code,c['watchdog_s'],os.environ.copy());result=read(out/'summary.json');selected=read(out/'selected_routes.json') if (out/'selected_routes.json').exists() else []
  sys.path.insert(0,str(code/'src'))
  import config;config.CHARGE_START_COST=0
  from audit_giro_known_columns import build_problem,HORIZON_MIN
  from run_exact_pool_mip import validate_injected_route
  from physical_audit import audit_routes
  problem=build_problem(Path(c['data_dir']),c['csv'],max_station_to_trip_wait_min=HORIZON_MIN)
  audit=audit_routes(problem,selected,validate_injected_route,HORIZON_MIN)
  if selected:
   assert audit['buses']<=5 and audit['aggregate_continuous_ending_kwh']>=m['terminal_target_kwh']-1e-6
  save(out/'independent_physical_audit.json',audit);check_code(code,m['execution_commit'])
  state.update(status='finished',ended_utc=now(),execution=execution,result_path=str(out/'summary.json'),result_sha256=sha(out/'summary.json'),physical_audit_path=str(out/'independent_physical_audit.json'),physical_audit_sha256=sha(out/'independent_physical_audit.json'),matched_five_bus_cost_comparison_eligible=bool(selected) and len(selected)==5,scientific_status=result.get('cg_stop',('fixed_frontiers_complete' if result.get('frontier_complete') else 'fixed_frontier_budget_exhausted')))
  save(attempt/'state.json',state);save(b/'cases'/cid/'completion.json',state)
 except BaseException as exc:
  state.update(status='failed_or_interrupted',error=repr(exc),ended_utc=now());save(attempt/'state.json',state);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();main(a.root,a.case)
