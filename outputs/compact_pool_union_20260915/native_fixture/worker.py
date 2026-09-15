"""Restart-safe, private campaign worker. Does not edit any shared registry."""
from pathlib import Path
import argparse,fcntl,os,sys
import common as w
import pool_logic as logic

def result_gate(v,c,source,construction):
 pa=v['physical_pool_audit'];prov=v['mip_provenance'];args=prov['arguments']
 if v['physical_replay_validated'] is not True or pa['rejected_columns'] or pa['deterministically_repaired']:raise ValueError('native physical admission changed')
 if v['source_result_sha256']!=source['result_sha256'] or v['source_journal_sha256']!=source['journal_sha256']:raise ValueError('native source identity mismatch')
 if pa['added_giro_route_count']!=0 or pa['base_pool_column_count']!=pa['post_augmentation_columns'] or pa['base_pool_ordered_sha256']!=pa['augmented_pool_ordered_sha256']:raise ValueError('unexpected native pool augmentation')
 if v['master_cost_semantics']!='expanded_grid_cost' or pa['master_sense']!='cover' or v['stage2_fleet_constraint']!='at_most':raise ValueError('native model semantics changed')
 if v['mip_start']['kind'] not in ('greedy_pool_partition','none') or v['mip_start'].get('source') is not None or args['initial_partition_routes'] is not None or args['verified_expanded_initial_partition']:raise ValueError('non-native-greedy initialization')
 if prov['git_commit']!=c['execution_commit'] or prov['final_observed_git_commit']!=c['execution_commit'] or not prov['tracked_clean_at_end']:raise ValueError('native execution identity')
 if not args['cover'] or not args['two_stage'] or args['threads']!=8 or args['timelimit']!=c['solver_budget_s'] or v['stage1_time_limit_s']!=c['stage1_s']:raise ValueError('native budgets/settings changed')
 expected=construction['union_pool_set']['native_unique_columns'] if c['treatment']=='union' else c['control_source']['native_pool_columns']
 if pa['base_pool_column_count']!=expected:raise ValueError('native pool count differs from construction/source')
 if c['treatment']=='control' and pa['base_pool_ordered_sha256']!=c['control_source']['native_pool_ordered_sha256']:raise ValueError('unchanged-pool control identity mismatch')
 return pa

def execute(root,cid):
 root=Path(root).resolve();w.safe_id(cid);manifest=w.read(root/'manifest.json');mh=w.sha(root/'manifest.json');c=manifest['cases'][cid]
 token=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');w.safe_id(token)
 stage=root/'cases'/cid;attempt=stage/'attempts'/token;attempt.mkdir(parents=True,exist_ok=False)
 state={'case_id':cid,'kind':c['kind'],'attempt':token,'manifest_sha256':mh,'started_utc':w.now(),'status':'preflight','is_validation':bool(c.get('is_validation')),'resources':c['resources'],'restart_count':os.environ.get('SLURM_RESTART_COUNT','0'),'job_id':os.environ['SLURM_JOB_ID']};w.save(attempt/'state.json',state)
 with (stage/'.lock').open('a') as lock:
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for n,h in manifest['tooling_sha256'].items():w.require_hash(root/n,h)
   for p,h in c['static_hashes'].items():w.require_hash(p,h)
   w.require_hash(c['input_path'],c['input_sha256'])
   if (stage/'completion.json').exists():
    old=w.read(stage/'completion.json')
    if old['manifest_sha256']!=mh:raise ValueError('existing completion differs')
    w.require_hash(old['result_path'],old['result_sha256'])
    if old.get('journal_path'):w.require_hash(old['journal_path'],old['journal_sha256'])
    w.save(attempt/'state.json',{**state,'status':'already_complete','prior_completion':old});return
   if c['kind']=='pool_construction':
    out=attempt/'pool.json';spec=attempt/'spec.json';w.save(spec,c)
    ex=w.run_process([sys.executable,str(root/'pool_logic.py'),'--spec',str(spec),'--out',str(out)],attempt,root,c['watchdog_s'],os.environ.copy())
    v=w.read(out);j=Path(v['columns_journal']);d=v['pool_construction']
    if v['certified_rc_optimal'] is not False or v['optimization_run'] is not False:raise ValueError('construction certificate scope')
    w.require_hash(j,d['union_journal_sha256'])
    final={**state,'status':'finished','usable':True,'optimization_run':False,'certified':False,'result_path':str(out),'result_sha256':w.sha(out),'journal_path':str(j),'journal_sha256':w.sha(j),'construction_path':str(attempt/'construction.json'),'construction_sha256':w.sha(attempt/'construction.json'),'execution':ex,'ended_utc':w.now()};target=stage/'pool.json'
   else:
    w.check_code(c['source_code'],c['execution_commit'])
    built=w.read(root/'cases'/c['source_case']/'completion.json')
    if built['case_id']!=c['source_case'] or built['kind']!='pool_construction' or not built['usable'] or built['manifest_sha256']!=mh:raise ValueError('own construction completion mismatch')
    for p,h in [('result_path','result_sha256'),('journal_path','journal_sha256'),('construction_path','construction_sha256')]:w.require_hash(built[p],built[h])
    construction=w.read(built['construction_path'])
    if c['treatment']=='union':source=built
    else:
     s=c['control_source'];logic.validate_source(s)
     source={'result_path':s['status_path'],'result_sha256':s['status_sha256'],'journal_path':s['journal_path'],'journal_sha256':s['journal_sha256']}
    out=attempt/'result.json';argv=[sys.executable,str(Path(c['source_code'])/'src/run_exact_pool_mip.py'),'--result',source['result_path'],'--data-dir',c['data_dir'],'--reference-data-dir',c['data_dir'],'--cover','--two-stage','--timelimit',str(c['solver_budget_s']),'--stage1-timelimit',str(c['stage1_s']),'--threads','8','--mipgap','0.0001','--progress-dir',str(attempt/'progress'),'--gurobi-log',str(attempt/'gurobi.log'),'--out',str(out)]
    if any(x in argv for x in ['--initial-partition-routes','--extra-routes','--verified-expanded-initial-partition']):raise ValueError('seed augmentation prohibited')
    env=os.environ.copy();env.update(EVSP_EXPECTED_COMMIT=c['execution_commit'],EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=source['result_sha256'],EVSP_MIP_EXPECTED_JOURNAL_SHA256=source['journal_sha256'])
    state.update(status='running',treatment=c['treatment'],optimization_run=True,source_status_sha256=source['result_sha256'],source_journal_sha256=source['journal_sha256']);w.save(attempt/'state.json',state)
    ex=w.run_process(argv,attempt,c['source_code'],c['watchdog_s'],env);v=w.read(out);pa=result_gate(v,c,source,construction)
    w.require_hash(source['result_path'],source['result_sha256']);w.require_hash(source['journal_path'],source['journal_sha256'])
    bound=v['fleet_bound'];final={**state,'status':'finished','usable':True,'result_path':str(out),'result_sha256':w.sha(out),'execution':ex,'ended_utc':w.now(),'buses':v['buses'],'fleet_bound':bound,'fleet_proven':v['fleet_proven'],'pool_bound_excludes_target':bound is not None and bound>c['target_k']+1e-6,'target_matched':v['buses'] is not None and v['buses']<=c['target_k'],'physical_replay_validated':True,'physical_pool_audit':pa,'native_greedy_start':v['mip_start'],'best_independent_donor_fleet_upper_bound':construction['best_independent_fleet_upper_bound'],'donor_bound_is_solver_incumbent':False,'source_pool_path':source['result_path'],'source_journal_path':source['journal_path'],'initialization_policy':'Unchanged frozen-native greedy policy; realized start may differ with the pool'};target=stage/'mip_result.json'
   temporary=target.with_name(target.name+'.link.'+str(os.getpid()));temporary.symlink_to(out.resolve());temporary.replace(target)
   w.save(stage/'completion.json',final);w.save(attempt/'state.json',final)
  except BaseException as e:
   w.save(attempt/'state.json',{**state,'status':'execution_failed','error':repr(e),'ended_utc':w.now()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();execute(a.root,a.case)
