from pathlib import Path
import argparse,fcntl,os,sys,json
import common as w
from target_solver import identity_gate

def execute(root,cid):
 root=Path(root).resolve();w.safe_id(cid);manifest=w.read(root/'manifest.json');mh=w.sha(root/'manifest.json');c=manifest['cases'][cid]
 token=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');w.safe_id(token)
 stage=root/'cases'/cid;attempt=stage/'attempts'/token;attempt.mkdir(parents=True,exist_ok=False)
 state={'case_id':cid,'attempt':token,'manifest_sha256':mh,'started_utc':w.now(),'status':'preflight','is_validation':c['is_validation'],'resources':c['resources'],'job_id':os.environ['SLURM_JOB_ID'],'restart_count':os.environ.get('SLURM_RESTART_COUNT','0')};w.save(attempt/'state.json',state)
 with (stage/'.lock').open('a') as lock:
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for n,h in manifest['tooling_sha256'].items():w.require_hash(root/n,h)
   for p,h in c['static_hashes'].items():w.require_hash(p,h)
   w.require_hash(c['input_path'],c['input_sha256']);w.require_hash(c['source_result'],c['source_result_sha256']);w.require_hash(c['source_journal'],c['source_journal_sha256']);w.check_code(c['source_code'],c['execution_commit'])
   if (stage/'completion.json').exists():
    old=w.read(stage/'completion.json')
    if old['manifest_sha256']!=mh:raise ValueError('existing completion identity changed')
    w.require_hash(old['result_path'],old['result_sha256']);w.save(attempt/'state.json',{**state,'status':'already_complete','prior_completion':old});return
   out=attempt/'result.json';spec=attempt/'case.json';w.save(spec,c)
   argv=[sys.executable,str(root/'target_solver.py'),'--result',c['source_result'],'--data-dir',c['data_dir'],'--reference-data-dir',c['data_dir'],'--cover','--two-stage','--target-cap',str(c['target_cap']),'--timelimit',str(c['solver_budget_s']),'--threads','8','--mipgap','0.0001','--gurobi-log',str(attempt/'gurobi.log'),'--out',str(out)]
   env=os.environ.copy();env.update(EVSP_EXPECTED_COMMIT=c['execution_commit'],EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=c['source_result_sha256'],EVSP_MIP_EXPECTED_JOURNAL_SHA256=c['source_journal_sha256'],EVSP_TARGET_CASE=str(spec),EVSP_TARGET_TOOLING=json.dumps(manifest['tooling_sha256']),EVSP_NATIVE_SOURCE=str(Path(c['source_code'])/'src/run_exact_pool_mip.py'))
   w.save(attempt/'state.json',{**state,'status':'running'})
   ex=w.run_process(argv,attempt,c['source_code'],c['watchdog_s'],env);v=w.read(out);identity_gate(v,c)
   for p,h in c['static_hashes'].items():w.require_hash(p,h)
   w.require_hash(c['input_path'],c['input_sha256'])
   done={**state,'status':'finished','result_path':str(out),'result_sha256':w.sha(out),'execution':ex,'ended_utc':w.now(),'proof_classification':v['proof_classification'],'buses':v['buses']}
   w.save(stage/'completion.json',done);w.save(attempt/'state.json',done)
  except BaseException as e:
   w.save(attempt/'state.json',{**state,'status':'execution_failed','error':repr(e),'ended_utc':w.now()});raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();execute(a.root,a.case)
