from pathlib import Path
import argparse,fcntl,os
import worker as w

def main(root,cid):
 b=Path(root).resolve();m=w.read(b/'manifest.json');c=m['cases'][cid];w.safe_id(cid)
 token=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');w.safe_id(token);stage=b/'cases'/cid;attempt=stage/'attempts'/token;attempt.mkdir(parents=True,exist_ok=False)
 state=dict(case_id=cid,kind='mip',attempt=token,manifest_sha256=w.sha(b/'manifest.json'),started_utc=w.now(),status='preflight')
 w.save(attempt/'state.json',state)
 with (stage/'.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   for p,h in m['tooling_sha256'].items():w.require_hash(b/p,h)
   for p,h in c['static_hashes'].items():w.require_hash(p,h)
   w.require_hash(c['input_path'],c['input_sha256']);w.check_code(c['source_code'],c['execution_commit'])
   if (stage/'completion.json').exists():
    old=w.read(stage/'completion.json');assert old['manifest_sha256']==state['manifest_sha256'];w.require_hash(old['result_path'],old['result_sha256']);w.save(attempt/'state.json',state|dict(status='already_complete'));return
   w.require_hash(c['source_status'],c['source_status_sha256']);s=w.read(c['source_status']);j=w.journal_path(c['source_status'],s);w.require_hash(j,c['source_journal_sha256'])
   assert s['artifact_kind']=='finite_pool_union' and not s['optimization_run'] and s['certified_rc_optimal'] is False and s['final']['iter']==0
   assert s['provenance']['instance_sha256']==c['input_sha256'] and s['pool_construction']['union_journal_sha256']==c['source_journal_sha256']
   out=attempt/'result.json';argv=w.expand_argv(c['argv'],out,attempt,c['source_status']);assert argv[argv.index('--out')+1]==str(out)
   env=os.environ.copy();env.update(EVSP_EXPECTED_COMMIT=c['execution_commit'],EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=c['source_status_sha256'],EVSP_MIP_EXPECTED_JOURNAL_SHA256=c['source_journal_sha256'])
   w.register_mip(b,cid,c,token,out);state.update(status='running',source_status_sha256=c['source_status_sha256'],source_journal_sha256=c['source_journal_sha256'],optimization_run=True);w.save(attempt/'state.json',state)
   execution=w.run_process(argv,attempt,c['source_code'],c['watchdog_s'],env);v=w.read(out);assert v['physical_replay_validated'] is True
   a=v['physical_pool_audit'];assert a['rejected_columns']==a['deterministically_repaired']==0
   assert v['source_result_sha256']==c['source_status_sha256'] and v['source_journal_sha256']==c['source_journal_sha256']
   final=state|dict(status='finished',usable=True,result_path=str(out),result_sha256=w.sha(out),execution=execution,physical_replay_validated=True,source_artifact_kind='finite_pool_union',buses=v['buses'],fleet_proven=v['fleet_proven'],fleet_bound=v['fleet_bound'],ended_utc=w.now(),target_k=c['target_k'])
   target=stage/'mip_result.json';tmp=stage/('mip_result.json.link.'+str(os.getpid()));tmp.symlink_to(out.resolve());tmp.replace(target);w.save(stage/'completion.json',final);w.save(attempt/'state.json',final)
  except BaseException as e:w.save(attempt/'state.json',state|dict(status='execution_failed',error=repr(e),ended_utc=w.now()));raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();main(a.root,a.case)
