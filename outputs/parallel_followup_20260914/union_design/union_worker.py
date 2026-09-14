"""Independent construction stages and hash-gated dependent union MIPs."""
from pathlib import Path
import argparse,fcntl,importlib.util,json,os,sys

def load_worker(root):
 spec=importlib.util.spec_from_file_location('frozen_worker',root/'worker.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main(root,cid):
 root=Path(root).resolve();w=load_worker(root);w.safe_id(cid)
 m=w.read(root/'manifest.json');c=m['cases'][cid];manifest_hash=w.sha(root/'manifest.json')
 token=os.environ['SLURM_JOB_ID']+'_r'+os.environ.get('SLURM_RESTART_COUNT','0');w.safe_id(token)
 stage=root/'cases'/cid;attempt=stage/'attempts'/token;attempt.mkdir(parents=True,exist_ok=False)
 state=dict(case_id=cid,kind=c['kind'],attempt=token,manifest_sha256=manifest_hash,started_utc=w.now(),status='preflight')
 w.save(attempt/'state.json',state)
 with (stage/'.lock').open('a') as lock:
  try:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   for n,h in m['tooling_sha256'].items():w.require_hash(root/n,h)
   for p,h in c['static_hashes'].items():w.require_hash(p,h)
   w.require_hash(c['input_path'],c['input_sha256'])
   if (stage/'completion.json').exists():
    old=w.read(stage/'completion.json');assert old['manifest_sha256']==manifest_hash
    w.require_hash(old['result_path'],old['result_sha256'])
    if old.get('journal_path'):w.require_hash(old['journal_path'],old['journal_sha256'])
    w.save(attempt/'state.json',{**state,'status':'already_complete'});return
   if c['kind']=='pool_construction':
    out=attempt/'pool.json';w.save(attempt/'construction_spec.json',c)
    argv=[sys.executable,str(root/'union_logic.py'),'--spec',str(attempt/'construction_spec.json'),'--out',str(out)]
    state['status']='constructing_pool';w.save(attempt/'state.json',state)
    execution=w.run_process(argv,attempt,root,c['watchdog_s'],os.environ.copy())
    value=w.read(out);assert value['artifact_kind']=='finite_pool_union' and value['certified_rc_optimal'] is False and value['final']['iter']==0
    journal=Path(value['columns_journal']);w.require_hash(journal,value['pool_construction']['union_journal_sha256'])
    final={**state,'status':'finished','usable':True,'optimization_run':False,'result_path':str(out),
     'result_sha256':w.sha(out),'journal_path':str(journal),'journal_sha256':w.sha(journal),
     'construction_path':str(attempt/'construction.json'),'construction_sha256':w.sha(attempt/'construction.json'),
     'execution':execution,'ended_utc':w.now(),'certified':False}
    target=stage/'pool.json'
   else:
    assert c['kind']=='mip';w.check_code(c['source_code'],c['execution_commit'])
    source=w.read(root/'cases'/c['source_case']/'completion.json')
    assert source['kind']=='pool_construction' and source['usable'] and source['manifest_sha256']==manifest_hash
    assert source['case_id']==c['source_case']
    w.require_hash(source['result_path'],source['result_sha256']);w.require_hash(source['journal_path'],source['journal_sha256'])
    status=w.read(source['result_path']);assert status['certified_rc_optimal'] is False and status['artifact_kind']=='finite_pool_union'
    assert Path(status['columns_journal']).resolve()==Path(source['journal_path']).resolve()
    assert status['provenance']['instance_sha256']==c['input_sha256']
    out=attempt/'result.json';argv=w.expand_argv(c['argv'],out,attempt,source['result_path'])
    assert '--out' in argv and argv[argv.index('--out')+1]==str(out)
    env=os.environ.copy();env.update(EVSP_EXPECTED_COMMIT=c['execution_commit'],EVSP_REQUIRE_DETACHED='1',
      EVSP_MIP_EXPECTED_RESULT_SHA256=source['result_sha256'],EVSP_MIP_EXPECTED_JOURNAL_SHA256=source['journal_sha256'])
    # The exact worker's locked append is used once per private MIP attempt.
    w.register_mip(root,cid,c,token,out)
    state.update(status='running',source_status_sha256=source['result_sha256'],source_journal_sha256=source['journal_sha256'],optimization_run=True)
    w.save(attempt/'state.json',state)
    execution=w.run_process(argv,attempt,c['source_code'],c['watchdog_s'],env);value=w.read(out)
    assert value['physical_replay_validated'] is True
    audit=value['physical_pool_audit'];assert audit['rejected_columns']==audit['deterministically_repaired']==0,'Union admission changed; do not publish source-pool equivalence'
    assert value['source_result_sha256']==source['result_sha256'] and value['source_journal_sha256']==source['journal_sha256']
    final={**state,'status':'finished','usable':True,'result_path':str(out),'result_sha256':w.sha(out),'execution':execution,
     'physical_replay_validated':True,'source_artifact_kind':'finite_pool_union','buses':value['buses'],
     'fleet_proven':value['fleet_proven'],'fleet_bound':value['fleet_bound'],'ended_utc':w.now()}
    target=stage/'mip_result.json'
   temporary=target.with_name(target.name+'.link.'+str(os.getpid()));temporary.symlink_to(out.resolve());temporary.replace(target)
   w.save(stage/'completion.json',final);w.save(attempt/'state.json',final)
  except BaseException as e:
   w.save(attempt/'state.json',{**state,'status':'execution_failed','error':repr(e),'ended_utc':w.now()});raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--case',required=True);a=p.parse_args();main(a.root,a.case)
