"""Real whole-instance singleton-pool native MIP compatibility fixture."""
from pathlib import Path
import json,os,subprocess
import prefix_logic as p
B=Path(__file__).resolve().parent;m=p.read(B/'manifest.json');c=m['cases']['c1_k15_prefix_mip'];build=m['cases'][c['source_case']];source=build['source'];p.require(source['status_path'],source['status_sha256']);parent=p.read(source['status_path'])
a=B/'validation'/os.environ['SLURM_JOB_ID'];a.mkdir(parents=True,exist_ok=False);journal=a/'initial.columns.jsonl';routes=0;covered=set()
with open(source['journal_path']) as f,journal.open('x') as out:
 for line in f:
  if not line.strip():continue
  r=json.loads(line)
  if r['found_iter']!=0:break
  out.write(line);routes+=1;covered.update(r['trips'])
assert covered==set(parent['trip_ids'])
status={k:parent[k] for k in p.IDENTITY};status.update(artifact_kind='retrospective_iteration_prefix',optimization_run=False,certified_rc_optimal=False,provenance={k:parent['provenance'][k] for k in p.HASHES},columns_journal=str(journal),final={'iter':0,'artificials':None},stop_reason='native_fixture_not_cg')
statuspath=a/'pool.json';p.save(statuspath,status);result=a/'mip.json';args=c['argv'][:]
args=[x.replace('{source_status}',str(statuspath)).replace('{out}',str(result)).replace('{attempt_dir}',str(a)) for x in args]
args[args.index('--timelimit')+1]='30';args[args.index('--stage1-timelimit')+1]='15';args[args.index('--threads')+1]='2'
env=os.environ.copy();env.update(EVSP_EXPECTED_COMMIT=c['execution_commit'],EVSP_REQUIRE_DETACHED='1',EVSP_MIP_EXPECTED_RESULT_SHA256=p.sha(statuspath),EVSP_MIP_EXPECTED_JOURNAL_SHA256=p.sha(journal))
with (a/'native.log').open('x') as log:r=subprocess.run(args,cwd=c['source_code'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
assert r.returncode==0;r=p.read(result);audit=r['physical_pool_audit'];assert r['physical_replay_validated'] is True;assert audit['rejected_columns']==audit['deterministically_repaired']==0
assert r['source_result_sha256']==p.sha(statuspath) and r['source_journal_sha256']==p.sha(journal)
p.save(B/'validation.json',{'status':'passed','manifest_sha256':p.sha(B/'manifest.json'),'job_id':os.environ['SLURM_JOB_ID'],'native_fixture_scope':'ExactC1k15wholeinput initialsingletonpool, syntheticiter0/artificialunknown/noCGcertificate; proves nativeinput/cost/replay and feasibleintegercover compatibility, not fullprefix efficacy.','initial_records':routes,'covered_trips':len(covered),'buses':r['buses'],'physical_replay_validated':True,'rejected_columns':audit['rejected_columns'],'repaired_columns':audit['deterministically_repaired'],'result_path':str(result),'result_sha256':p.sha(result),'source_status_sha256':source['status_sha256'],'fixture_status_sha256':p.sha(statuspath),'fixture_journal_sha256':p.sha(journal),'native_log':str(a/'native.log')})
print('Nativewholeinstanceprefixfixture passed',flush=True)
