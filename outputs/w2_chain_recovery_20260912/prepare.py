from pathlib import Path
import json,hashlib,shutil,subprocess,copy,os
BASE=Path('/home/nc437/ladder-lite/w2_chain_recovery_20260912');OLD=Path('/home/nc437/ladder-lite/overnight_extension_20260912');FIX=Path('/home/nc437/ladder-lite/w2_k14_import_fix_20260912');CODE=FIX/'code';MIP=Path('/home/nc437/ladder-lite/execution/871d057e1067411f09581e37d78f7c1ca43f68bb');PIN='68fce0093ec9768392442fe1b107a1b67ab0cb7b';PYTHON='/home/nc437/evsp_env/bin/python'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
assert not (BASE/'manifest.json').exists()
BASE.mkdir(exist_ok=True);(BASE/'inputs').mkdir(exist_ok=True);(BASE/'logs').mkdir(exist_ok=True)
assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==PIN
assert not subprocess.check_output(['git','-C',str(CODE),'status','--porcelain','--untracked-files=no'],text=True).strip()
parent=FIX/'cases/w2_k14/cg.json';pd=json.loads(parent.read_text());end=json.loads((parent.parent/'end.json').read_text())
assert end['returncode']==0 and not end['watchdog_fired'] and pd['certified_rc_optimal'] and pd['final']['artificials']==0
assert sha(parent)=='f82e815f0c2e02511ab9e3dee599b05c3a1d2ffb374977d0f040a1759dc52415'
assert sha(pd['columns_journal'])=='d9248ea26caa1af5e86070f9d1fb29a516bd871cdde4535819dcba8601d69013'
shutil.copy2(parent,BASE/'inputs/parent_original.json');shutil.copy2(pd['columns_journal'],BASE/'inputs/parent.columns.jsonl');pd['columns_journal']=str(BASE/'inputs/parent.columns.jsonl');save(BASE/'inputs/parent_replay.json',pd)
# Cache producer semantics were audited against this exact code in the prior retry.
att=FIX/'cache_compatibility.json';ad=json.loads(att.read_text());assert ad['target_head']==PIN and not ad['target_status'];assert all(v['all_equal'] for k,v in ad['ast_checks'].items() if k!='_file_sha256');assert all(ad['default_branch_checks'].values())
case=json.loads((OLD/'manifest.json').read_text())['cases']['w2_k15'];assert sha(CODE/'data'/case['csv'])==case['input_sha256']==sha(OLD/'code/data'/case['csv'])
cache=Path(case['source_cache']);producer=json.loads(Path(str(cache)+'.manifest.json').read_text());assert producer['identity']['instance_sha256']==case['input_sha256'];assert sha(cache)==producer['pickle_sha256']
os.link(cache,BASE/'inputs/network.pkl');save(BASE/'inputs/cache_producer_manifest.json',producer)
consumer=copy.deepcopy(producer);consumer['build_identity']=producer['identity'];consumer['identity']['git_commit']=PIN;consumer['source_manifest_sha256']=sha(Path(str(cache)+'.manifest.json'));consumer['compatibility_attestation_sha256']=sha(att);consumer['compatibility_scope']='Same audited producer/consumer graph code as w2k14, separately verified k15 input and actual pickle hash; no graph edits.';save(BASE/'inputs/network.pkl.manifest.json',consumer)
required=[BASE/'inputs/parent_original.json',BASE/'inputs/parent_replay.json',BASE/'inputs/parent.columns.jsonl',BASE/'inputs/network.pkl.manifest.json',att,CODE/'data'/case['csv'],CODE/'data/hourly_prices_flat.csv',CODE/'data/Ref_dict.csv',CODE/'data/par_ref_dhd.csv']
required=[{'path':str(p),'sha256':sha(p)} for p in required]
worker=CODE/'scripts/retry_inherited_import.py';assert worker.exists()
orig=json.loads((FIX/'manifest.json').read_text());argv=orig['argv'][:];out=BASE/'cases/w2_k15/cg.json'
changes={'--csv':case['csv'],'--out':str(out),'--phase-telemetry':str(out)+'.phase-telemetry.jsonl','--gurobi-log':str(out)+'.gurobi.log','--inherit-event-pool-from':str(BASE/'inputs/parent_replay.json'),'--event-network-cache':str(BASE/'inputs/network.pkl')}
for k,v in changes.items():argv[argv.index(k)+1]=v
cg={'code':str(CODE),'execution_commit':PIN,'required_files':required,'output':str(out),'argv':argv,'watchdog_seconds':14520,'scope':'Existing w2k15 continuation, corrected k14 parent, same science; no inherited proof'}
save(BASE/'cg15_manifest.json',cg)
def mip_manifest(k,status):
 target=BASE/f'cases/w2_k{k}/mip/onehour/result.json'
 args=[PYTHON,str(MIP/'src/run_exact_pool_mip.py'),'--result',str(status),'--data-dir',str(CODE/'data'),'--reference-data-dir',str(CODE/'data'),'--cover','--two-stage','--timelimit','3600','--stage1-timelimit','1800','--threads','8','--mipgap','0.0001','--gurobi-log',str(target.with_suffix('.gurobi.log')),'--out',str(target)]
 # The entry point binds expected output and journal hashes after predecessor completion.
 return {'code':str(MIP),'execution_commit':MIP.name,'required_files':[x for x in required if 'network' not in x['path']],'output':str(target),'argv':args,'watchdog_seconds':6000,'source_status':str(status),'data_execution_commit':PIN,'scope':'3600s two-stage covering pool MIP; unchanged 1800s first stage; unique attempt'}
save(BASE/'mip14_manifest.json',mip_manifest(14,parent));save(BASE/'mip15_manifest.json',mip_manifest(15,out))
shell='''#!/bin/bash
set -euo pipefail
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LM_LICENSE_FILE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GRB_LICENSE_FILE=/share/apps/software/gurobi/gurobi.lic
source /home/nc437/ladder-lite/w2_k14_import_fix_20260912/code/scripts/event_uniform_envelope/gurobi_worker_preflight.sh
exec /home/nc437/evsp_env/bin/python -u /home/nc437/ladder-lite/w2_chain_recovery_20260912/entry.py "$1"
'''
(BASE/'worker.sub').write_text(shell)
entry='''from pathlib import Path
import json,os,runpy,sys,hashlib
p=Path(sys.argv[1]);m=json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
if 'source_status' in m:
 d=json.loads(Path(m['source_status']).read_text());assert d['final']['artificials']==0 and d['final']['iter']>0
 os.environ['EVSP_EXPECTED_COMMIT']=m['execution_commit'];os.environ['EVSP_REQUIRE_DETACHED']='1'
 os.environ['EVSP_MIP_EXPECTED_RESULT_SHA256']=sha(m['source_status']);os.environ['EVSP_MIP_EXPECTED_JOURNAL_SHA256']=sha(d['columns_journal'])
sys.argv=['retry_inherited_import.py',str(p)]
runpy.run_path('/home/nc437/ladder-lite/w2_k14_import_fix_20260912/code/scripts/retry_inherited_import.py',run_name='__main__')
'''
compile(entry,'entry.py','exec');(BASE/'entry.py').write_text(entry)
manifest={'schema':'evsp-chain2-recovery-v1','cg_execution_commit':PIN,'mip_execution_commit':MIP.name,'completed_parent_job':'15687','completed_parent_status_sha256':sha(parent),'completed_parent_journal_sha256':sha(json.loads(parent.read_text())['columns_journal']),'original_blocked_jobs_preserved':['949704','949705','949706'],'cases':{'w2_k14':{'source_status':str(parent)},'w2_k15':case},'files':[{ 'path':str(p),'sha256':sha(p)} for p in [BASE/'cg15_manifest.json',BASE/'mip14_manifest.json',BASE/'mip15_manifest.json',BASE/'worker.sub',BASE/'entry.py']], 'resource_policy':'default partition, exclude scaglione-compute-01; two independent successors no arbitrary throttle; k15 MIP after CG; no requeue because exclusive attempt claims require distinct retries','cache_sha256':producer['pickle_sha256'],'parent_replay_descriptor_changes':'journal path only; original and frozen parent bytes retained'}
save(BASE/'manifest.json',manifest);print(json.dumps({'prepared':True,'root':str(BASE),'manifest_sha256':sha(BASE/'manifest.json')}))
