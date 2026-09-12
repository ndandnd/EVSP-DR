import copy, hashlib, json, os, shutil, subprocess
from pathlib import Path
BASE=Path('/home/nc437/ladder-lite/w2_k14_import_fix_20260912')
OLD=Path('/home/nc437/ladder-lite/overnight_extension_20260912')
CODE=BASE/'code'
PIN='68fce0093ec9768392442fe1b107a1b67ab0cb7b'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
assert not CODE.exists()
subprocess.run(['git','clone','--shared','--no-checkout',str(OLD/'code'),str(CODE)],check=True)
subprocess.run(['git','-C',str(CODE),'fetch',str(BASE/'fix.bundle'),'HEAD'],check=True)
subprocess.run(['git','-C',str(CODE),'checkout','--detach',PIN],check=True)
assert subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()==PIN
case=json.loads((OLD/'manifest.json').read_text())['cases']['w2_k14']
root=BASE/'cases'/'w2_k14'; root.mkdir(parents=True)
inputs=BASE/'inputs';inputs.mkdir()
parent=Path(case['parent_status']);assert sha(parent)=='9d4487510dd56ec09edfb646584530ba4795bfb0db1994ffb3bafda3164486d1'
raw=parent.read_bytes();(inputs/'parent_original.json').write_bytes(raw)
payload=json.loads(raw);journal=Path(payload['columns_journal'])
shutil.copyfile(journal,inputs/'parent.columns.jsonl')
assert sha(journal)==sha(inputs/'parent.columns.jsonl')
payload['columns_journal']=str(inputs/'parent.columns.jsonl');save(inputs/'parent_replay.json',payload)
source_manifest=OLD/'cases/w2_k14/source_network_manifest.json'
assert sha(source_manifest)=='10c47518d9cc5bdfb908b7eed5176351764e7bffc618fb6d693702fe199e162a'
shutil.copyfile(source_manifest,inputs/'cache_producer_manifest.json')
producer=json.loads(source_manifest.read_text())
cache=OLD/'cases/w2_k14/network.pkl'
assert sha(cache)==producer['pickle_sha256']
os.link(cache,inputs/'network.pkl')
evidence=json.loads((BASE/'cache_compatibility.json').read_text())
assert evidence['target_head']==PIN and not evidence['target_status']
assert all(c['all_equal'] for name,c in evidence['ast_checks'].items() if name!='_file_sha256')
assert all(evidence['default_branch_checks'].values()) and all(i['all_match'] for i in evidence['inputs'])
assert sha(CODE/'src/exact_pricer_expanded.py')==evidence['target_exact_sha256']
consumer=copy.deepcopy(producer);consumer['identity']['git_commit']=PIN
consumer['build_identity']=producer['identity']
consumer['source_manifest_sha256']=sha(source_manifest)
consumer['compatibility_attestation_sha256']=sha(BASE/'cache_compatibility.json')
consumer['compatibility_scope']='Graph/cache/physics equivalence; importer shutdown, cancellation and telemetry changes only; actual pickle hash verified.'
save(inputs/'network.pkl.manifest.json',consumer)
PYTHON='/home/nc437/evsp_env/bin/python';out=root/'cg.json'
argv=[PYTHON,str(CODE/'src/exact_pricer_expanded.py'),'--csv',case['csv'],'--prices_csv','hourly_prices_flat.csv','--time-model','event','--event-arc-mode','lazy','--soc-step','2.5','--block-min','5','--max-iters','50000','--columns_per_iter','30','--column-selection','reduced_cost','--rc-eps','0.0001','--master-sense','cover','--master-backend','gurobi','--initial-pool','singletons','--wall-limit-s','14400','--checkpoint-every','25','--g-kwh','240','--charge-kw','240','--min-soc-frac','0','--phase-telemetry',str(out)+'.phase-telemetry.jsonl','--gurobi-log',str(out)+'.gurobi.log','--out',str(out),'--inherit-event-pool-from',str(inputs/'parent_replay.json'),'--inherit-event-pool-workers','8','--inherit-max-columns','512','--inherit-time-limit-s','900','--event-network-cache',str(inputs/'network.pkl'),'--event-network-cache-mode','require']
required=[inputs/p for p in ['parent_original.json','parent_replay.json','parent.columns.jsonl','cache_producer_manifest.json','network.pkl.manifest.json']]+[BASE/'cache_compatibility.json']
required += [CODE/i['path'] for i in evidence['inputs']]
manifest={'schema':'evsp-import-shutdown-retry-v1','case':case,'execution_commit':PIN,'code':str(CODE),'output':str(out),'argv':argv,'watchdog_seconds':14520,'required_files':[{'path':str(p),'sha256':sha(p)} for p in required], 'source_parent_status':str(parent),'source_parent_status_sha256':sha(parent),'source_parent_journal':str(journal),'source_parent_journal_sha256':sha(journal),'original_failed_job':'949703','original_output':str(OLD/'cases/w2_k14/cg.json'),'cache_pickle_sha256':sha(cache),'cache_verified_bytes':cache.stat().st_size,'original_cache_build_commit':producer['identity']['git_commit'],'cache_build_seconds_not_retry_runtime':producer['original_build_s'],'resources':{'partition':'default_partition','cpus':8,'mem':'96G','time':'04:45:00','exclude':'scaglione-compute-01','requeue':False},'dependencies':{'previous_k_job':'949701','previous_k_artifact_frozen':True,'scheduler_dependency':'afterok:949701','affected_existing_jobs_unchanged':['949704','949705','949706']},'scope':'Only failed w2_k14 retry; graph, physics, objective, prices, selection and solver budgets retained. No CG or fleet claim until validated.'}
save(BASE/'manifest.json',manifest)
print(json.dumps({'manifest':str(BASE/'manifest.json'),'sha256':sha(BASE/'manifest.json'),'source_commit':PIN}))
