"""Read-only gate, source, matrix, start and selected-route identity audit; no solver."""
from pathlib import Path
import json,subprocess
P=Path(__file__).resolve().parent
script=r'''
from pathlib import Path
import json,hashlib
root=Path('/home/nc437/ladder-lite/mip_structure_20260922')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
base=json.loads((root/'manifest.json').read_text());v3=json.loads((root/'manifest_v3.json').read_text());old=json.loads((root/'jobs.json').read_text());new=json.loads((root/'recovery_v3.json').read_text());active={k:v for k,v in old.items() if '__prepare' not in k};active.update({k:v for k,v in new['jobs'].items() if '__prepare' not in k});assert len(active)==25
out={'checks':[],'preparations':{},'rows':[],'physics':base['model']['physics'],'model':base['model'],'source_manifests':{}}
def check(label,value):
 out['checks'].append({'check':label,'passed':bool(value)})
check('model unchanged across original and v3',base['model']==v3['model'])
for name,m in [('manifest.json',base),('manifest_v3.json',v3)]:
 code=root/('code_v3' if 'v3' in name else 'code');checks={f:sha(code/f)==h for f,h in m['tooling_sha256'].items()};check(name+' code source hashes',all(checks.values()));out['source_manifests'][name]={'sha256':sha(root/name),'code_commit':m['code_commit'],'source_hash_checks':checks}
for case in base['cases']:
 cid=case['id'];g=json.loads((root/'prepared'/f'{cid}.json').read_text());check(cid+' complete gate hash',sha(g['complete'])==g['sha256']);c=json.loads(Path(g['complete']).read_text());meta=json.loads(Path(c['matrix_metadata']).read_text());endpoint=json.loads(Path(case['endpoint']).read_text())
 check(cid+' npz actual bytes hash',sha(c['matrix_file'])==c['matrix_file_sha256']==meta['matrix_file_sha256']);check(cid+' metadata hash',sha(c['matrix_metadata'])==c['matrix_metadata_sha256']);check(cid+' native original ordered pool',meta['ordered_native_pool_sha256']==case['original_prepared_ordered_pool_sha256']==endpoint['physical_pool_audit']['mip_ordered_pool_sha256']);check(cid+' original endpoint hash',sha(case['endpoint'])==case['pins'][case['endpoint']]);check(cid+' pinned source declarations',meta['pins']==case['pins']);check(cid+' native physical admission',meta['physical_pool_audit']['rejected_columns']==meta['physical_pool_audit']['deterministically_repaired']==0)
 out['preparations'][cid]={k:meta.get(k) for k in ['rows','columns','nonzeros','matrix_identity_sha256','matrix_file_sha256','ordered_native_pool_sha256','physical_pool_audit','strong_start_available','signed_costs','source_integer_witness_cost','source_stage2_solver_objective','source_integer_witness_minus_solver_objective']};out['preparations'][cid]['complete_path']=g['complete'];out['preparations'][cid]['metadata_sha256']=c['matrix_metadata_sha256']
 for arm in base['arms']:
  key=cid+'__'+arm;jid=active[key]['job_id'];attempts=list((root/'results'/cid/arm).glob(jid+'_r*'));complete=[a for a in attempts if (a/'COMPLETE.json').exists()];check(key+' single completed active attempt',len(complete)==1)
  if len(complete)!=1:continue
  p=complete[0];r=json.loads((p/'result.json').read_text());e=json.loads((p/'execution.json').read_text());done=json.loads((p/'COMPLETE.json').read_text());manifest_path=e.get('manifest_path',str(root/'manifest.json'));m=json.loads(Path(manifest_path).read_text());expected=meta['strong_start'] if arm=='strong_start' else meta['greedy_start']
  checks={'result_complete_hash':sha(p/'result.json')==done['result_sha256'],'full_log_hash':sha(p/'gurobi.log')==r['log_sha256'],'execution_manifest_hash':sha(manifest_path)==e['manifest_sha256'],'runner_core_pins':e['script_sha256']==m['runner_sha256'] and e['core_sha256']==m['core_sha256'],'matrix_identity':r['matrix_identity_sha256']==meta['matrix_identity_sha256'],'metadata_hash':r['prepared_metadata_sha256']==c['matrix_metadata_sha256'],'start_indices':r['start_indices']==expected,'start_count':r['start_buses']==len(expected),'selected_route_identity':[meta['route_hashes'][j] for j in r['selected_indices']]==r['selected_route_hashes'],'selected_count':len(r['selected_indices'])==r['fleet'],'same_dimensions':r['model_dimensions']=={'rows':case['rows'],'variables':case['columns'],'nonzeros':case['nonzeros']},'no_shared_capacity':r['shared_capacity_enforced'] is False,'saved_start_native_indices':arm!='strong_start' or expected==endpoint['two_stage']['stage1_selected_route_indices']}
  for name,value in checks.items():check(key+' '+name,value)
  out['rows'].append({'key':key,'job_id':jid,'attempt':str(p),'checks':checks,'code_commit':e['code_commit'],'manifest_sha256':e['manifest_sha256'],'result_sha256':sha(p/'result.json'),'log_sha256':sha(p/'gurobi.log')})
out['passed']=all(c['passed'] for c in out['checks']);print(json.dumps(out))
'''
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','unicorn','python3 -'],input=script,text=True,capture_output=True)
if r.returncode:print(r.stderr);r.check_returncode()
d=json.loads(r.stdout);(P/'source_identity_audit.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({'passed':d['passed'],'checks':len(d['checks']),'active_cells':len(d['rows'])}))
