import hashlib,json,subprocess
from pathlib import Path
base=Path('/home/nc437/ladder-lite')
w=base/'integer_columns_k15_20260921'; w.mkdir(exist_ok=True)
old=json.loads((base/'cumulative_budget_20260913/manifest.json').read_text())
long=json.loads((base/'review_fresh_k15_longmip_20260917/manifest.json').read_text())
t=json.loads((base/'integer_columns_20260921/code/scripts/research/diving_pricing_20260919/cases_k08.json').read_text())
t['cases']={}
for cell in long['cells']:
 if cell['arm']!='plain':continue
 key=cell['case'];c=old['cases'][key]; p=Path(cell['result']);s=json.loads(p.read_text())
 def sha(path):
  h=hashlib.sha256()
  with open(path,'rb') as f:
   for b in iter(lambda:f.read(1<<20),b''):h.update(b)
  return h.hexdigest()
 assert sha(p)==cell['result_sha256']
 assert sha(s['columns_journal'])==cell['journal_sha256']
 cm=Path(c['cache_manifest_path']);cache=Path(c['cache_path']);assert cm.is_file() and cache.is_file()
 t['cases'][key]={'chain':int(key[1]),'target_k':15,'fresh_cg_result':str(p),'fresh_cg_result_sha256':cell['result_sha256'], 'fresh_journal_sha256':cell['journal_sha256'],'csv':c['csv'],'instance_sha256':c['input_sha256'],'event_network_cache':str(cache),'event_network_cache_manifest':str(cm),'event_network_cache_manifest_sha256':c['cache_manifest_sha256'],'cache_identity':c['cache_identity'],'target_external_graph_build_s':c['target_external_graph_build_s'],'fresh_cg_certified':s.get('certified_rc_optimal'),'cache_bytes_observed':cache.stat().st_size}
(w/'six_chain_inventory.json').write_text(json.dumps(t,indent=1))
t['cases']={k:v for k,v in t['cases'].items() if k in ['c1_k15','c3_k15','c5_k15']}
t['note']='Fresh C1/C3/C5 selected before this experiment: C1 hard18 and C3/C5 best16 in prior12h arms; conditional matched pilot.'
(w/'cases_k15.json').write_text(json.dumps(t,indent=1))
print({k:v['cache_bytes_observed'] for k,v in t['cases'].items()})
