"""Prelaunch metadata completion; no scientific argv or source artifact changes."""
from pathlib import Path
import worker as w
B=Path(__file__).resolve().parent
m=w.read(B/'manifest.json');prior=w.sha(B/'manifest.json');w.save(B/'manifest.pre_metadata_freeze.json',m)
for key in ['mip_seconds','stage1_seconds']:m['baseline_physics'].pop(key,None)
m['baseline_physics']['inherit_workers']=1;m['baseline_physics']['initial_pool']='singletons plus native replayed previous-k compact sequences'
for c in m['cases'].values():
 if c['kind']=='cg':
  cache=c['argv'][c['argv'].index('--event-network-cache')+1];meta=w.read(cache+'.manifest.json');done=w.read(Path(cache).parent/'cache_result.json');c['target_graph_cost']=dict(original_build_s=meta.get('original_build_s'),preparation_publication=done,scope='Common preexisting target graph; build cost recorded separately from parent CG/MIP and child native wall budget')
m['tooling_sha256']['prepare.py']=w.sha(B/'prepare.py');w.save(B/'manifest.json',m)
v=w.read(B/'preflight.json');v['manifest_sha256']=w.sha(B/'manifest.json');w.save(B/'preflight.json',v)
w.save(B/'metadata_freeze.json',dict(prior_manifest_sha256=prior,final_manifest_sha256=w.sha(B/'manifest.json'),changes=['Historical MIP budget removed from physics; actual replay worker count1','Target graph original build cost separate from source CG/MIP and child wall time','Updated deterministic preparation script hash'],scientific_argv_unchanged=True,script_sha256=w.sha(__file__)))
print(w.sha(B/'manifest.json'))
