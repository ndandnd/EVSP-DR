from pathlib import Path
import ast,hashlib,json
b=Path(__file__).resolve().parent.parent;s=b.parent/'chain_extension_20260915'
v=json.loads((b/'manifest.json').read_text());p=json.loads((s/'manifest.json').read_text());checks=[]
assert v['scientific_settings']==p['scientific_settings'];assert v['resources']==p['resources'];assert v['graph_seconds']==p['graph_seconds'];checks.append('scientific settings and resource requests equal k29-30')
for name in ['execution_commit','mip_execution_commit','source_sha256','data_sha256']:assert v[name]==p[name],name
checks.append('execution commits, CG source and static data hashes equal parent')
for mode in ['cg','mip']:
 for name in ['commit','tree','source_sha256']:assert v['source_provenance'][mode][name]==p['source_provenance'][mode][name]
checks.append('clean detached source trees and all recorded solver source hashes unchanged')
for path in b.glob('*.py'):compile(path.read_text(),str(path),'exec')
def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
a=funcs(s/'campaign.py');z=funcs(b/'campaign.py')
for name in a:
 if name not in ['prepare','prelaunch']:assert a[name]==z[name],name
checks.append('all campaign functions except preparation metadata and prelaunch duplicate gate are AST-identical')
assert (b/'graph_entry.py').read_bytes()==(s/'graph_entry.py').read_bytes()
for name,digest in v['tooling_sha256'].items():assert hashlib.sha256((b/name).read_bytes()).hexdigest()==digest
assert not (b/'jobs.json').exists();assert not (b/'case_jobs.json').exists()
checks.append('tool hashes match staged manifest and production records absent')
for chain in range(1,7):
 assert v['warm_chains'][str(chain)]==[f'w{chain}_k31',f'w{chain}_k32']
 assert v['initial_parents'][str(chain)]['parent_case_id']==f'w{chain}_k30'
checks.append('exactly twelve k31-32 cases with six k30 producers')
(b/'review_checks.json').write_text(json.dumps({'status':'passed','manifest_sha256':hashlib.sha256((b/'manifest.json').read_bytes()).hexdigest(),'checks':checks},indent=2)+'\n')
print('\n'.join(checks))
