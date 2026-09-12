"""Read-only source comparison; writes evidence only alongside this script."""
import ast, hashlib, json, subprocess
from pathlib import Path
REPO=Path('/Users/nadan/Documents/projects/demandresponse')
TARGET=Path('/private/tmp/evsp-import-deadline-fix-20260912')
OUT=Path(__file__).parent
PRODUCER='21fbecba826824c44f897feef038fcf51c532582'
BASE='a29992196acb74d02b8c7891be4061718889999f'
def git(*args): return subprocess.check_output(['git','-C',str(REPO),*args])
def sha(b): return hashlib.sha256(b).hexdigest()
def src(ref,path): return git('show',f'{ref}:{path}')
def tree(ref):
 return {line.split(None,3)[3]:line.split()[2] for line in git('ls-tree','-r',ref,'src').decode().splitlines()}
def canon(node): return ast.dump(node,include_attributes=False)
def funcs(text): return {n.name:canon(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.AsyncFunctionDef))}
pt,bt=tree(PRODUCER),tree(BASE)
paths=sorted(set(pt)|set(bt)); inventory=[]
for path in paths:
 local=TARGET/path
 inventory.append(dict(path=path,producer_blob=pt.get(path),base_blob=bt.get(path),producer_base_equal=pt.get(path)==bt.get(path),
                       target_matches_base=local.is_file() and git('hash-object',str(local)).decode().strip()==bt.get(path)))
p='src/exact_pricer_expanded.py'
texts=[src(PRODUCER,p).decode(),src(BASE,p).decode(),(TARGET/p).read_text()]
fs=[funcs(t) for t in texts]
checks={}
for name in ['_file_sha256','_event_network_cache_identity','_event_network_cache_manifest_path','_load_event_network_cache','_write_event_network_cache','ExpandedNetwork']:
 checks[name]=dict(sha256=[sha(f[name].encode()) for f in fs],all_equal=len({f[name] for f in fs})==1)
def segment(t):
 body=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name=='run_cg').body
 start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='prices' for x in n.targets))
 end=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='network_metrics' for x in n.targets))
 return canon(ast.Module(body=body[start:end+1],type_ignores=[]))
segments=[segment(t) for t in texts]
checks['run_cg_prices_through_network_metrics']=dict(sha256=[sha(t.encode()) for t in segments],all_equal=len(set(segments))==1)
def buildcall(t):
 return [canon(n) for n in ast.walk(ast.parse(t)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='build_problem']
checks['build_problem_calls']=dict(values=[buildcall(t) for t in texts],all_equal=buildcall(texts[0])==buildcall(texts[1])==buildcall(texts[2]))
# Only erase the exact optional checkpoint addition, proving its None branch
# reduces structurally to the original implementation; reject other edits.
import copy, io
class DefaultCheckpoint(ast.NodeTransformer):
 def visit_FunctionDef(self,node):
  self.generic_visit(node)
  keep=[]; defaults=[]
  for arg,default in zip(node.args.kwonlyargs,node.args.kw_defaults):
   if arg.arg=='checkpoint':
    assert isinstance(default,ast.Constant) and default.value is None
   else: keep.append(arg); defaults.append(default)
  node.args.kwonlyargs=keep; node.args.kw_defaults=defaults
  return node
 def visit_If(self,node):
  expected=ast.parse('if checkpoint is not None: checkpoint()').body[0]
  if canon(node)==canon(expected): return None
  return self.generic_visit(node)
def default_form(text): return canon(DefaultCheckpoint().visit(ast.parse(text)))
def function_text(text,name):
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return ast.unparse(node)
hash_before=function_text(texts[1],'_file_sha256'); hash_after=function_text(texts[2],'_file_sha256')
default_checks={'hash_helper_None_branch_AST_equal':default_form(hash_before)==default_form(hash_after)}
cache_calls=[]
for name in ['_load_event_network_cache','_write_event_network_cache']:
 node=ast.parse(function_text(texts[2],name))
 for call in ast.walk(node):
  if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='_file_sha256':
   cache_calls.append({'function':name,'call':ast.unparse(call),'omits_checkpoint':len(call.args)==1 and not call.keywords})
default_checks['all_cache_hash_calls_use_None_default']=bool(cache_calls) and all(c['omits_checkpoint'] for c in cache_calls)
durable_before=src(BASE,'src/durable_io.py').decode();durable_after=(TARGET/'src/durable_io.py').read_text()
default_checks['entire_durable_module_None_branch_AST_equal']=default_form(durable_before)==default_form(durable_after)
db,da=funcs(durable_before),funcs(durable_after)
default_checks['other_durable_functions_AST_unchanged']=all(da.get(n)==v for n,v in db.items() if n!='read_jsonl_records') and set(db)==set(da)
# Execute the isolated old/new hash helper, including both chunk boundaries.
def helper(text):
 namespace={'hashlib':hashlib,'Path':Path}
 exec(compile(ast.parse(text),'<audited_hash_helper>','exec'),namespace)
 return namespace['_file_sha256']
old,new=helper(hash_before),helper(hash_after)
class MemoryPath:
 def __init__(self,data): self.data=data
 def open(self,mode): assert mode=='rb'; return io.BytesIO(self.data)
hash_cases=[]
for size in [0,1,1024*1024-1,1024*1024,1024*1024+1,2*1024*1024+7]:
 data=(bytes(range(256))*((size+255)//256))[:size]
 expected=sha(data); a=old(MemoryPath(data));b=new(MemoryPath(data))
 hash_cases.append({'bytes':size,'sha256':expected,'all_equal':a==b==expected})
default_checks['default_hash_boundary_regressions']=all(c['all_equal'] for c in hash_cases)
monitor=OUT.parent/'monitor_supplement.json'; doc=json.loads(monitor.read_text())
entry=next(f for f in doc['files'] if f['path'].endswith('/source_network_manifest.json'))
source_manifest=json.loads(entry['text'])
assert sha(entry['text'].encode())==entry['sha256']
(OUT/'source_manifest.json').write_text(entry['text'])
ident=source_manifest['identity']; inputs=[]
for path,key in [('data/scale_ladder/instances/nested_probability_k2_15_20260908/Practice_Custom_DutyUnion_k14_p02_20260908.csv','instance_sha256'),('data/hourly_prices_flat.csv','prices_sha256'),('data/Ref_dict.csv','reference_sha256'),('data/par_ref_dhd.csv','deadhead_sha256')]:
 vals=[sha(src(PRODUCER,path)),sha(src(BASE,path)),sha((TARGET/path).read_bytes())]
 inputs.append(dict(path=path,expected=ident[key],sha256=vals,all_match=all(v==ident[key] for v in vals)))
critical=['event_pricer_network','audit_giro_known_columns','config','expanded_path_realization','fixed_duty_continuous_optimizer','run_exact_pool_mip','utils_v2','durable_io']
critical_hashes={n:[sha(src(PRODUCER,f'src/{n}.py')),sha(src(BASE,f'src/{n}.py')),sha((TARGET/f'src/{n}.py').read_bytes())] for n in critical}
result=dict(producer=PRODUCER,base=BASE,target=str(TARGET),target_head=subprocess.check_output(['git','-C',str(TARGET),'rev-parse','HEAD'],text=True).strip(),target_status=subprocess.check_output(['git','-C',str(TARGET),'status','--short'],text=True),
            target_exact_sha256=sha(texts[2].encode()),monitor_sha256=sha(monitor.read_bytes()),source_manifest_sha256=entry['sha256'],
            source_manifest_remote_path=entry['path'],source_manifest=source_manifest,source_inventory=inventory,critical_source_sha256=critical_hashes,
            ast_checks=checks,default_branch_checks=default_checks,cache_hash_calls=cache_calls,hash_boundary_cases=hash_cases,inputs=inputs,
            changed_top_level_functions_producer_to_base=[n for n in fs[0] if fs[0][n]!=fs[1].get(n)],
            changed_top_level_functions_base_to_target=[n for n in fs[1] if fs[1][n]!=fs[2].get(n)],
            added_top_level_functions_base_to_target=sorted(set(fs[2])-set(fs[1])),
            conclusion='Graph construction and cache serialization are source-compatible for audited source snapshot; require final fix snapshot re-audit and remote artifact hash/metrics validation before reuse.',
            limitations=['No remote filesystem access or pickle load performed.','Source equivalence proves graph-code compatibility, not the historical pickle contents; loader must verify original pickle hash and metrics.','Target working tree may change after this snapshot.'])
(OUT/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['target_head','target_status','changed_top_level_functions_producer_to_base','changed_top_level_functions_base_to_target']}))
print('AST literal checks', {k:v['all_equal'] for k,v in checks.items()}, 'default branch checks',default_checks, 'input matches',all(v['all_match'] for v in inputs))
print('src differences producer/base',[r['path'] for r in inventory if not r['producer_base_equal']])
print('src differences base/target',[r['path'] for r in inventory if not r['target_matches_base']])
