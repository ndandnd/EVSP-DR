from pathlib import Path
import sys,tempfile,json,hashlib,unittest.mock as mock
B=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(B));import atomic_pool_copy as module
root=Path(tempfile.mkdtemp(prefix='action3-atomic-pool-'));source=root/'source.jsonl';source.write_bytes(b'{"route":1}\n'*200000);data=source.read_bytes();tests=[]
output=root/'a/pool.jsonl';receipt=module.atomic_pool_copy(source,output);assert output.read_bytes()==data and source.read_bytes()==data;assert receipt['bytes']==len(data);tests.append({'test':'complete_hash_matched_copy','passed':True})
try:module.atomic_pool_copy(source,output);raise AssertionError('overwrite accepted')
except FileExistsError:pass
assert output.read_bytes()==data;tests.append({'test':'existing_destination_preserved','passed':True})
output=root/'b/pool.jsonl'
with mock.patch.object(module.os,'replace',side_effect=OSError('injected interruption before publication')):
 try:module.atomic_pool_copy(source,output);raise AssertionError('injected failure missed')
 except OSError:pass
assert not output.exists() and not list(output.parent.iterdir());assert source.read_bytes()==data;tests.append({'test':'interruption_before_replace_no_partial_pool','passed':True})
output=root/'c/pool.jsonl';old=module._file_sha256

def changed(p):
 if Path(p)==source:
  with source.open('ab') as f:f.write(b'{"route":2}\n')
 return old(p)
with mock.patch.object(module,'_file_sha256',side_effect=changed):
 try:module.atomic_pool_copy(source,output);raise AssertionError('changed source accepted')
 except ValueError as exc:assert 'source pool changed' in str(exc)
assert not output.exists() and not list(output.parent.iterdir());tests.append({'test':'concurrent_source_change_rejected','passed':True})
print(json.dumps({'tests':tests,'helper_sha256':hashlib.sha256((B/'atomic_pool_copy.py').read_bytes()).hexdigest(),'no_solver_or_scheduler_calls':True},indent=2))
