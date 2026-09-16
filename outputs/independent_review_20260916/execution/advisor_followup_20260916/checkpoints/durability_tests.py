"""Tiny local durability tests; touches only a fresh temporary directory."""
from pathlib import Path
import importlib.util,tempfile,json,hashlib
P=Path(__file__).resolve().parent;sp=importlib.util.spec_from_file_location('pinned_durable',P/'sources/durable_io.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);T=Path(tempfile.mkdtemp(prefix='evsp-json-durability-'));target=T/'status.json';m.atomic_write_json(target,{'generation':1});old=target.read_bytes();events=[];replace=m.os.replace;fsync=m.os.fsync
try:
 def spy_fsync(fd):events.append('file_fsync');return fsync(fd)
 def broken_replace(src,dst):events.append('replace_raises');raise OSError('injected-before-rename')
 m.os.fsync=spy_fsync;m.os.replace=broken_replace
 try:m.atomic_write_json(target,{'generation':2})
 except OSError:pass
 else:raise AssertionError('fault not raised')
 assert target.read_bytes()==old
 assert not list(T.glob('.*.tmp.*'))
finally:m.os.replace=replace;m.os.fsync=fsync
m.atomic_write_json(target,{'generation':3});assert json.loads(target.read_text())=={'generation':3}
# Output lock is a kernel lock, not a permanent lock-file existence prohibition.
with m.exclusive_output_lock(target):
 try:
  with m.exclusive_output_lock(target):pass
 except m.DurableFileError:lock_blocks=True
 else:raise AssertionError('concurrent lock acquired')
with m.exclusive_output_lock(target):reacquired=True
r={'tests':[{'name':'failure_before_atomic_rename','passed':True,'old_complete_status_preserved':True,'temporary_file_cleaned':True,'observed_io_events':events},{'name':'atomic_success','passed':True,'readable_generation':3},{'name':'exclusive_lock','passed':lock_blocks and reacquired,'blocks_concurrent_writer':lock_blocks,'reacquire_after_release':reacquired}],'module_sha256':hashlib.sha256((P/'sources/durable_io.py').read_bytes()).hexdigest(),'limits':'No filesystem/server crash simulated. Parent-directory fsync is absent in pinned implementation; atomic visibility is not a universal host/storage-crash durability guarantee.','temp_root':str(T)};(P/'durability_tests.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
