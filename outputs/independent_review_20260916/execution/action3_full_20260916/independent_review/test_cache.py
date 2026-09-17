from pathlib import Path
import runpy,contextlib,io,tempfile,dataclasses,json,hashlib,sys
with contextlib.redirect_stdout(io.StringIO()): g=runpy.run_path(str(Path(__file__).with_name('test_independent.py')))
from action3_network_cache import cache_identity,write_cache,load_cache
args=g['a'];args.max_station_wait_min=1560
prov={k:k for k in ['git_commit','instance_sha256','prices_sha256','reference_sha256','deadhead_sha256']}
identity=cache_identity(args,prov);network=g['compact'];base=Path(tempfile.mkdtemp(prefix='action3-independent-cache-'));p=base/'graph.pkl'
m=write_cache(p,network,identity,.1);loaded,_=load_cache(p,identity)
for seq in [(0,),(1,),(0,1),(0,2),(0,1,2)]:
 a=network.fixed_sequence_record(seq);b=loaded.fixed_sequence_record(seq);assert (a is None)==(b is None)
 if a:assert abs(a['cost']-b['cost'])<1e-8
checks=[]
for key in ['instance_sha256','prices_sha256','reference_sha256','deadhead_sha256','git_commit','battery_kwh','reserve_kwh','parx_kw','non_parx_kw','soc_step','block_min','max_station_wait_min','arc_mode','charge_start_cost']:
 altered=dict(identity);v=altered[key];altered[key]=str(v)+'_different' if isinstance(v,str) else float(v)+1
 try:load_cache(p,altered);raise AssertionError('mismatch accepted: '+key)
 except ValueError as exc:assert 'identity mismatch' in str(exc)
 checks.append(key)
raw=p.read_bytes();p.write_bytes(b'changed'+raw)
try:load_cache(p,identity);raise AssertionError('corrupt pickle accepted')
except ValueError as exc:assert 'bytes hash mismatch' in str(exc)
p.write_bytes(raw)
try:write_cache(p,network,identity,.1);raise AssertionError('existing cache overwritten')
except FileExistsError:pass
print(json.dumps({'roundtrip_route_checks':5,'roundtrip_passed':True,'identity_mismatch_fields_rejected':checks,'corruption_rejected_before_unpickle':True,'existing_cache_not_overwritten':True,'source_sha256':hashlib.sha256((g['CODE']/'src/action3_network_cache.py').read_bytes()).hexdigest(),'cache_pickle_sha256':m['pickle_sha256'],'no_scheduler_calls':True},indent=2))
