from pathlib import Path
import json,os,sys
import gate
import process_worker as w
B=Path(__file__).resolve().parent
m=w.read(B/'manifest.json');s=B/'native_smoke';s.mkdir(exist_ok=True)
# This manifest is a validation artifact, never used for production endpoints.
v=w.read(s/'manifest.json');gate.run(s,'noop')
c=v['cases']['tiny'];attempt=s/'tiny_attempt';attempt.mkdir(exist_ok=False)
for p,d in c['static_hashes'].items():w.require_hash(p,d)
args=[a.replace('{attempt}',str(attempt)) for a in c['argv']]
r=w.run_process(args,attempt,m['source_code'],900,os.environ.copy())
meta,digest=gate.validate_cache(attempt/'network.pkl',attempt/'network.pkl.manifest.json',c)
w.save(s/'validation.json',dict(status='passed',production_manifest_sha256=w.sha(B/'manifest.json'),noop=w.read(s/'cases/noop/completion.json'),tiny_native_cache_identity=meta['identity'],tiny_native_cache_sha256=digest,tiny_execution=r,scope='Authenticated existing native cache no-op plus one real-trip same-source native graph construction; timeout branching separately unit-tested.'))
print(json.dumps(dict(status='passed',tiny_wall_s=r['wall_s'],tiny_pickle_bytes=meta['pickle_bytes'])))
