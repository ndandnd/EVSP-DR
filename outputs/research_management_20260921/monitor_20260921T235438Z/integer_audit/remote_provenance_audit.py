"""Read-only provenance audit run on Unicorn; exports hashes, never large pools."""
from pathlib import Path
import hashlib,json
P=Path('/home/nc437/ladder-lite/integer_columns_20260921');m=json.loads((P/'manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
out={'fresh_inputs':{},'own_journal_exports':[],'code':{}}
for c,v in m['cases'].items():
 out['fresh_inputs'][c]={k:dict(path=v[k],actual_sha256=sha(v[k]),expected_sha256=v[k+'_sha256']) for k in ['fresh_cg_result','fresh_journal']}
for p in sorted((P/'results').glob('*/*/*/dive/dive_incumbent.json')):
 ex=json.loads(p.read_text());d=json.loads((p.parent/'manifest.json').read_text());journal=Path(d['augmented_pool']['augmented_journal']);wanted=dict(zip(ex['source_record_ordinals'],ex['record_sha256']));found={}
 with journal.open() as f:
  ordinal=0
  for line in f:
   if not line.strip():continue
   ordinal+=1
   if ordinal in wanted:
    r=json.loads(line);found[ordinal]=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 out['own_journal_exports'].append(dict(export=str(p),export_sha256=sha(p),journal=str(journal),journal_sha256=sha(journal),expected_journal_sha256=ex['source_journal_sha256'],ordinal_hashes=found,all_selected_records_match=found==wanted))
for rel in ['scripts/research/diving_pricing_20260919/run_replication.py','src/diving_pricing_pilot.py','src/run_exact_pool_mip.py']:
 out['code'][rel]={'sha256':sha(P/'code'/rel)}
print(json.dumps(out,indent=2))
