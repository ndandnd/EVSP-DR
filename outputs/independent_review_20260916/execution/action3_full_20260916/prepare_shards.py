"""One full-source hash check; write shared read-only shards for all six arms."""
import argparse,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
m=json.loads((root/'manifest.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
source=Path(m['source_sequences']);meta=json.loads(source.with_name('extraction.json').read_text())
assert sha(source)==m['source_sequences_sha256']==meta['sequences_sha256']
assert meta['source_ordered_pool_sha256']==m['source_ordered_pool_sha256'] and meta['unique_ordered_sequences']==254068
rows=[json.loads(line) for line in open(source)];assert len(rows)==254068 and len({r['sequence_sha256'] for r in rows})==254068
rows.sort(key=lambda r:(tuple(r['trip_sequence']),r['sequence_sha256']))
out=root/'shards';out.mkdir(parents=True,exist_ok=False);shards=[]
for i,offset in enumerate(range(0,len(rows),m['chunk_size'])):
 part=rows[offset:offset+m['chunk_size']];path=out/f'{i:03d}.jsonl'
 with open(path,'w') as f:
  for row in part:f.write(json.dumps(row,separators=(',',':'))+'\n')
 shards.append({'shard':i,'count':len(part),'sha256':sha(path),'path':str(path)})
assert sum(r['count'] for r in shards)==254068
(root/'shards.json').write_text(json.dumps({'manifest_sha256':sha(root/'manifest.json'),'source_sequences_sha256':sha(source),'count':254068,'shards':shards},indent=2)+'\n')
print(json.dumps({'full_sequences':len(rows),'shards':len(shards),'array_tasks':len(shards)*6}))
