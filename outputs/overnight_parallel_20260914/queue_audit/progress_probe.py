import json,pathlib
out={}
for name in ['chain_extension_20260913','chain_extension_20260914']:
 b=pathlib.Path('/home/nc437/ladder-lite')/name;m=json.loads((b/'manifest.json').read_text());out[name]={}
 for cid,c in m['cases'].items():
  paths=sorted((b/'cases'/cid/'cache').glob('*/progress.jsonl'),key=lambda p:p.stat().st_mtime)
  if not paths:continue
  rows=[json.loads(x) for x in paths[-1].read_text().splitlines() if x.strip()]
  out[name][cid]={'trips':c['trip_count'],'path':str(paths[-1]),'rows':rows}
print(json.dumps(out))
