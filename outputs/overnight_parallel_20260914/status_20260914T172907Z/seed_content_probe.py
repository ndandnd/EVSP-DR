import pathlib,json,csv,hashlib
B=pathlib.Path('/home/nc437/ladder-lite/overnight_parallel_20260914');C=B.parent/'cumulative_budget_20260913';m=json.load(open(B/'manifest.json'));out={'pairs':[],'fresh_controls':{}}
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for p in m['pairs']:
 cid=p['pair_id'];mapping={}
 with open(p['parent_input_path']) as f:
  for r in csv.DictReader(f):mapping[int(r['count_trip_id'])]=int(r['Ordered_Trip_ID'])
 arms={}
 for arm in ['integer','lpweight']:
  c=m['cases'][cid+'_'+arm];status=json.load(open(c['seed_status_path']));journal=pathlib.Path(status['columns_journal']);routes=[json.loads(x) for x in journal.read_text().splitlines() if x.strip()]
  arms[arm]={'stable_sequences':[[mapping[t] for t in r['trips']] for r in routes],'seed_sha256':sha(c['seed_status_path']),'journal_sha256':sha(journal)}
 out['pairs'].append({'pair_id':cid,'parent_input_sha256':sha(p['parent_input_path']),'parent_k':p['parent_k'],'target_k':p['target_k'],'arms':arms})
for cid in ['c1_k15','c2_k15','c4_k15']:
 root=C/'cases'/cid;completion=json.load(open(root/'base/completion.json'));statuspath=pathlib.Path(completion['result_path']);s=json.load(open(statuspath));args=s.get('provenance',{}).get('args',{});files=[]
 for p in (root/'base').rglob('*'):
  if p.is_file():files.append({'path':str(p),'bytes':p.stat().st_size})
 out['fresh_controls'][cid]={'completion':completion,'status_sha256':sha(statuspath),'status':{k:s.get(k) for k in ['csv','wall_s','certified_rc_optimal','stop_reason','columns_journal','final','snapshot_at_minutes','terminal_pool_snapshot']},'provenance':{k:s.get('provenance',{}).get(k) for k in ['git_commit','git_dirty','instance_sha256','prices_sha256','reference_sha256','deadhead_sha256']},'args':args,'files':files}
print(json.dumps(out))
