from pathlib import Path
import copy,csv,json
import worker as w
import seed_logic
B=Path(__file__).resolve().parent
OLD=B.parent/'overnight_parallel_20260914'
D=Path('/share/scaglione/nc437/evsp-dr/compact_seed_support_20260914')
def main():
 assert not (B/'manifest.json').exists()
 old=w.read(OLD/'manifest.json');cases={};pairs=[]
 D.mkdir(parents=True,exist_ok=True);(D/'cases').mkdir(exist_ok=True);(B/'cases').symlink_to(D/'cases');(B/'logs').mkdir(exist_ok=True)
 for prior in old['pairs']:
  pid=prior['pair_id'];base=old['cases'][pid+'_integer'];a=copy.deepcopy(prior)
  for path,h in [(a['parent_status_path'],a['parent_status_sha256']),(a['mip_bound_parent_path'],a['mip_bound_parent_sha256']),(a['parent_mip_path'],a['parent_mip_sha256']),(a['parent_input_path'],a['parent_input_sha256']),(a['parent_journal_path'],a['parent_journal_sha256'])]:w.require_hash(path,h)
  parent=w.read(a['mip_bound_parent_path']);original=w.read(a['parent_status_path']);mip=w.read(a['parent_mip_path'])
  assert original['final_lp']==parent['final_lp'] and mip['source_result_sha256']==a['mip_bound_parent_sha256'] and mip['source_journal_sha256']==a['parent_journal_sha256']
  assert parent['provenance']['instance_sha256']==a['parent_input_sha256']
  def rows(p):
   with open(p) as f:return {r['Ordered_Trip_ID']:r for r in csv.DictReader(f)}
  pr=rows(a['parent_input_path']);ch=rows(base['input_path']);assert pr.keys()<ch.keys()
  for key,r in pr.items():assert {k:v for k,v in r.items() if k!='count_trip_id'}=={k:v for k,v in ch[key].items() if k!='count_trip_id'}
  with open(a['parent_journal_path']) as f:arms,selection=seed_logic.choose(parent,mip,(json.loads(l) for l in f if l.strip()))
  a.update(selection=selection,parent_journal_hash_scope='Full immutable journal consumed and SHA-256 verified',count_scope='Core preserved; filler has distinct native trip-set incidences. Actual replay and child additions are measured separately.');a.pop('selected_sequence_count',None);pairs.append(a)
  for arm,records in arms.items():
   cid=pid+'_'+arm;p=B/'seeds'/cid;p.mkdir(parents=True);journal=p/'sequences.jsonl';journal.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records));audit=a|selection['arms'][arm]
   seed=p/'seed.json';w.save(seed,seed_logic.status(parent,journal,arm,audit))
   c=copy.deepcopy(base);c.update(id=cid,treatment="previous_k_"+arm,pair_id=pid,seed=audit,seed_status_path=str(seed),seed_status_sha256=w.sha(seed),seed_journal_sha256=w.sha(journal),changed_factor='Mandatory previous integer plus all positive LP support; optional deterministic filler to 512 native incidences',interpretation='Previous-k compact support comparison; no current-k seeds or parent certificates transferred')
   c['argv'][c['argv'].index('--inherit-event-pool-from')+1]=str(seed)
   c['static_hashes']={k:v for k,v in c['static_hashes'].items() if '/seeds/' not in k};c['static_hashes'].update({str(seed):w.sha(seed),str(journal):w.sha(journal)})
   cases[cid]=c
   child=copy.deepcopy(old['cases'][pid+'_integer_mip']);child.update(id=cid+'_mip',treatment='previous_k_'+arm,source_case=cid,pair_id=pid,seed=audit);cases[child['id']]=child
 m=copy.deepcopy(old);m.update(schema='evsp-compact-seed-support-v1',prepared_utc=w.now(),cases=cases,pairs=pairs,storage_root=str(D),frozen_previous_campaign_manifest_sha256=w.sha(OLD/'manifest.json'),interpretation=['Previous-k MIP integer witness plus ALL positive LP support core.','Second arm fills only absent native incidences to 512; core never dropped.','Historical upstream costs retained; child replay costs measured.','No current-k seed or transferred pricing certificate.'],tooling_sha256={n:w.sha(B/n) for n in ['worker.py','worker.sub','seed_logic.py','prepare.py','submit.py','native_smoke.py','test_design.py']})
 assert len(cases)==72;w.save(B/'manifest.json',m)
 for c in cases.values():
  if c['kind']=='cg':w.preflight(B,m,c)
  else:w.check_code(c['source_code'],c['execution_commit']);w.require_hash(c['input_path'],c['input_sha256'])
 w.save(B/'preflight.json',dict(status='passed',manifest_sha256=w.sha(B/'manifest.json'),cg_cases=36,mip_static_checks=36));print(w.sha(B/'manifest.json'))
if __name__=='__main__':main()
