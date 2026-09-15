"""Publish validation only after both independent native CG/MIP smoke gates."""
from pathlib import Path
import worker as w
B=Path(__file__).resolve().parent;S=B.parent/(B.name+'_smoke')
def main():
 m=w.read(B/'manifest.json');sm=w.read(S/'manifest.json');results=[]
 for arm in ['core','core512']:
  cid='c5_k25_'+arm;cg=w.read(S/'cases'/cid/'completion.json');mi=w.read(S/'cases'/(cid+'_mip')/'completion.json')
  for c in [cg,mi]:w.require_hash(c['result_path'],c['result_sha256']);assert c['usable'] and c['status']=='finished'
  w.require_hash(cg['journal_path'],cg['journal_sha256']);v=w.read(cg['result_path']);mv=w.read(mi['result_path']);audit=v['inherited_event_pool_audit']
  assert audit['source_status_sha256']==m['cases'][cid]['seed_status_sha256'] and audit['source_journal_sha256']==m['cases'][cid]['seed_journal_sha256']
  assert audit['selected_for_replay']==m['cases'][cid]['seed']['selected_sequence_count'] and audit['replay_completed']==audit['selected_for_replay'] and audit['rejected_columns']==0
  assert audit['inherited_lp_certificate'] is False and audit['inherited_duals'] is False and audit['inherited_basis'] is False
  assert mv['physical_replay_validated'] is True
  assert mv['source_result_sha256']==cg['result_sha256'] and mv['source_journal_sha256']==cg['journal_sha256']
  assert mv['physical_pool_audit']['rejected_columns']==0 and mv['physical_pool_audit']['deterministically_repaired']==0
  assert cg['execution_commit']==m['cases'][cid]['execution_commit']
  results.append(dict(case_id=cid,selected=m['cases'][cid]['seed']['selected_sequence_count'],inheritance=audit,cg_completion=cg,mip_completion=mi,physical_pool_audit=mv['physical_pool_audit']))
 for n,h in m['tooling_sha256'].items():w.require_hash(B/n,h)
 w.save(B/'validation.json',dict(status='passed',manifest_sha256=w.sha(B/'manifest.json'),native_smoke_manifest_sha256=w.sha(S/'manifest.json'),native_smoke=results,scope='Native compatibility and physical replay of one-iteration output, not production scientific outcome'))
 print('passed')
if __name__=='__main__':main()
