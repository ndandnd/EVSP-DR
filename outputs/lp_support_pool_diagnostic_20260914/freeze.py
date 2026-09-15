from pathlib import Path
import copy,json
import worker as w
B=Path(__file__).resolve().parent;H=B.parent
NAMES=['worker.py','mip_worker.py','worker.sub','native_union.py','construct.py','prepare_index.py','freeze.py','submit.py','test_construct.py']
def main():
 assert not (B/'manifest.json').exists();ix=w.read(B/'source_index.json');old=w.read(H/'compact_seed_support_20260914/manifest.json');cases={};records=[]
 for pair,g in ix['groups'].items():
  for role,src in [('donor',g['donor'])]+[('recipient',r) for r in g['recipients']]:
   for path,digest in [('status_path','status_sha256'),('journal_path','journal_sha256'),('mip_evidence_path','mip_evidence_sha256')]:w.require_hash(src[path],src[digest])
   sv=w.read(src['status_path']);sm=w.read(src['mip_evidence_path']);assert Path(sv['columns_journal']).resolve()==Path(src['journal_path']).resolve()
   assert sm['fleet_proven'] and sm['physical_replay_validated']
   assert (sm['buses']==g['target_k']) if role=='donor' else (sm['buses']>g['target_k'])
  p=B/'pools'/(pair+'_prepared.json');built=w.read(p);assert built['source_index_sha256']==w.sha(B/'source_index.json');records.append(built)
  for r in built['outputs']:
   for a,b in [('status_path','status_sha256'),('journal_path','journal_sha256'),('construction_path','construction_sha256')]:w.require_hash(r[a],r[b])
   status=w.read(r['status_path']);assert status['final']['iter']==0 and status['certified_rc_optimal'] is False
   cid=r['case_id'];c=copy.deepcopy(old['cases'][pair+'_core_mip']);c.pop('source_case',None);c.pop('seed',None);c.update(id=cid,pair_id=pair,treatment=r['selection']['treatment'],source_status=r['status_path'],source_status_sha256=r['status_sha256'],source_journal_sha256=r['journal_sha256'],source_artifact_kind='finite_pool_union',pool_construction_path=r['construction_path'],pool_construction_sha256=r['construction_sha256'],pool_selection=r['selection'],changed_factor='Novel final-positive LP support versus equally many hash-ranked LP-zero donor incidences; separate donor-support-only controls',interpretation='Selected difficult recipient pools, not random sample. Constructed pool has no new CG certificate; finite-pool proof and physical replay reported separately.')
   c['static_hashes'].update({r['construction_path']:r['construction_sha256'],str(B/'source_index.json'):w.sha(B/'source_index.json')})
   c['solver_budget_s']=12600;c['watchdog_s']=15300;c['resources']={'cpus':8,'mem':'24G','allocation_s':16200};c['stage1_budget_s']=10800
   a=c['argv'];a[a.index('--timelimit')+1]='12600';a[a.index('--stage1-timelimit')+1]='10800'
   cases[cid]=c
 assert len(cases)==35
 m=dict(schema='evsp-lp-support-pool-diagnostic-v1',prepared_utc=w.now(),cases=cases,groups=ix['groups'],source_index_path=str(B/'source_index.json'),source_index_sha256=w.sha(B/'source_index.json'),pool_constructions=records,tooling_sha256={n:w.sha(B/n) for n in NAMES},baseline_physics=old['baseline_physics'],execution_budgets={'mip_total_s':12600,'fleet_stage_s':10800},storage_root=ix['storage_root'],selection=ix['selection'],interpretation='13 recipient pools ×2 matched-size augmentations plus9 unique donor-support-only pools.35 independent MIPs, no CG dependencies. Donor weights reconstruct fractional source LP; synthetic pool does not acquire source pricing certificate. Source compute costs retained in source index.',resource_policy_sha256=w.sha(H/'SCAGLIONE_RESOURCE_POLICY.md'))
 w.save(B/'manifest.json',m)
 checks=[]
 for cid,c in cases.items():
  w.check_code(c['source_code'],c['execution_commit']);w.require_hash(c['input_path'],c['input_sha256'])
  for p,h in c['static_hashes'].items():w.require_hash(p,h)
  s=w.read(c['source_status']);assert s['provenance']['instance_sha256']==c['input_sha256'];assert '--cover' in c['argv'] and '--two-stage' in c['argv']
  checks.append({'case_id':cid,'status':'passed','pool_columns':s['final']['pool_columns'],'added_columns':c['pool_selection']['added_columns']})
 w.save(B/'preflight.json',{'status':'passed','manifest_sha256':w.sha(B/'manifest.json'),'checks':checks});print(w.sha(B/'manifest.json'))
if __name__=='__main__':main()
