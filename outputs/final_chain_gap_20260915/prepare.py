from pathlib import Path
import copy,json,os,subprocess
import worker as w
B=Path(__file__).resolve().parent; H=B.parent; E=H/'chain_extension_20260913'
IDS=['w5_k25']
def main():
 assert not (B/'manifest.json').exists()
 old=w.read(H/'mip_repeatability_20260914/manifest.json'); em=w.read(E/'manifest.json')
 template=old['cases']['w1_k19_newgap_longmip'];cases={};audit=[]
 manifests=[]
 for path in H.glob('*/manifest.json'):
  if path.parent==B:continue
  try:
   m=w.read(path)
   for cid,c in m.get('cases',{}).items():
    if c.get('kind')=='mip' and c.get('source_status') and c.get('solver_budget_s',0)>3600:manifests.append((str(path),cid,c))
  except (ValueError,AttributeError):continue
 for cid in IDS:
  e=em['cases'][cid];sp=(E/'cases'/cid/'cg.json').resolve();s=w.read(sp);jp=w.journal_path(sp,s);rp=(E/'cases'/cid/'mip_result.json').resolve();r=w.read(rp)
  assert r['physical_replay_validated'] and r['source_journal_sha256']==w.sha(jp)
  assert r['source_result_sha256']==w.sha(sp) or w.read(r['source_result'])==s
  duplicates=[(p,n) for p,n,c in manifests if c.get('input_sha256')==e['input_sha256'] and c.get('source_journal_sha256')==w.sha(jp)]
  if r['buses']<=e['k'] or duplicates:
   audit.append(dict(case_id=cid,skipped=True,buses=r['buses'],duplicates=duplicates));continue
  c=copy.deepcopy(template);c.update(id=cid+'_longmip',chain=e['chain'],target_k=e['k'],target_duties=e['k'],csv=e['csv'],input_path=str(E/'code/data'/e['csv']),input_sha256=e['input_sha256'],source_status=str(sp),source_status_sha256=w.sha(sp),source_journal_sha256=w.sha(jp),source_cg_commit=s['provenance']['git_commit'],original_case=cid,treatment='remaining_original_pool_longer_search',selection_snapshot='20260915T050815Z',comparator={'prior_result':str(rp),'prior_sha256':w.sha(rp)},latest_original_result=str(rp),latest_original_sha256=w.sha(rp),latest_original_buses=r['buses'],latest_original_target_matched=False,interpretation='New tree on unchanged original pool; longer fleet search. Finite-pool proof only; independent structural/global bounds remain separate.')
  c['static_hashes'][str(rp)]=w.sha(rp)
  cases[c['id']]=c;audit.append(dict(case_id=cid,skipped=False,buses=r['buses'],source_result_sha256=w.sha(sp),source_journal_sha256=w.sha(jp),physical_pool_audit=r.get('physical_pool_preparation',r.get('physical_pool_audit'))))
 D=Path('/share/scaglione/nc437/evsp-dr')/B.name;D.mkdir(exist_ok=False);(D/'cases').mkdir();(B/'cases').symlink_to(D/'cases');(B/'logs').mkdir()
 m=dict(schema='evsp-final-chain-gap-v1',prepared_utc=w.now(),cases=cases,selection_audit=audit,storage_root=str(D),tooling_sha256={n:w.sha(B/n) for n in ['worker.py','worker.sub','prepare.py','submit.py']},execution_budgets={'total_s':12600,'fleet_stage_s':10800},policy_sha256=w.sha(H/'SCAGLIONE_RESOURCE_POLICY.md'),interpretation='One independent same-pool MIP for the last unresolved original k16–25 endpoint; no new CG certificate. Frozen worker registry cohort retained; exact campaign root and case ID identify population.')
 w.save(B/'manifest.json',m);checks=[]
 for cid,c in cases.items():
  src=w.preflight(B,m,c);a=w.expand_argv(c['argv'],'/validation/result.json','/validation',src['path'])
  assert a[a.index('--timelimit')+1]=='12600' and a[a.index('--stage1-timelimit')+1]=='10800' and '--cover' in a and '--two-stage' in a
  checks.append(dict(case_id=cid,status='passed',source_status_sha256=src['status_sha256'],source_journal_sha256=src['journal_sha256']))
 w.save(B/'validation.json',dict(status='passed',manifest_sha256=w.sha(B/'manifest.json'),checks=checks,native_runner='Exact previously native-validated worker and871 runner unchanged; every attempt performs large-license check and source pool physical replay.'))
 print(json.dumps({'cases':len(cases),'sha256':w.sha(B/'manifest.json'),'selection':audit}))
if __name__=='__main__':main()
