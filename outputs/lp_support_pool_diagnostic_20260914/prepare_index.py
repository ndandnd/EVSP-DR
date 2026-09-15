from pathlib import Path
import json
import worker as w
B=Path(__file__).resolve().parent;H=B.parent
RECIPIENTS='c1_k08_integer c1_k08_lpweight c1_k10_integer c2_k08_lpweight c3_k08_integer c3_k08_lpweight c4_k08_integer c4_k08_lpweight c4_k10_integer c4_k10_lpweight c5_k08_lpweight c5_k10_lpweight c6_k10_lpweight'.split()
def bind(root,cid):
 p=(root/'cases'/cid/'cg.json').resolve();v=w.read(p);j=w.journal_path(p,v);mp=(root/'cases'/(cid+'_mip')/'mip_result.json').resolve();m=w.read(mp)
 assert m['source_result_sha256']==w.sha(p) and m['source_journal_sha256']==w.sha(j)
 assert v['certified_rc_optimal'] and v['final_lp']['artificial_total']==0 and m['physical_replay_validated']
 a=m['physical_pool_audit'];assert a['rejected_columns']==a['deterministically_repaired']==0
 return dict(case_id=cid,status_path=str(p),status_sha256=w.sha(p),journal_path=str(j),journal_sha256=w.sha(j),mip_evidence_path=str(mp),mip_evidence_sha256=w.sha(mp),input_sha256=v['provenance']['instance_sha256'],buses=m['buses'],fleet_proven=m['fleet_proven'],source_cg_wall_s=v['wall_s'],source_mip_runtime_s=m['runtime_s'],final_lp_objective=v['final_lp']['objective'])
def main():
 assert not (B/'source_index.json').exists();D=Path('/share/scaglione/nc437/evsp-dr')/B.name;D.mkdir(exist_ok=False);(D/'pools').mkdir();(D/'cases').mkdir();(B/'pools').symlink_to(D/'pools');(B/'cases').symlink_to(D/'cases');(B/'logs').mkdir()
 groups={}
 for cid in RECIPIENTS:
  pair=cid.rsplit('_',1)[0];k=int(pair.split('_k')[1]);r=bind(H/'overnight_parallel_20260914',cid);assert r['fleet_proven'] and r['buses']>k
  if pair not in groups:
   donor=bind(H/'compact_seed_support_20260914',pair+'_core');assert donor['buses']==k
   groups[pair]=dict(donor=donor,recipients=[],target_k=k)
  assert groups[pair]['donor']['input_sha256']==r['input_sha256'];groups[pair]['recipients'].append(r)
 source=dict(schema='evsp-lp-support-transplant-source-v1',created_utc=w.now(),groups=groups,tooling_sha256={n:w.sha(B/n) for n in ['worker.py','native_union.py','prepare_index.py','construct.py']},selection='13 proved-above-target difficult pools;9 predetermined matching CORE donors. Not a random sample.',selector='FinalLPvalue>0; canonical native cheapest-tripset winner. Novel positive additions counted by incidence; equally many novel LP-zero incidences ordered by SHA256 of canonical sorted trip-ID JSON, then key. No MIP selected routes used.',storage_root=str(D))
 w.save(B/'source_index.json',source);print(w.sha(B/'source_index.json'))
if __name__=='__main__':main()
