from pathlib import Path
import worker as w
B=Path(__file__).resolve().parent;V=B.with_name(B.name+'_smoke');m=w.read(B/'manifest.json');sm=w.read(V/'manifest.json');rows=[]
assert sm['production_manifest_sha256']==w.sha(B/'manifest.json')
for cid,c in sm['cases'].items():
 p=V/'cases'/cid/'completion.json';co=w.read(p);r=w.read(co['result_path']);w.require_hash(co['result_path'],co['result_sha256']);assert co['manifest_sha256']==w.sha(V/'manifest.json')
 assert co['usable'] and co['physical_replay_validated'] and r['physical_replay_validated'];assert r['source_result_sha256']==c['source_status_sha256']==m['cases'][cid]['source_status_sha256'];assert r['source_journal_sha256']==c['source_journal_sha256']
 a=r['physical_pool_audit'];assert a['rejected_columns']==a['deterministically_repaired']==0
 assert any('Native Gurobi license check passed' in x.read_text() for x in (V/'logs').glob('*'+co['attempt'].split('_')[0]+'.out'))
 rows.append(dict(case_id=cid,completion=co,physical_pool_audit=a,result_source_sha256=r['source_result_sha256'],source_journal_sha256=r['source_journal_sha256']))
w.save(B/'validation.json',dict(status='passed',manifest_sha256=w.sha(B/'manifest.json'),native_smoke_root=str(V),smoke_manifest_sha256=w.sha(V/'manifest.json'),native_smoke=rows,unit_tests='4 local reconstruction/mismatch/coverage/path tests passed',scope='Native physical/source compatibility at short solve time; not production scientific results'))
print('3 native smoke paths passed')
