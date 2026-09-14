import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import gate
class GateTests(unittest.TestCase):
 def test_timeout_only(self):
  self.assertTrue(gate.eligible({'state':'TIMEOUT','exit_code':'0:15'},{}))
  e={'status':'time_limit','watchdog_time_limit':True}
  self.assertTrue(gate.eligible({'state':'FAILED','exit_code':'124:0'},e))
  for state,exitcode in [('FAILED','1:0'),('CANCELLED','0:15'),('OUT_OF_MEMORY','0:9')]:self.assertFalse(gate.eligible({'state':state,'exit_code':exitcode},e))
  for state in ['RUNNING','PENDING','REQUEUED']:
   with self.assertRaises(ValueError):gate.eligible({'state':state,'exit_code':'0:0'},e)
 def test_dependency_previous_k_preserved(self):
  self.assertEqual(gate.replacement_dependencies('afterok:187967_0(unfulfilled),afterok:133927(unfulfilled)','187967_0','200000'),'afterok:200000,afterok:133927')
  self.assertEqual(gate.replacement_dependencies('afterok:187967_1(unfulfilled),afterok:187968(unfulfilled)','187967_1','200001'),'afterok:200001,afterok:187968')
  with self.assertRaises(ValueError):gate.replacement_dependencies('afterok:99','187967_0','200000')
 def test_hash_identity_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'network.pkl';p.write_bytes(b'fixture');meta=Path(str(p)+'.manifest.json');digest=hashlib.sha256(p.read_bytes()).hexdigest();meta.write_text(json.dumps({'identity':{'instance_sha256':'a'},'pickle_sha256':digest}));c={'cache_identity':{'instance_sha256':'a'}}
   gate.validate_cache(p,meta,c)
   with self.assertRaises(ValueError):gate.validate_cache(p,meta,{'cache_identity':{'instance_sha256':'b'}})
   p.write_bytes(b'tampered')
   with self.assertRaises(ValueError):gate.validate_cache(p,meta,c)
   with self.assertRaises(ValueError):gate.link_absent(p,meta)
 def test_valid_original_cache_no_solver(self):
  with tempfile.TemporaryDirectory() as t:
   b=Path(t)/'recovery';b.mkdir();old=Path(t)/'original';old.mkdir();cache=old/'network.pkl';cache.write_bytes(b'complete-cache');digest=gate.w.sha(cache)
   gate.w.save(old/'network.pkl.manifest.json',{'identity':{'instance_sha256':'a'},'pickle_sha256':digest});gate.w.save(old/'cache_result.json',{'cache_sha256':digest})
   parent=Path(t)/'original_manifest.json';parent.write_text('{}')
   m={'cases':{'case':{'original_case_dir':str(old),'original_graph_job':'1_0','cache_identity':{'instance_sha256':'a'},'static_hashes':{}}},'tooling_sha256':{},'original_manifest_path':str(parent),'original_manifest_sha256':gate.w.sha(parent),'source_code':'fixture','execution_commit':'fixture'};gate.w.save(b/'manifest.json',m)
   with patch.object(gate,'scheduler',return_value={'state':'COMPLETED'}),patch.object(gate.w,'check_code'),patch.object(gate.w,'run_process') as solver:gate.run(b,'case');solver.assert_not_called()
   result=gate.w.read(b/'cases/case/completion.json');self.assertFalse(result['graph_rebuilt']);self.assertEqual(result['source_construction_kind'],'original')
if __name__=='__main__':unittest.main()
