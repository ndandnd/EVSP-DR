"""Tiny local end-to-end chunk -> coverage -> checkpoint import regression test."""
import json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from sequence_replay import sha,canonical
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'.codex-work/review-strict-chain-20260916').exists())
CODE=ROOT/'.codex-work/review-strict-chain-20260916'
class WorkflowTest(unittest.TestCase):
 def test_tiny_replay_assembly_and_pinned_cg_import(self):
  with tempfile.TemporaryDirectory() as raw:
   root=Path(raw);(root/'inputs').mkdir();(root/'extracted').mkdir();(root/'chunks').mkdir()
   shutil.copyfile(ROOT/'outputs/independent_review_20260916/execution/p2_strict/smoke/trips2.csv',root/'inputs/w5_k31.csv')
   (root/'inputs/groups.json').write_text(json.dumps({'0':{'group':'18E1'},'1':{'group':'18E1'}}))
   m=json.loads((HERE/'manifest.json').read_text());m.update(instance_sha256=sha(root/'inputs/w5_k31.csv'),group_map_sha256=sha(root/'inputs/groups.json'),trip_count=2,source_ordered_pool_sha256='synthetic_test')
   (root/'manifest.json').write_text(json.dumps(m))
   seq=root/'extracted/sequences.jsonl'
   # Leave singleton 1 out to exercise the explicitly recorded fallback.
   seq.write_text(json.dumps({'sequence_index':0,'sequence_sha256':canonical([0]),'trip_sequence':[0],'source_pool_indices':[0],'source_route_sha256':['synthetic']})+'\n')
   (seq.parent/'extraction.json').write_text(json.dumps({'sequences_sha256':sha(seq),'source_ordered_pool_sha256':'synthetic_test'}))
   def run(script,*args):
    subprocess.run([sys.executable,str(HERE/script),'--root',str(root),'--code',str(CODE),*map(str,args)],check=True,capture_output=True,text=True,timeout=60)
   run('replay_chunk.py','--sequences',seq,'--arm','baseline','--out',root/'chunks/000')
   run('assemble.py','--sequences',seq,'--arm','baseline','--chunks',root/'chunks','--out',root/'assembled')
   report=json.loads((root/'assembled/assembly.json').read_text())
   self.assertEqual(report['coverage_before_singletons']['missing_trip_ids'],[1])
   self.assertEqual(len(report['singleton_fallbacks']),1);self.assertTrue(report['cg_seed_ready'])
   run('continue_cg.py','--seed',root/'assembled','--out',root/'prepared')
   imported=json.loads((root/'prepared/seed_import.json').read_text())
   self.assertTrue(imported['every_route_replayed']);self.assertFalse(imported['cg_executed'])
   self.assertFalse((root/'prepared/cg.json').exists());self.assertEqual(imported['seed_routes'],2)
   extraction=json.loads((seq.parent/'extraction.json').read_text());extraction['scope']='pilot_subset_not_full_pool'
   (seq.parent/'extraction.json').write_text(json.dumps(extraction))
   rejected=subprocess.run([sys.executable,str(HERE/'assemble.py'),'--root',str(root),'--code',str(CODE),'--sequences',str(seq),'--arm','baseline','--chunks',str(root/'chunks'),'--out',str(root/'should_not_exist')],capture_output=True,text=True,timeout=30)
   self.assertNotEqual(rejected.returncode,0);self.assertIn('pilot sequences cannot',rejected.stderr)
if __name__=='__main__':unittest.main()
