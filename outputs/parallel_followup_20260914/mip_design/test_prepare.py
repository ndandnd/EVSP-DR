import importlib.util,pathlib,unittest
spec=importlib.util.spec_from_file_location('pool_prepare',pathlib.Path(__file__).with_name('prepare.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class UnionTests(unittest.TestCase):
 def test_whole_record_lower_cost_and_source_tie(self):
  a={'trips':[1,2],'cost':10,'witness':'a'};b={'trips':[2,1],'cost':9,'witness':'b'}
  tie={'trips':[1,2],'cost':9,'witness':'tie'};extra={'trips':[3],'cost':4}
  got,n=m.union_records([[a],[b,extra],[tie]],[1,2,3])
  self.assertEqual(n,4);self.assertEqual(got,[b,extra]);self.assertIs(got[0],b)
 def test_invalid_columns_rejected(self):
  for r in [{'trips':[7],'cost':1},{'trips':[1,1],'cost':1},{'trips':[1],'cost':float('nan')}]:
   with self.assertRaises(ValueError):m.union_records([[r]],[1])
 def test_identity_mismatch_rejected(self):
  v={k:1 for k in m.IDENTITY};v.update(master_sense='cover',time_model='event',provenance={k:'bound' for k in m.PROVENANCE})
  m.check_identity([v,v])
  for field in ['soc_step','trip_ids','csv']:
   bad={**v,field:2}
   with self.assertRaises(ValueError):m.check_identity([v,bad])

if __name__=='__main__':unittest.main()
