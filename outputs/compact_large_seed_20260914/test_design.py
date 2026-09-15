import unittest
from seed_logic import choose
class Tests(unittest.TestCase):
 def source(self):
  integer=[{'trips':[i],'cost':10.} for i in range(3)]
  lp=[{'trips':r,'cost':15.,'value':.5} for r in [[0,1],[1,2],[0,2]]]
  return {'trip_ids':[0,1,2],'final_lp':{'artificial_total':0,'positive_routes':lp}}, {'physical_replay_validated':True,'selected_routes':integer,'buses':3},integer+lp+[{'trips':[0,1,2],'cost':20.}]
 def test_core_over_cap(self):
  p,m,j=self.source();arms,a=choose(p,m,j,1);self.assertEqual(len(arms['core']),6);self.assertEqual(arms['core'],arms['core512']);self.assertTrue(a['core_exceeds_cap'])
 def test_fill_and_determinism(self):
  p,m,j=self.source();arms,a=choose(p,m,j,7);self.assertEqual(len(arms['core512']),7);self.assertEqual(arms['core512'][-1]['trips'],[0,1,2]);self.assertEqual(choose(p,m,list(reversed(j)),7), (arms,a))
 def test_conflicting_native_core_rejected(self):
  p,m,j=self.source();j.append({'trips':[1,0],'cost':14.})
  with self.assertRaises(AssertionError):choose(p,m,j)
 def test_bad_lp_rejected(self):
  p,m,j=self.source();p['final_lp']['positive_routes'][0]['value']=float('nan')
  with self.assertRaises(AssertionError):choose(p,m,j)
if __name__=='__main__':unittest.main()
