import random,unittest
from time_only_vsp import maximum_matching,graph,overlap,minutes,travel_data

def brute_matching(adj):
 def rec(i,used):
  if i==len(adj):return 0
  return max([rec(i+1,used)]+[1+rec(i+1,used|{v}) for v in adj[i] if v not in used])
 return rec(0,set())
class MatchingTests(unittest.TestCase):
 def test_random_dags_against_exhaustive_matching(self):
  rng=random.Random(1617)
  for n in range(1,7):
   for trial in range(15):
    adj=[[j for j in range(i+1,n) if rng.random()<.4] for i in range(n)];r=maximum_matching(adj);self.assertEqual(n-r['minimum_path_cover'],brute_matching(adj))
 def test_detour_can_repair_missing_direct_transition(self):
  trips=[dict(start=0,end=10,start_ref=0,end_ref=0),dict(start=15,end=25,start_ref=2,end_ref=2)];direct=[[0,2,999],[2,0,2],[999,2,0]];closure=[[0,2,4],[2,0,2],[4,2,0]]
  self.assertEqual(maximum_matching(graph(trips,direct,1))['minimum_path_cover'],2);self.assertEqual(maximum_matching(graph(trips,closure,1))['minimum_path_cover'],1)
 def test_halfopen_and_extended_hours(self):
  self.assertEqual(overlap([dict(start=0,end=10),dict(start=10,end=20)]),1);self.assertEqual(minutes('25:10'),1510)
 def test_halfminute_not_rounded_to_exclude_edge(self):
  trips=[dict(start=0,end=10,start_ref=0,end_ref=0),dict(start=18,end=20,start_ref=1,end_ref=1)]
  self.assertEqual(graph(trips,[[0,15],[15,0]],2),[[1],[]])
 def test_production_aliases(self):
  refs,scale,direct,closure,resolve=travel_data();self.assertEqual(resolve('PARX_0'),resolve('PARX_1'));self.assertEqual(resolve('2190L_0'),resolve('2190'))
if __name__=='__main__':unittest.main()
