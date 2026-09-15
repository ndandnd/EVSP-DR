import copy,json,sqlite3,unittest
import construct as c
class Tests(unittest.TestCase):
 def fixture(self):
  db=sqlite3.connect(':memory:');db.execute('CREATE TABLE cols(tripkey TEXT,cost REAL,payload TEXT)');r={'trips':[0,1],'cost':100003.,'full_route_payload':{'preserve':True}};db.execute('INSERT INTO cols VALUES(?,?,?)',(c.key(r['trips']),r['cost'],json.dumps(r)))
  v={'trip_ids':[0,1],'certified_rc_optimal':True,'final_lp':{'artificial_total':0,'positive_routes':[{'trips':[0,1],'cost':100003.,'value':1.}],'objective':100003.,'route_weight':1.}}
  return db,v
 def test_reconstruction(self):
  d,v=self.fixture();weights,a=c.positive(v,d);self.assertEqual(a['minimum_trip_coverage'],1);self.assertEqual(a['objective_difference'],0);self.assertEqual(len(weights),1)
 def test_bad_objective(self):
  d,v=self.fixture();v['final_lp']['objective']+=1
  with self.assertRaises(AssertionError):c.positive(v,d)
 def test_uncovered_weight(self):
  d,v=self.fixture();v['final_lp']['positive_routes'][0]['value']=.5
  with self.assertRaises(AssertionError):c.positive(v,d)
 def test_ordered_path_mismatch(self):
  d,v=self.fixture();v['final_lp']['positive_routes'][0]['trips']=[1,0]
  with self.assertRaises(AssertionError):c.positive(v,d)
if __name__=='__main__':unittest.main()
