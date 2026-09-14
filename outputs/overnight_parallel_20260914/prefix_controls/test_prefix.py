import tempfile,json,unittest
from pathlib import Path
from prefix_logic import extract,sha
class Prefix(unittest.TestCase):
 def test_before_insertion_and_replacement(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);rows=[{'trips':[1],'cost':10,'found_iter':0},{'trips':[2],'cost':10,'found_iter':0},{'trips':[1,2],'cost':8,'found_iter':1},{'trips':[2,1],'cost':7,'found_iter':2},{'trips':[1,2],'cost':6,'found_iter':3}];src=p/'source';src.write_text(''.join(json.dumps(x)+'\n' for x in rows));a=extract(src,p/'out',3,{1,2});self.assertEqual(a['selected_records'],4);self.assertEqual(a['unique_pool_columns'],3);self.assertEqual(a['source_journal_sha256'],sha(src));self.assertEqual([json.loads(x) for x in (p/'out').read_text().splitlines()],rows[:4])
 def test_nonmonotonic_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);src=p/'source';src.write_text(''.join(json.dumps({'trips':[1],'cost':1,'found_iter':i})+'\n' for i in [0,2,1]));
   with self.assertRaises(ValueError):extract(src,p/'out',2,{1})
if __name__=='__main__':unittest.main()
