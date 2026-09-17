import unittest
from analyze import classify
class ClassificationTests(unittest.TestCase):
 def test_group_mixing_is_not_duty_mixing(self):
  mapping={0:100,1:101,2:200};duties={100:'13401',101:'13402',200:'13316uwt'}
  rows=classify([{'trips':[0,1],'value':.75,'cost':100},{'trips':[1,2],'value':.25,'cost':100}],mapping,duties)
  self.assertFalse(rows[0]['mixed_group']);self.assertTrue(rows[1]['mixed_group']);self.assertEqual(sum(r['lambda_value'] for r in rows if r['mixed_group']),.25)
 def test_mapping_uses_trip_identifiers_not_row_position(self):
  r=classify([{'trips':[9,2],'value':.4,'cost':100}],{9:55,2:66},{55:'13414',66:'13324muw'})[0]
  self.assertEqual(r['groups'],'18E1/18E2');self.assertEqual(r['original_duties'],'13324muw;13414')
 def test_tiny_positive_lambda_is_retained(self):
  r=classify([{'trips':[0,1],'value':1e-12,'cost':100}],{0:1,1:2},{1:'13401',2:'13301'})[0]
  self.assertEqual(r['lambda_value'],1e-12);self.assertTrue(r['mixed_group'])
if __name__=='__main__':unittest.main()
