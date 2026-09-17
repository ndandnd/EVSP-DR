import unittest
from check_predictions import classify_lp,classify_mip

def cg(weight,certificate=True,reason=None):
 return {'certified_rc_optimal':certificate,'stop_reason':reason or ('exact_nonnegative_reduced_cost' if certificate else 'cg_wall_limit'),'terminal_exact_min_reduced_cost':0 if certificate else None,'final':{'route_weight':weight,'objective':100000*weight+123,'artificial_total':0}}
def mip(fleet,proved=False):
 return {'capacity_enforced_in_mip':False,'result':{'fleet':fleet,'has_solution':True,'stage1':{'fleet_proven':proved,'validated_incumbent':True,'incumbent_fleet':fleet,'fleet_integer_lower_bound':fleet if proved else fleet-1}},'duplicate_service_audit':{'all_trips_covered':True},'physical_station_capacity_audit':{'valid':False}}
class Tests(unittest.TestCase):
 def test_certified_wrong_weight_refuted(self):
  self.assertEqual(classify_lp({'predicted_lp_route_weight':[30]}, {'baseline':cg(31)},1e-5)['status'],'REFUTED_CERTIFIED_NUMERIC_PREDICTION')
 def test_uncertified_wrong_weight_unresolved(self):
  self.assertEqual(classify_lp({'predicted_lp_route_weight':[30]}, {'baseline':cg(31,False)},1e-5)['status'],'UNRESOLVED_UNCERTIFIED')
 def test_inconsistent_certificate_unresolved(self):
  self.assertEqual(classify_lp({'predicted_lp_route_weight':[30]}, {'baseline':cg(30,True,'pricing_deadline')},1e-5)['status'],'UNRESOLVED_UNCERTIFIED')
 def test_split_required(self):
  pred={'predicted_lp_route_weight':[31],'predicted_group_weights':{'18E1':12,'18E2':19}}
  self.assertEqual(classify_lp(pred,{'18E1':cg(11),'18E2':cg(20)},1e-5)['status'],'REFUTED_CERTIFIED_NUMERIC_PREDICTION')
  self.assertEqual(classify_lp(pred,{'18E1':cg(12),'18E2':cg(19,False)},1e-5)['status'],'UNRESOLVED_UNCERTIFIED')
  self.assertEqual(classify_lp(pred,{'18E1':cg(12),'18E2':cg(19)},1e-5)['status'],'VERIFIED_NUMERIC_PREDICTION')
 def test_missing_group_pending(self):
  self.assertEqual(classify_lp({'predicted_lp_route_weight':[31]}, {'18E1':cg(12),'18E2':None},1e-5)['status'],'PENDING')
 def test_mip_incumbent_not_proof(self):
  r=classify_mip({'predicted_integer_fleet':'31'},{'all':mip(31)})
  self.assertEqual(r['status'],'UNRESOLVED_INCUMBENT_NOT_PROOF');self.assertFalse(r['finite_pool_proved'])
 def test_mip_proof_finite_pool_only(self):
  r=classify_mip({'predicted_integer_fleet':'31'},{'all':mip(31,True)})
  self.assertEqual(r['status'],'VERIFIED_FINITE_POOL_PREDICTION');self.assertIn('finite saved column pools',r['proof_scope'])
 def test_mip_components_each_need_proof(self):
  self.assertEqual(classify_mip({'predicted_integer_fleet':'31'},{'18E1':mip(12,True),'18E2':mip(19)})['status'],'UNRESOLVED_INCUMBENT_NOT_PROOF')
 def test_invalid_coverage_not_attainment(self):
  d=mip(31,True);d['duplicate_service_audit']['all_trips_covered']=False
  self.assertEqual(classify_mip({'predicted_integer_fleet':'31'},{'all':d})['status'],'UNRESOLVED_PHYSICAL_OR_NO_SOLUTION')
if __name__=='__main__':unittest.main()
