import dataclasses,json,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
from sequence_replay import ARMS,coverage_gate,ordered_sequence,replay,restricted_network_class
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'.codex-work/review-strict-chain-20260916').exists())
CODE=ROOT/'.codex-work/review-strict-chain-20260916';sys.path.insert(0,str(CODE/'src'))
from audit_giro_known_columns import build_problem,STATIONS
from event_pricer_network import EventExpandedNetwork,_event_times,normalize_event_station_prices
from utils_v2 import load_station_hourly_prices,base_station_name

class SequenceReplayTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=ROOT/'outputs/independent_review_20260916/execution/p2_strict/smoke/trips2.csv'
  cls.problem=build_problem(p.parent,p.name,reference_data_dir=CODE/'data',max_station_to_trip_wait_min=1560)
  cls.prices=load_station_hourly_prices(CODE/'data/hourly_prices_flat.csv',sorted({base_station_name(s) for s in STATIONS}))
  cls.events=_event_times(cls.problem,normalize_event_station_prices(cls.prices,horizon_min=1560,strict_tariff_coverage=False),5)
 def test_sequence_preserves_order_not_set_or_sorted_ids(self):
  self.assertEqual(ordered_sequence({'trips':[0,1],'route_nodes':['PARX_0',1,'PARX_1',0,'PARX_0']}),(1,0))
  with self.assertRaises(ValueError):ordered_sequence({'trips':[0,1],'route_nodes':['PARX_0',0,'PARX_0']})
 def test_factors_change_exactly_one_dimension(self):
  b=ARMS['baseline']
  for name,arm in ARMS.items():
   if name!='baseline':self.assertEqual(sum(b[k]!=arm[k] for k in b),1,name)
 def test_restricted_dp_matches_full_graph_fixed_sequence(self):
  for name,arm in ARMS.items():
   full=EventExpandedNetwork(self.problem,self.prices,soc_step=2.5,block_min=5,g_kwh=arm['battery_kwh'],charge_kw=240.,reserve_kwh=arm['reserve_kwh'],station_charge_kw={'PARX':arm['parx_kw']},arc_mode='explicit')
   for seq in [(0,),(1,),(0,1)]:
    expected=full.fixed_sequence_record(seq)
    record,route=replay(self.problem,self.prices,self.events,seq,arm,{0:'18E1',1:'18E1'},seconds=10)
    self.assertNotIn(record['status'],['unknown_error','unknown_timeout','unknown_physical_validation_failure'],(name,seq,record))
    self.assertEqual(route is None,expected is None,(name,seq))
    if expected:
     self.assertAlmostEqual(route['cost'],expected['cost'],places=6)
     self.assertEqual(record['event_lattice_sha256'],full.metrics()['event_lattice_sha256'])
     self.assertTrue(record['physical_replay_validated'])
 def test_mixed_groups_excluded_only_by_structural_arm(self):
  rec,route=replay(self.problem,self.prices,self.events,(0,1),ARMS['segregation_only'],{0:'18E1',1:'18E2'})
  self.assertEqual(rec['status'],'structurally_excluded_mixed_groups');self.assertIsNone(route)
 def test_charging_required_and_depot_power_changes_duration(self):
  problem=dataclasses.replace(self.problem,start_min={0:60.,1:130.},end_min={0:70.,1:140.},trip_energy={0:150.,1:150.},
   adjacency={'PARX_0':[(0,0.,0.,'depot_trip')],0:[(1,0.,0.,'trip_trip'),('PARX_1',0.,0.,'trip_station')],
   'PARX_1':[(1,0.,0.,'station_trip'),('PARX_0',0.,0.,'station_depot')],1:[('PARX_0',0.,0.,'trip_depot')]})
  events=_event_times(problem,normalize_event_station_prices(self.prices,horizon_min=1560,strict_tariff_coverage=False),5)
  durations={}
  for name in ['baseline','parx60_only','reserve15_only','battery236p44_only','battery239p01_only']:
   arm=ARMS[name];rec,route=replay(problem,self.prices,events,(0,1),arm,{0:'18E1',1:'18E1'},10)
   self.assertEqual(rec['status'],'feasible',rec)
   charging=route['expanded_grid_charging_stops'];self.assertTrue(charging['kwh'])
   durations[name]=sum(b-a for a,b in zip(charging['cst'],charging['cet']))
   full=EventExpandedNetwork(problem,self.prices,soc_step=2.5,block_min=5,g_kwh=arm['battery_kwh'],charge_kw=240.,reserve_kwh=arm['reserve_kwh'],station_charge_kw={'PARX':arm['parx_kw']},arc_mode='explicit')
   self.assertAlmostEqual(route['cost'],full.fixed_sequence_record((0,1))['cost'])
  self.assertAlmostEqual(durations['parx60_only'],4*durations['baseline'])
 def test_infeasible_is_scoped_to_fixed_sequence_event_graph(self):
  impossible=dataclasses.replace(self.problem,trip_energy={0:300.,1:300.})
  rec,route=replay(impossible,self.prices,self.events,(0,),ARMS['baseline'],{0:'18E1'})
  self.assertEqual(rec['status'],'infeasible_in_fixed_sequence_event_model');self.assertFalse(rec['full_model_pricing_certificate'])
 def test_timeout_is_unknown_never_infeasible(self):
  class Slow:
   def __init__(self,*a,**k):time.sleep(.1)
  with patch('sequence_replay.restricted_network_class',return_value=Slow):
   rec,route=replay(self.problem,self.prices,self.events,(0,),ARMS['baseline'],{0:'18E1'},seconds=.005)
  self.assertEqual(rec['status'],'unknown_timeout');self.assertIsNone(route)
 def test_exception_is_unknown_never_infeasible(self):
  with patch('sequence_replay.restricted_network_class',side_effect=RuntimeError('test')):
   rec,route=replay(self.problem,self.prices,self.events,(0,),ARMS['baseline'],{0:'18E1'})
  self.assertEqual(rec['status'],'unknown_error')
 def test_missing_coverage_is_not_hidden_by_failed_replay(self):
  rows=[{'status':'feasible','trip_sequence':[0]},{'status':'unknown_timeout','trip_sequence':[1]}]
  self.assertEqual(coverage_gate(rows,[0,1])['missing_trip_ids'],[1])
if __name__=='__main__':unittest.main()
