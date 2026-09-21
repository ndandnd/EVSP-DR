"""Regression gates for strict no-capacity graph representation and inheritance."""
import json
import random
import unittest
from types import SimpleNamespace
from unittest import mock
from test_event_pricer_network import two_trip_problem, prices, STATION
from test_inherit_capacity_pool import InheritanceTests
from run_capacity_speed_event_cg import build_network
from event_pricer_network import EventExpandedNetwork

BASE='50ceb6c095a580f79f87b53bef536cac31f81963'

class StrictPackedTests(unittest.TestCase):
    def test_shared_capacity_rejects_packed_before_construction(self):
        for arm in ('capacity','combined'):
            with self.subTest(arm=arm), mock.patch('run_capacity_speed_event_cg.EventExpandedNetwork') as ctor:
                with self.assertRaisesRegex(ValueError,'shared capacity'):
                    build_network(SimpleNamespace(arm=arm,arc_mode='lazy'),None,None)
                ctor.assert_not_called()

    def test_strict_soc_and_parx60_oracles_agree(self):
        problem=two_trip_problem(first_energy=160.0)
        kw=dict(soc_step=2.5,block_min=5,g_kwh=239.01,charge_kw=240.0,
                reserve_kwh=35.8515,station_charge_kw={STATION:60.0})
        explicit=EventExpandedNetwork(problem,prices(),arc_mode='explicit',**kw)
        packed=EventExpandedNetwork(problem,prices(),arc_mode='lazy',**kw)
        self.assertEqual(explicit.metrics()['event_lattice_sha256'],packed.metrics()['event_lattice_sha256'])
        rng=random.Random(20260921)
        for i in range(20):
            duals={0:rng.uniform(0,130000),1:rng.uniform(0,130000)}
            for objective in ('combined-cost','artificial-elimination','fleet-only','charging-cost'):
                with self.subTest(i=i,objective=objective):
                    a=explicit.min_reduced_cost_route(duals,objective=objective)
                    b=packed.min_reduced_cost_route(duals,objective=objective)
                    self.assertAlmostEqual(a['rc'],b['rc'],places=7)
                    self.assertEqual(a['trips'],b['trips'])
                    self.assertEqual(a['_event_record'],b['_event_record'])
        self.assertEqual(explicit.fixed_sequence_record((0,1)),packed.fixed_sequence_record((0,1)))

class CompatibleInheritanceTests(InheritanceTests):
    def test_different_execution_commit_requires_exact_audited_gate(self):
        self.doc['provenance']['git_commit']=BASE
        self.doc['physics']['capacity_enforced']=False
        self.status.write_text(json.dumps(self.doc))
        kw=dict(expected_physics={'battery_kwh':236.44,'capacity_enforced':False})
        with self.assertRaisesRegex(ValueError,'git_commit'): self.load(**kw)
        routes,meta=self.load(**kw,compatible_parent_commit=BASE)
        self.assertEqual(meta['audited_compatible_parent_commit'],BASE)
        self.assertEqual(meta['parent_execution_commit'],BASE)
        self.assertTrue(meta['every_inherited_route_replayed'])
        with self.assertRaisesRegex(ValueError,'physical replay'):
            self.load(**kw,compatible_parent_commit=BASE,route_validator=lambda r:'lowSOC')
        with self.assertRaisesRegex(ValueError,'physics'):
            self.load(expected_physics={'battery_kwh':240,'capacity_enforced':False},compatible_parent_commit=BASE)
        with self.assertRaisesRegex(ValueError,'without shared capacity'):
            self.load(expected_physics={'battery_kwh':236.44,'capacity_enforced':True},compatible_parent_commit=BASE)
        with self.assertRaisesRegex(ValueError,'git_commit|unaudited'):
            self.load(**kw,compatible_parent_commit='some_other_commit')

if __name__=='__main__': unittest.main()
