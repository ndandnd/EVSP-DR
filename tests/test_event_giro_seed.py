import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prepare_event_giro_seed import SCHEMA, validate_event_seed
from exact_pricer_expanded import validated_fixed_duty_seed_records

class EventSeedTests(unittest.TestCase):
    def setUp(self):
        self.route = dict(trips=[1], route_nodes=['PARX_0',1,'PARX_0'],
            charging_stops={}, expanded_grid_charging_stops={}, expanded_grid_cost=100000.,
            continuous_realized_cost=100000., continuous_realized_charging_blocks=[],
            cost=100000., master_cost_semantics='expanded_grid_cost', cost_tariff_sha256='tariff')
        self.net = SimpleNamespace(g=240.,charge_kw=240.,reserve=0.,soc_step=2.5,block_min=5,
            metrics=lambda: {'event_lattice_sha256':'same-lattice'},
            fixed_sequence_record=lambda trips: copy.deepcopy(self.route))
        self.payload = dict(schema=SCHEMA,instance_sha256='actual-instance',tariff={'sha256':'tariff'},
            physics=dict(g_kwh=240.,charge_kw=240.,reserve_kwh=0.,soc_step=2.5,block_min=5),
            event_lattice_sha256='same-lattice',continuous_cost_pricing_certified=False,
            routes=[copy.deepcopy(self.route)])
    def validate(self, **kwargs):
        return validate_event_seed(self.payload,self.net,problem=SimpleNamespace(trips=[1]),
            tariff_sha256='tariff',instance_sha256=kwargs.get('instance_sha256','actual-instance'))
    def test_accepts_recomputed_partition(self):
        self.assertEqual(self.validate()[0]['origin'],'validated_event_giro_seed')
    def test_wrong_instance_rejected_even_same_lattice_and_route(self):
        with self.assertRaisesRegex(ValueError,'identity'):
            self.validate(instance_sha256='different-current-instance')
    def test_changed_charge_witness_rejected(self):
        self.payload['routes'][0]['expanded_grid_cost'] += 1
        with self.assertRaisesRegex(ValueError,'differs'): self.validate()
    def test_duplicate_partition_rejected(self):
        self.payload['routes'] *= 2
        with self.assertRaisesRegex(ValueError,'exact trip partition'): self.validate()
    def test_current_network_required(self):
        self.net=None
        with self.assertRaisesRegex(ValueError,'current event network'): self.validate()
    def test_other_tariff_rejected(self):
        self.payload['tariff']['sha256']='different'
        with self.assertRaisesRegex(ValueError,'identity'): self.validate()

if __name__=='__main__': unittest.main()
