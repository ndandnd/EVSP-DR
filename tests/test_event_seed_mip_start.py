import copy, json, sys, tempfile, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_exact_pool_mip import merge_validated_partition_start
class EventSeedMipStartTests(unittest.TestCase):
    def setUp(self):
        self.seed=json.loads((ROOT/'tests/fixtures/event_giro_eligible5_flat_seed.json').read_text())
        self.instance=ROOT/'data/scale_ladder/instances/original_replay_eligible_20260908/Practice_Custom_DutyUnion_original_eligible_k05_20260908.csv'
        self.tariff=ROOT/'data/tariff_response/flat_h26.csv'
        self.status=dict(csv=str(self.instance),prices_csv=str(self.tariff),soc_step=2.5,block_min=5,g_kwh=240.,charge_kw=350.,min_soc_frac=0.,provenance=self.seed['provenance'])
    def merge(self, seed):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'seed.json';p.write_text(json.dumps(seed))
            return merge_validated_partition_start([],sorted(t for r in self.seed['routes'] for t in r['trips']),p,str(self.tariff),self.status,data_dir=ROOT/'data',reference_data_dir=ROOT/'data',preserve_expanded_grid_cost=True)
    def test_real_event_partition_preserves_grid_cost_and_physical_blocks(self):
        routes,start,detail=self.merge(self.seed)
        self.assertEqual(len(start),5)
        self.assertEqual([r['cost'] for r in routes],[r['cost'] for r in self.seed['routes']])
        self.assertTrue(any(r['expanded_grid_cost']>r['continuous_realized_cost'] for r in routes))
        self.assertEqual([r['continuous_realized_charging_blocks'] for r in routes],[r['continuous_realized_charging_blocks'] for r in self.seed['routes']])
    def test_altered_persisted_energy_is_rejected(self):
        damaged=copy.deepcopy(self.seed)
        damaged['routes'][0]['continuous_realized_charging_blocks'][0]['realized_kwh'] += 1
        with self.assertRaises((ValueError,SystemExit)): self.merge(damaged)
    def test_altered_expanded_master_cost_is_rejected(self):
        damaged=copy.deepcopy(self.seed); damaged['routes'][0]['expanded_grid_cost'] += 1
        with self.assertRaises((ValueError,SystemExit)): self.merge(damaged)
if __name__=='__main__':unittest.main()
