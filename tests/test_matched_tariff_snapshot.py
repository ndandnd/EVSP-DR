import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/event_uniform_envelope/matched_tariff_pilot.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SnapshotTests(unittest.TestCase):
    def attempt(self,change=None,journal='{"trips":[1],"cost":100000,"found_iter":0}\n'):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); seed=p/'seed.json';seed.write_text('{}')
            status=dict(time_model='event',soc_step=2.5,block_min=5,g_kwh=240.,charge_kw=350.,min_soc_frac=0.,master_sense='partition',column_pool_treatment='GIRO-AUGMENTED',strict_tariff_coverage=True,validated_seed_routes_sha256=m.sha(seed),stop_reason='wall_limit',final=None,final_lp={'artificial_total':0.},columns=1,provenance={'git_commit':'commit','instance_sha256':'instance','prices_sha256':'tariff'})
            if change:status.update(change)
            source=p/'cg.json';source.write_text(json.dumps(status));Path(str(source)+'.columns.jsonl').write_text(journal)
            return m.freeze(source,p/'snapshot.json',dict(charge_kw=350,instance_sha256='instance',tariff_sha256='tariff',cell='cell'),{'commit':'commit'},seed)
    def test_final_null_valid_final_lp_accepted(self):self.assertEqual(len(self.attempt()[0]),64)
    def test_signal_is_censored(self):
        with self.assertRaisesRegex(ValueError,'not terminal'):self.attempt({'stop_reason':'external_signal'})
    def test_nonfinite_artificials_rejected(self):
        with self.assertRaisesRegex(ValueError,'artificials'):self.attempt({'final_lp':{'artificial_total':float('nan')}})
    def test_truncated_journal_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete journal'):self.attempt(journal='{"trips":[1],"cost":100000}')
    def test_pool_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'pool count'):self.attempt({'columns':2})
if __name__=='__main__': unittest.main()
