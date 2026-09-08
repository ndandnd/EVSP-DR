import copy, importlib.util, json, os, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot',ROOT/'scripts/event_uniform_envelope/matched_tariff_pilot.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SplitTariffTests(unittest.TestCase):
    def test_preparer_uses_manifest_tariffs_and_peak_fleet_bounds(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'check_identity'):
            root=Path(tmp)/'pilot';m.prepare(SimpleNamespace(root=root,commit='commit',cg_seconds=14400,mip_seconds=3600,tariffs=['peak12','peak18'],split_stages=True))
            plan=json.loads((root/'plan.json').read_text())
            self.assertEqual(len(plan['cells']),8)
            self.assertEqual({c['trips'] for c in plan['cells']},{56,98,127,62})
            self.assertTrue(all(c['fleet']==c['peak_concurrency'] for c in plan['cells']))
            self.assertEqual(plan['layout'],'split_cg_mip')
            self.assertEqual(plan['stage_resources']['mip']['partition'],'scaglione')
            self.assertEqual(plan['stage_resources']['mip']['exclude'],['scaglione-compute-01','scaglione-cpu-04'])

    def exercise(self,tamper=False):
        with tempfile.TemporaryDirectory() as tmp,patch.object(m,'check_identity'):
            root=Path(tmp)/'pilot';m.prepare(SimpleNamespace(root=root,commit='commit',cg_seconds=14400,mip_seconds=3600,tariffs=['flat'],split_stages=True))
            plan=json.loads((root/'plan.json').read_text());index=3;cell=plan['cells'][index];folder=root/cell['cell']
            fixture=json.loads((ROOT/'tests/fixtures/event_giro_eligible5_flat_seed.json').read_text())
            calls=[]
            def fake_run(argv,**kwargs):
                script=Path(argv[2]).name; calls.append(script);out=Path(argv[argv.index('--out')+1])
                if script=='compare_original_giro_charging.py':
                    value={'summary':[{'matched_physics_comparator_eligible':True,'charging_cost_lower':447.,'terminal_surplus_total_kwh':280.}]}
                elif script=='prepare_event_giro_seed.py':value=fixture
                elif script=='exact_pricer_expanded.py':
                    value=dict(time_model='event',soc_step=2.5,block_min=5,g_kwh=240.,charge_kw=350.,min_soc_frac=0.,master_sense='partition',column_pool_treatment='GIRO-AUGMENTED',strict_tariff_coverage=True,validated_seed_routes_sha256=m.sha(folder/'seed.json'),stop_reason='wall_limit',final=None,final_lp={'artificial_total':0.},columns=5,provenance={'git_commit':'commit','instance_sha256':cell['instance_sha256'],'prices_sha256':cell['tariff_sha256']})
                    Path(str(out)+'.columns.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in fixture['routes']))
                elif script=='run_exact_pool_mip.py':
                    self.assertEqual(argv[argv.index('--timelimit')+1],'3600')
                    self.assertIn('--verified-expanded-initial-partition',argv)
                    value={'buses':5,'fleet_proven':True,'selected_routes':fixture['routes'],'optimal_scope':'full_pool_lexicographic','absolute_cost_gap':0.}
                else:raise AssertionError(script)
                out.write_text(json.dumps(value))
            with patch.dict(os.environ,{'EVSP_PLAN_SHA256':m.sha(root/'plan.json'),'SLURM_RESTART_COUNT':'0'}),patch.object(m.subprocess,'run',side_effect=fake_run),patch.object(m.platform,'platform',return_value='test-platform'):
                m.worker(SimpleNamespace(root=root,index=index,stage='cg'))
                self.assertNotIn('run_exact_pool_mip.py',calls)
                self.assertTrue((folder/'FROZEN.json').is_file())
                if tamper:
                    (folder/'seed.json').write_text('{}')
                    with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):m.worker(SimpleNamespace(root=root,index=index,stage='mip'))
                    self.assertNotIn('run_exact_pool_mip.py',calls)
                else:
                    m.worker(SimpleNamespace(root=root,index=index,stage='mip'))
                    self.assertEqual(calls.count('prepare_event_giro_seed.py'),1)
                    self.assertEqual(calls.count('run_exact_pool_mip.py'),1)
                    self.assertTrue((folder/'COMPLETE.json').is_file())
                    self.assertTrue((folder/'cg.allocation.json').is_file())
                    self.assertTrue((folder/'mip.allocation.json').is_file())
    def test_split_cg_freezes_then_mip_reads_same_verified_seed(self):self.exercise()
    def test_tampered_seed_rejected_between_allocations(self):self.exercise(tamper=True)
if __name__=='__main__':unittest.main()
