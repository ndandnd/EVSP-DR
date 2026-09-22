import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cached_worker', HERE / 'cached_cg_worker.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
PLAN = w.read(HERE / 'manifest.json')
RECEIPTS = HERE.parent / 'operations/strict_graph'


class RecoveryGateTests(unittest.TestCase):
    def copy_receipts(self, root):
        for relative in PLAN['graph_receipt_sha256']:
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RECEIPTS / relative, dest)

    def reseal(self, root, plan):
        cold = root / 'artifacts/cold.json'
        reload = root / 'attempts/772820_r0/reload.json'
        result_path = root / 'attempts/772820_r0/result.json'
        result = w.read(result_path)
        result.update(cold_sha256=w.sha(cold), reload_sha256=w.sha(reload))
        result_path.write_text(json.dumps(result))
        complete_path = root / 'attempts/772820_r0/COMPLETE.json'
        complete = w.read(complete_path); complete['result_sha256'] = w.sha(result_path)
        complete_path.write_text(json.dumps(complete))
        seal_path = root / 'artifacts/COLD_COMPLETE.json'
        seal = w.read(seal_path); seal['cold_sha256'] = w.sha(cold)
        seal_path.write_text(json.dumps(seal))
        plan['graph_receipt_sha256'] = {r: w.sha(root / r) for r in plan['graph_receipt_sha256']}

    def test_authentic_completed_native_gate(self):
        self.assertEqual(w.graph_gate(RECEIPTS, PLAN)['initial_pool_columns'], 8397)

    def test_corrupt_or_missing_proof_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.copy_receipts(root)
            file = root / 'artifacts/cold.json'; file.write_text(file.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'receipt hash'):
                w.graph_gate(root, PLAN)
            file.unlink()
            with self.assertRaises(FileNotFoundError):
                w.graph_gate(root, PLAN)

    def test_resealed_route_or_initializer_drift_rejects(self):
        for field in ('query', 'initializer'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); self.copy_receipts(root); plan = copy.deepcopy(PLAN)
                file = root / 'attempts/772820_r0/reload.json'; record = w.read(file)
                if field == 'query': record['queries'][0]['route']['cost'] += 1
                else: record['initial_pool']['singleton_checks'][0]['record_sha256'] = 'bad'
                file.write_text(json.dumps(record)); self.reseal(root, plan)
                with self.assertRaisesRegex(ValueError, 'parity mismatch'):
                    w.graph_gate(root, plan)

    def test_command_preserves_science_and_true_parent(self):
        cmd = w.command(PLAN, Path('/unique_attempt'))
        old = PLAN['original_command']
        for flag in ('--mode','--arm','--battery-kwh','--reserve-kwh','--non-parx-kw','--soc-step','--block-min','--max-station-wait-min','--threads','--arc-mode','--cg-wall-s','--max-iters'):
            self.assertEqual(cmd[cmd.index(flag)+1], old[old.index(flag)+1], flag)
        self.assertNotIn('--resume', cmd)
        self.assertEqual(cmd[cmd.index('--inherit-compatible-commit')+1], w.PARENT)
        self.assertEqual(cmd[cmd.index('--expected-commit')+1], w.MODEL)
        self.assertEqual(cmd[cmd.index('--out')+1], '/unique_attempt/result.json')
        with self.assertRaises(ValueError):
            w.command({**PLAN, 'original_command': old + ['--resume']}, Path('/bad'))

    def test_requeue_and_duplicate_fresh_job_fail_before_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'manifest.json').write_text('{}')
            with patch.dict('os.environ', {'SLURM_JOB_ID':'1','SLURM_RESTART_COUNT':'1'}), patch.object(w, 'graph_gate') as gate:
                with self.assertRaisesRegex(ValueError, 'requeue'):
                    w.worker(root, PLAN)
                gate.assert_not_called()
            (root / 'STARTED.json').write_text('{}')
            with patch.dict('os.environ', {'SLURM_JOB_ID':'2','SLURM_RESTART_COUNT':'0'}), patch.object(w, 'graph_gate') as gate:
                with self.assertRaises(FileExistsError):
                    w.worker(root, PLAN)
                gate.assert_not_called()

    def test_native_license_failure_does_not_start_cg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); graph = root / 'graph'; graph.mkdir()
            gm = {'model_source_sha256':{}, 'input_files':{}, 'native_lineage_sha256':{}, 'native_lineage_attempt':str(graph)}
            (graph / 'manifest.json').write_text(json.dumps(gm)); (root / 'manifest.json').write_text('{}')
            (graph / 'artifacts').mkdir(); cache=graph / 'artifacts/k19_fedf4214.graph.cache'; cache.write_bytes(b'cache')
            (graph / 'input_artifacts').mkdir(); (graph / 'input_artifacts/k19_command.json').write_text(json.dumps(PLAN['original_command']))
            plan={**PLAN, 'graph_root':str(graph), 'graph_manifest_sha256':w.sha(graph / 'manifest.json'), 'cache_bytes':5, 'cache_sha256':w.sha(cache)}
            with patch.dict('os.environ', {'SLURM_JOB_ID':'3','SLURM_RESTART_COUNT':'0'}), patch.object(w,'graph_gate'), patch.object(w.subprocess,'check_output',side_effect=[w.MODEL+'\n','']), patch.object(w.subprocess,'run') as run:
                run.return_value.returncode=1
                with self.assertRaisesRegex(ValueError, 'license preflight failed'):
                    w.worker(root,plan)
                self.assertEqual(run.call_count,1)
                self.assertTrue(run.call_args.args[0][1].endswith('/src/gurobi_preflight.py'))
                self.assertFalse((root / 'attempts/3_r0/CG_STARTED.json').exists())
                self.assertFalse(w.read(root / 'attempts/3_r0/FAILED.json')['cg_started'])
                self.assertTrue((root / 'attempts/3_r0/command.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
