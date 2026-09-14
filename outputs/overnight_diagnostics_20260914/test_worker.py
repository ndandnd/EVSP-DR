"""Contract tests with real temporary Git checkouts and fake solver processes."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('diagnostic_worker', HERE / 'worker.py')
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)

FAKE_SOLVER = '''import argparse,json,pathlib,sys,os
p=argparse.ArgumentParser()
p.add_argument('--out',required=True)
p.add_argument('--commit',required=True)
p.add_argument('--input-hash',required=True)
p.add_argument('--exit-code',default='0')
p.add_argument('--artificials',default='0')
p.add_argument('--resume',action='store_true')
p.add_argument('--mip',action='store_true')
p.add_argument('--physical',action='store_true')
a=p.parse_args()
out=pathlib.Path(a.out)
journal=pathlib.Path(str(out)+'.columns.jsonl')
journal.write_text('{"trips":[1],"cost":100000}\\n')
value={'columns_journal':str(journal),'final':{'iter':1,'artificials':int(a.artificials)},
 'provenance':{'git_commit':a.commit,'instance_sha256':a.input_hash},
 'certified_rc_optimal':False,'stop_reason':'wall_limit','wall_s':0.01}
if a.mip:
 assert os.environ['EVSP_EXPECTED_COMMIT']==a.commit
 assert os.environ['EVSP_REQUIRE_DETACHED']=='1'
 assert len(os.environ['EVSP_MIP_EXPECTED_RESULT_SHA256'])==64
 assert len(os.environ['EVSP_MIP_EXPECTED_JOURNAL_SHA256'])==64
 value={'physical_replay_validated':a.physical,'buses':1,'fleet_bound':1,'fleet_proven':True}
out.write_text(json.dumps(value))
sys.exit(int(a.exit_code))
'''


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'campaign'
        self.root.mkdir()
        self.code = self.base / 'code'
        self.code.mkdir()
        (self.code / 'fake_solver.py').write_text(FAKE_SOLVER)
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')
        self.git('add', 'fake_solver.py')
        self.git('commit', '-qm', 'test fixture')
        self.commit = self.git('rev-parse', 'HEAD')
        self.git('checkout', '--detach', '-q')
        self.input = self.base / 'input.csv'
        self.input.write_text('trip_id\n1\n')
        self.static = self.base / 'prices.csv'
        self.static.write_text('hour,price\n0,1\n')
        for name in ('worker.py', 'worker.sub'):
            shutil.copy2(HERE / name, self.root / name)
        self.case = dict(kind='cg', chain=1, target_k=1, treatment='fixture',
            source_code=str(self.code), execution_commit=self.commit,
            data_dir=str(self.base), input_path=str(self.input), input_sha256=worker.sha(self.input),
            static_hashes={str(self.static):worker.sha(self.static)},
            argv=[sys.executable, str(self.code / 'fake_solver.py'), '--out', '{out}',
                  '--commit', self.commit, '--input-hash', worker.sha(self.input)],
            solver_budget_s=1, watchdog_s=5)
        self.manifest = dict(cases={'case':self.case},
            tooling_sha256={name:worker.sha(self.root/name) for name in ('worker.py','worker.sub')})
        registry = self.base / 'mip_preemption_study_20260911' / 'registry.json'
        worker.save(registry, {'cases':[]})
        self.env = patch.dict(os.environ, {'SLURM_JOB_ID':'12345','SLURM_RESTART_COUNT':'0'})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git','-C',str(self.code),*args], text=True).strip()

    def run_case(self):
        worker.save(self.root / 'manifest.json', self.manifest)
        return worker.run_case(self.root, 'case')

    def test_placeholder_is_literal_and_unresolved_is_rejected(self):
        self.assertEqual(worker.expand_argv(['{out}', '{attempt_dir}/log', '{source_status}'],
            '/tmp/out with spaces', '/tmp/attempt', '/tmp/source'),
            ['/tmp/out with spaces','/tmp/attempt/log','/tmp/source'])
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            worker.expand_argv(['{source_status}'], '/out', '/attempt')
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            worker.expand_argv(['{invented}'], '/out', '/attempt')

    def test_changed_frozen_input_rejected_before_launch(self):
        self.input.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.run_case()
        self.assertFalse((self.root/'cases/case/completion.json').exists())
        self.assertFalse((self.root/'cases/case/attempts/12345_r0/execution.json').exists())

    def test_attached_or_dirty_execution_code_is_rejected(self):
        self.git('switch', '-qc', 'attached')
        with self.assertRaisesRegex(ValueError, 'detached'):
            worker.preflight(self.root, self.manifest, self.case)
        self.git('checkout','--detach','-q')
        (self.code/'fake_solver.py').write_text(FAKE_SOLVER+'\n# modification\n')
        with self.assertRaisesRegex(ValueError, 'tracked modifications'):
            worker.preflight(self.root, self.manifest, self.case)

    def test_failure_with_plausible_result_does_not_publish(self):
        self.case['argv'] += ['--exit-code', '3']
        with self.assertRaisesRegex(RuntimeError, 'return code 3'):
            self.run_case()
        attempt = self.root/'cases/case/attempts/12345_r0'
        self.assertTrue((attempt/'cg.json').exists())
        self.assertEqual(worker.read(attempt/'execution.json')['status'], 'failed')
        self.assertFalse((self.root/'cases/case/cg.json').exists())
        self.assertFalse((self.root/'cases/case/completion.json').exists())

    def test_zero_exit_artificials_do_not_publish(self):
        self.case['argv'] += ['--artificials', '1']
        with self.assertRaisesRegex(ValueError, 'artificial-free'):
            self.run_case()
        self.assertFalse((self.root/'cases/case/completion.json').exists())

    def test_time_limited_usable_cg_publishes_hash_bound_source(self):
        result = self.run_case()
        self.assertTrue(result['usable'])
        self.assertFalse(result['certified'])
        self.assertTrue((self.root/'cases/case/cg.json').is_symlink())
        self.assertEqual(result['result_sha256'], worker.sha(result['result_path']))
        child = dict(self.case, kind='mip', source_case='case', source_cg_commit=self.commit)
        source = worker.preflight(self.root, self.manifest, child)
        self.assertEqual(source['status_sha256'], result['result_sha256'])
        Path(source['journal_path']).write_text('corrupted')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            worker.preflight(self.root, self.manifest, child)

    def test_resume_copies_bound_files_and_rejects_wrong_commit(self):
        original = self.run_case()
        target = self.base/'resume/cg.json'
        target.parent.mkdir()
        self.case.update(resume_from=original['result_path'], resume_from_sha256=original['result_sha256'],
                         resume_journal_sha256=original['journal_sha256'])
        worker.copy_resume(self.case, target)
        self.assertEqual(worker.sha(target), original['result_sha256'])
        self.assertEqual(worker.sha(str(target)+'.columns.jsonl'), original['journal_sha256'])
        self.case['execution_commit'] = '0'*40
        with self.assertRaisesRegex(ValueError, 'commit mismatch'):
            worker.copy_resume(self.case, target)

    def test_mip_requires_replay_and_receives_bound_environment(self):
        original = self.run_case()
        mip = dict(self.case, kind='mip', source_status=original['result_path'],
                   source_status_sha256=original['result_sha256'],
                   source_journal_sha256=original['journal_sha256'], source_cg_commit=self.commit,
                   argv=self.case['argv'] + ['--mip'])
        self.manifest['cases']['mip'] = mip
        worker.save(self.root/'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'physical replay'):
            worker.run_case(self.root, 'mip')
        self.assertFalse((self.root/'cases/mip/completion.json').exists())
        mip['argv'] += ['--physical']
        worker.save(self.root/'manifest.json', self.manifest)
        os.environ['SLURM_RESTART_COUNT'] = '1'
        result = worker.run_case(self.root, 'mip')
        self.assertTrue(result['physical_replay_validated'])
        self.assertTrue(result['target_matched'])
        self.assertTrue((self.root/'cases/mip/mip_result.json').is_symlink())

    def test_requeued_mip_registry_keeps_both_attempts(self):
        case = dict(self.case, kind='mip')
        worker.register_mip(self.root, 'case', case, '12345_r0', Path('/attempt0/result.json'))
        os.environ['SLURM_RESTART_COUNT'] = '1'
        worker.register_mip(self.root, 'case', case, '12345_r1', Path('/attempt1/result.json'))
        worker.register_mip(self.root, 'case', case, '12345_r1', Path('/attempt1/result.json'))
        entries = worker.read(self.base/'mip_preemption_study_20260911/registry.json')['cases']
        self.assertEqual(len(entries), 2)
        self.assertEqual([x['restart_count'] for x in entries], ['0','1'])
        validation_case = dict(case, is_validation=True)
        worker.register_mip(self.root, 'smoke', validation_case, '12345_r1', Path('/smoke/result.json'))
        entry = worker.read(self.base/'mip_preemption_study_20260911/registry.json')['cases'][-1]
        self.assertEqual(entry['cohort'], 'validation_overnight_diagnostics_20260914')
        self.assertTrue(entry['is_validation'])

    def test_watchdog_records_termination(self):
        attempt = self.base/'watchdog'
        attempt.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'watchdog'):
            worker.run_process([sys.executable,'-c','import time; time.sleep(20)'],
                               attempt, self.code, 0.05, os.environ.copy())
        result = worker.read(attempt/'execution.json')
        self.assertTrue(result['watchdog'])
        self.assertLess(result['wall_s'], 5)


if __name__ == '__main__':
    unittest.main()
