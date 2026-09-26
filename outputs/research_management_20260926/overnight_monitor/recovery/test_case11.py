"""Guard/provenance tests only; no optimization model is built."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cleanup_case11 as case


def original_module():
    remote = case.CODE / 'src/terminal_duplicate_cleanup.py'
    if remote.is_file():
        return remote
    return (Path(__file__).resolve().parents[4] / '.codex-work' /
            'spatial-tariff-recipe-20260925/src/terminal_duplicate_cleanup.py')


def synthetic_selection():
    # Same duplicate incidence and lengths as the pinned real source; distinct
    # filler IDs are only test data and never feed the production wrapper.
    repeated = [[48, 94, 95, 97, 98, 99, 100, 101, 102, 106], [48], [],
                [63, 64], [63, 64, 94, 95, 97, 98, 99, 100, 101, 102, 106]]
    result, next_id = [], 1000
    for values, length in zip(repeated, case.TRIP_COUNTS):
        count = length - len(values)
        result.append({'trips': values + list(range(next_id, next_id + count))})
        next_id += count
    return result


class RecoveryTests(unittest.TestCase):
    def test_patch_is_exactly_one_guard_change_and_compiles(self):
        original = original_module().read_bytes()
        revised = case.patched_source(original)
        self.assertEqual(original.replace(case.OLD, case.NEW, 1), revised)
        self.assertEqual(revised.replace(case.NEW, case.OLD, 1), original)
        self.assertEqual(sum(a != b for a, b in zip(original, revised)), 1)
        self.assertEqual(len(original), len(revised))
        old_ast, new_ast = ast.parse(original), ast.parse(revised)
        changes = [(a.value, b.value) for a, b in zip(ast.walk(old_ast), ast.walk(new_ast))
                   if isinstance(a, ast.Constant) and isinstance(b, ast.Constant) and a.value != b.value]
        self.assertEqual(changes, [(10, 11)])
        compile(revised, '<test patch>', 'exec')

    def test_changed_upstream_module_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'module hash'):
            case.patched_source(original_module().read_bytes() + b'\n')

    def test_already_patched_module_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'module hash'):
            case.patched_source(case.patched_source(original_module().read_bytes()))

    def test_wrong_source_path_and_cli_hash_are_rejected_before_read(self):
        for source, checksum in [(Path('/tmp/another.json'), case.SOURCE_SHA),
                                 (case.SOURCE, '0' * 64)]:
            with self.assertRaisesRegex(ValueError, 'pinned'):
                case.validate_source(source, checksum)

    def test_expected_enumeration_count(self):
        receipt = case.selection_counts(synthetic_selection())
        self.assertEqual(receipt['subsequence_upper_count'], 3079)
        self.assertEqual(receipt['unique_trips'], 111)
        bad = synthetic_selection()
        bad[0]['trips'].append(99999)
        with self.assertRaisesRegex(ValueError, 'audited'):
            case.selection_counts(bad)

    def test_actual_byte_hash_pins_and_input_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, selected, input_path = root / 'summary.json', root / 'selected_routes.json', root / 'input.csv'
            input_path.write_text('source input bytes\n')
            selected.write_text(json.dumps(synthetic_selection()))
            source.write_text(json.dumps(dict(fleet_cap=5, terminal_target_kwh=379.7984451,
                                              input_hashes={str(input_path): case.sha(input_path)})))
            source_hash, selection_hash = case.sha(source), case.sha(selected)
            with patch.multiple(case, SOURCE=source.resolve(), SOURCE_SHA=source_hash, SELECTED_SHA=selection_hash):
                _, receipt = case.validate_source(source, source_hash)
                self.assertEqual(receipt['subsequence_upper_count'], 3079)
                selected.write_text(selected.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'selected routes hash'):
                    case.validate_source(source, source_hash)
                selected.write_text(selected.read_text()[:-1])
                input_path.write_text('changed\n')
                with self.assertRaisesRegex(ValueError, 'input hash'):
                    case.validate_source(source, source_hash)
                source.write_text(source.read_text() + '\n')
                with self.assertRaisesRegex(ValueError, 'source summary hash'):
                    case.validate_source(source, source_hash)

    def test_collector_discovers_latest_finished_recovery_attempt(self):
        import importlib.util
        collector_path = (Path(__file__).resolve().parents[4] / 'outputs' /
                          'independent_review_20260925_spatial_tariffs/campaign/collect.py')
        if not collector_path.is_file():
            collector_path = case.ROOT / 'collect.py'
        spec = importlib.util.spec_from_file_location('recovery_test_collector', collector_path)
        collector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(collector)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for job, state in [('498962', 'failed'), ('999999', 'finished')]:
                attempt = root / 'results' / case.CELL / 'cleanup' / ('job' + job + '_r0')
                attempt.mkdir(parents=True)
                (attempt / 'attempt.json').write_text(json.dumps(dict(state=state, finished_utc='2026-09-26T09:00:00Z')))
            out, states = collector.find_attempt(root, case.CELL, 'cleanup')
            self.assertEqual(out.parent.name, 'job999999_r0')
            self.assertEqual(len(states), 2)


if __name__ == '__main__':
    unittest.main()
