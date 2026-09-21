import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'tests')]
from event_pricer_network import EventExpandedNetwork
from graph_arc_checkpoints import ArcCheckpoints
from test_event_pricer_network import two_trip_problem, prices


def network(directory=None, **changes):
    arguments = dict(soc_step=30, block_min=10, g_kwh=240,
                     charge_kw=240, reserve_kwh=0, arc_checkpoint_dir=directory,
                     arc_checkpoint_identity={'git_commit': 'fixture', 'input_sha256': 'fixture'})
    arguments.update(changes)
    return EventExpandedNetwork(two_trip_problem(), prices(), **arguments)


def real_network(directory=None):
    from test_event_pricer_tie_keys import build_problem, restricted_problem
    problem = build_problem(REPO / 'data/overnight_decomposition_20260912',
                            'd00_g3.csv', reference_data_dir=REPO / 'data')
    problem = restricted_problem(problem, problem.trips[:3])
    return EventExpandedNetwork(problem, prices(), soc_step=30, block_min=10,
                                g_kwh=240, charge_kw=240, reserve_kwh=0,
                                arc_checkpoint_dir=directory,
                                arc_checkpoint_identity={'fixture': 'real3'})


def fingerprint(graph):
    value = {
        'targets': hashlib.sha256(graph._arc_targets.tobytes()).hexdigest(),
        'costs': hashlib.sha256(graph._arc_costs.tobytes()).hexdigest(),
        'recipes': hashlib.sha256(graph._arc_recipes.tobytes()).hexdigest(),
        'slices': graph._arc_slices,
        'sink': graph.sink_arcs,
        'metrics': graph.metrics(),
        'fixed': graph.fixed_sequence_record((0, 1)),
        'queries': [graph.min_reduced_cost_route({0: a, 1: b})
                    for a, b in [(0, 0), (100000, 100000), (150000, 80000)]],
    }
    return json.loads(json.dumps(value, default=lambda x: x.tolist()))


class GraphCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'checkpoints'
        self.reference = fingerprint(network())

    def tearDown(self):
        self.temp.cleanup()

    def crash_child(self, before_manifest=False, real=False):
        program = '''
import os, signal, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1] + '/src', sys.argv[1] + '/tests']
from test_graph_arc_checkpoints import network, real_network
from graph_arc_checkpoints import ArcCheckpoints
original = ArcCheckpoints.maybe_commit
counter = 0
if sys.argv[3] == 'before_manifest':
    replace = os.replace
    def crash_replace(source, destination):
        global counter
        if Path(destination).name == 'manifest.json':
            counter += 1
            if counter == 3:
                os.kill(os.getpid(), signal.SIGKILL)
        return replace(source, destination)
    os.replace = crash_replace

def commit(self, graph, finished, **kwargs):
    self.shard_bytes = 1
    original(self, graph, finished, **kwargs)
    if sys.argv[3] == 'after_manifest' and finished == 3:
        os.kill(os.getpid(), signal.SIGKILL)
ArcCheckpoints.maybe_commit = commit
(real_network if sys.argv[4] == 'real' else network)(sys.argv[2])
'''
        result = subprocess.run([sys.executable, '-c', program, str(REPO), str(self.path),
                                 'before_manifest' if before_manifest else 'after_manifest', 'real' if real else 'toy'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, -signal.SIGKILL, result.stderr)

    def test_killed_after_commit_skips_rows_and_is_identical(self):
        self.crash_child()
        calls = []
        original = EventExpandedNetwork._finalize_source
        def finalize(graph, source):
            calls.append(source)
            return original(graph, source)
        with mock.patch.object(EventExpandedNetwork, '_finalize_source', finalize):
            graph = network(self.path)
        self.assertEqual(graph.graph_checkpoint_report['resumed_sources'], 3)
        source_order = [0] + [s for _, s in sorted(graph.trip_node.items())]
        self.assertEqual(calls, source_order[3:])
        self.assertEqual(fingerprint(graph), self.reference)

    def test_killed_before_manifest_ignores_orphan_shard(self):
        self.crash_child(before_manifest=True)
        self.assertTrue((self.path / 'shard-000002.bin').exists())
        graph = network(self.path)
        self.assertEqual(graph.graph_checkpoint_report['resumed_sources'], 2)
        self.assertEqual(fingerprint(graph), self.reference)

    def test_complete_checkpoint_reuses_all_rows_and_keeps_cache_compatible(self):
        first = network(self.path)
        with mock.patch.object(EventExpandedNetwork, '_finalize_source', side_effect=AssertionError('rebuilt')):
            restored = network(self.path)
        self.assertEqual(fingerprint(first), fingerprint(restored))
        from exact_pricer_expanded import _write_event_network_cache, _load_event_network_cache
        cache = Path(self.temp.name) / 'complete.pkl'
        _write_event_network_cache(cache, restored, {'fixture': 1}, 0)
        loaded, _ = _load_event_network_cache(cache, {'fixture': 1})
        self.assertEqual(fingerprint(loaded), self.reference)

    def test_corrupt_binary_is_rejected(self):
        network(self.path)
        path = self.path / 'shard-000000.bin'
        data = bytearray(path.read_bytes());data[0] ^= 1;path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'hash/size'):
            network(self.path)

    def test_corrupt_row_metadata_is_rejected(self):
        network(self.path)
        path = self.path / 'manifest.json';doc = json.loads(path.read_text())
        doc['shards'][0]['rows'][0][2] += 1;path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, 'offsets|hash'):
            network(self.path)

    def test_changed_physics_and_provenance_are_rejected(self):
        network(self.path)
        for changes in [dict(charge_kw=120), dict(arc_checkpoint_identity={'git_commit': 'changed'})]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'identity'):
                network(self.path, **changes)

    def test_changed_trip_and_tariff_are_rejected(self):
        network(self.path)
        altered_prices = prices();altered_prices[next(iter(altered_prices))][1] = 7
        for problem, tariff in [(two_trip_problem(180), prices()), (two_trip_problem(), altered_prices)]:
            with self.assertRaisesRegex(ValueError, 'identity'):
                EventExpandedNetwork(problem, tariff, soc_step=30, block_min=10,
                                     g_kwh=240, charge_kw=240, reserve_kwh=0,
                                     arc_checkpoint_dir=self.path,
                                     arc_checkpoint_identity={'git_commit':'fixture','input_sha256':'fixture'})

    def test_concurrent_writer_is_rejected_and_lock_released_on_error(self):
        self.path.mkdir()
        with (self.path / 'writer.lock').open('a') as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                network(self.path)
        with mock.patch.object(EventExpandedNetwork, '_direct_arcs', side_effect=RuntimeError('interrupt')):
            with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                network(self.path)
        self.assertEqual(fingerprint(network(self.path)), self.reference)

    def test_real_csv_rebuilds_same_identity_across_processes(self):
        self.crash_child(real=True)
        restored = real_network(self.path)
        self.assertEqual(restored.graph_checkpoint_report['resumed_sources'], 3)
        self.assertEqual(fingerprint(restored), fingerprint(real_network()))

    def test_explicit_mode_rejects_checkpoints(self):
        with self.assertRaisesRegex(ValueError, 'lazy packed'):
            network(self.path, arc_mode='explicit')


if __name__ == '__main__':
    unittest.main()
