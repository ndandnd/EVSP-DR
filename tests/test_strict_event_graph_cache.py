"""Small same-physics graph parity and fail-closed persistence gates; no solver."""
import copy
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from test_event_pricer_network import two_trip_problem, prices
import test_inherit_capacity_pool as inheritance_fixture
from strict_event_graph_cache import (
    BUFFERS, MAGIC, graph_fingerprint, load_graph_cache, write_graph_cache,
)
import run_capacity_speed_event_cg as runner
from run_exact_pool_mip import validate_injected_route
from audit_giro_known_columns import HORIZON_MIN, ProblemData


class StrictGraphCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.path = self.root / "graph.cache"
        self.args = SimpleNamespace(
            arm="parx60", arc_mode="lazy", battery_kwh=239.01,
            reserve_kwh=35.8515, soc_step=2.5, block_min=5,
            non_parx_kw=240.0, max_station_wait_min=1560.0,
            capacity_selector="reference", graph_cache=self.path,
            mode="prepare-graph", threads=1, cg_wall_s=5.0,
            max_iters=2, rc_eps=1e-5, resume=False,
        )
        self.problem = two_trip_problem(first_energy=160.0)
        self.prices = prices()
        self.prov = {"git_commit": "test-execution-commit",
            "instance_sha256": "a" * 64, "prices_sha256": "b" * 64,
            "reference_sha256": "c" * 64, "deadhead_sha256": "d" * 64}
        self.identity = runner.strict_graph_identity(self.args, self.problem, self.prices, self.prov)
        self.network = runner.build_network(self.args, self.problem, self.prices)

    def write(self):
        return write_graph_cache(self.path, self.network, self.identity, build_s=19.0)

    def edit_manifest(self, mutate):
        data = self.path.read_bytes()
        trailer = len(MAGIC) + 8
        length = struct.unpack("<Q", data[-trailer:-len(MAGIC)])[0]
        offset = len(data) - trailer - length
        manifest = json.loads(data[offset:offset + length])
        mutate(manifest)
        encoded = json.dumps(manifest, separators=(",", ":")).encode()
        self.path.write_bytes(data[:offset] + encoded + struct.pack("<Q", len(encoded)) + MAGIC)

    def test_reload_exact_fingerprints_fixed_duals_and_physical_routes(self):
        manifest = self.write()
        loaded, observed = load_graph_cache(self.path, self.identity)
        self.assertEqual(observed, manifest)
        self.assertEqual(graph_fingerprint(loaded), graph_fingerprint(self.network))
        self.assertEqual(loaded.metrics(), self.network.metrics())
        self.assertEqual(loaded._window_cache, {})
        self.assertEqual(loaded._selected_action_cache, {})
        for name, _code in BUFFERS:
            self.assertTrue(np.shares_memory(getattr(loaded, name + "_np"),
                np.frombuffer(getattr(loaded, name), dtype=getattr(loaded, name + "_np").dtype)))
        for duals in ({0: 0.0, 1: 0.0}, {0: 110000.0, 1: 120000.0},
                      {0: 170000.0, 1: -5000.0}, {0: 1.0, 1: 220000.0}):
            for objective in ("combined-cost", "artificial-elimination", "fleet-only", "charging-cost"):
                with self.subTest(duals=duals, objective=objective):
                    cold = self.network.min_reduced_cost_route(duals, objective=objective)
                    warm = loaded.min_reduced_cost_route(duals, objective=objective)
                    self.assertEqual(cold["rc"], warm["rc"])
                    self.assertEqual(cold["trips"], warm["trips"])
                    self.assertEqual(cold["_event_record"], warm["_event_record"])
                    reason = validate_injected_route(self.problem, warm["_event_record"],
                        self.args.battery_kwh, self.args.non_parx_kw, self.args.reserve_kwh,
                        HORIZON_MIN, arrival_grace_min=0.0,
                        station_charge_kw=runner.station_power(self.args.arm))
                    self.assertIsNone(reason)
        fixed = loaded.fixed_sequence_record((0, 1))
        self.assertIsNotNone(fixed)
        self.assertTrue(fixed["expanded_grid_charging_stops"]["stations"])
        self.assertEqual(self.network.fixed_sequence_record((0, 1)), fixed)
        self.assertIsNone(validate_injected_route(self.problem, fixed,
            self.args.battery_kwh, self.args.non_parx_kw, self.args.reserve_kwh,
            HORIZON_MIN, arrival_grace_min=0.0,
            station_charge_kw=runner.station_power(self.args.arm)))

    def test_dataframe_backed_problem_roundtrips(self):
        problem = ProblemData(frame=pd.DataFrame({"Ordered_Trip_ID": [101, 102]}),
            trips=self.problem.trips, adjacency=self.problem.adjacency,
            start_min=self.problem.start_min, end_min=self.problem.end_min,
            trip_energy=self.problem.trip_energy)
        network = runner.build_network(self.args, problem, self.prices)
        write_graph_cache(self.path, network, self.identity, build_s=0.0)
        loaded, _ = load_graph_cache(self.path, self.identity)
        pd.testing.assert_frame_equal(loaded.problem.frame, problem.frame)
        self.assertEqual(graph_fingerprint(loaded), graph_fingerprint(network))
        self.assertEqual(loaded.fixed_sequence_record((0, 1)),
                         network.fixed_sequence_record((0, 1)))

    def test_identity_changes_rejected_before_deserialization(self):
        self.write()
        variants = []
        for field in ("implementation_git_commit", "problem_sha256", "event_lattice_sha256",
                      "normalized_prices_sha256"):
            changed = copy.deepcopy(self.identity)
            changed[field] = "changed"
            variants.append(changed)
        for field in ("instance_sha256", "prices_sha256", "reference_sha256", "deadhead_sha256"):
            changed = copy.deepcopy(self.identity)
            changed["inputs"][field] = "changed"
            variants.append(changed)
        for field in ("battery_kwh", "reserve_kwh", "soc_step_kwh", "event_block_min",
                      "max_station_wait_min", "station_charge_kw", "capacity_selector"):
            changed = copy.deepcopy(self.identity)
            changed["physics"][field] = "changed"
            variants.append(changed)
        changed = copy.deepcopy(self.identity)
        changed["trip_order"].reverse()
        variants.append(changed)
        changed = copy.deepcopy(self.identity)
        changed["source_sha256"]["event_pricer_network.py"] = "changed"
        variants.append(changed)
        for expected in variants:
            with mock.patch("strict_event_graph_cache.pickle.load") as deserialize:
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    load_graph_cache(self.path, expected)
                deserialize.assert_not_called()

    def test_derived_identity_changes_with_actual_problem_and_price_data(self):
        for change in ("order", "energy", "adjacency", "prices"):
            problem, tariff = copy.deepcopy(self.problem), copy.deepcopy(self.prices)
            if change == "order":
                problem.trips = tuple(reversed(problem.trips))
            elif change == "energy":
                problem.trip_energy[0] += 1.0
            elif change == "adjacency":
                problem.adjacency[0].reverse()
            else:
                next(iter(tariff.values()))[0] += 0.1
            identity = runner.strict_graph_identity(self.args, problem, tariff, self.prov)
            self.assertNotEqual(self.identity, identity)

    def test_corruption_and_truncation_rejected_before_deserialization(self):
        self.write()
        original = self.path.read_bytes()
        variants = [original[:8], original[:-1], original[:-100],
                    bytes([original[0] ^ 1]) + original[1:]]
        for data in variants:
            self.path.write_bytes(data)
            with mock.patch("strict_event_graph_cache.pickle.load") as deserialize:
                with self.assertRaises(ValueError):
                    load_graph_cache(self.path, self.identity)
                deserialize.assert_not_called()

    def test_manifest_layout_and_metrics_mismatch_rejected(self):
        self.write()
        original = self.path.read_bytes()
        for mutate in (lambda doc: doc.update(schema="old"),
                       lambda doc: doc.update(metadata_bytes=doc["metadata_bytes"] + 1),
                       lambda doc: doc["network"].update(dag_arcs=0),
                       lambda doc: doc["buffers"][0].update(count=-1)):
            self.path.write_bytes(original)
            self.edit_manifest(mutate)
            with self.assertRaises(ValueError):
                load_graph_cache(self.path, self.identity)

    def test_interrupted_write_never_publishes_partial_and_cleans_temporary(self):
        with mock.patch("strict_event_graph_cache.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaisesRegex(OSError, "interrupted"):
                self.write()
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        self.write()
        original = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.path])

    def test_wrong_object_identity_and_existing_lock_reject_export(self):
        self.network.events[next(iter(self.network.events))] = (1.0,)
        with self.assertRaisesRegex(ValueError, "object identity"):
            self.write()
        self.assertFalse(self.path.exists())
        lock = self.path.with_name(self.path.name + ".lock")
        lock.touch()
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertTrue(lock.exists())

    def test_prepare_and_verify_never_create_master_or_pool(self):
        with mock.patch.object(runner, "ExactCapacityMaster") as master:
            prepared = runner.run_graph_preparation(self.args, self.problem, self.prices,
                self.prov, self.root / "prepare.json")
            self.args.mode = "verify-graph"
            with mock.patch.object(runner, "build_network") as builder:
                verified = runner.run_graph_preparation(self.args, self.problem, self.prices,
                    self.prov, self.root / "verify.json")
                builder.assert_not_called()
            master.assert_not_called()
        self.assertFalse(prepared["solver_started"])
        self.assertFalse(verified["solver_started"])
        self.assertEqual(prepared["graph_payload_sha256"], verified["graph_payload_sha256"])
        self.assertEqual(verified["network_build_s"], 0)
        self.assertGreaterEqual(prepared["runtime_s"], prepared["network_build_s"])
        self.assertEqual(list(self.root.glob("*.jsonl")), [])

    def test_preparation_reserves_before_build_and_rejects_output_alias(self):
        with mock.patch.object(runner, "build_network") as builder:
            with self.assertRaisesRegex(ValueError, "must differ"):
                runner.run_graph_preparation(self.args, self.problem, self.prices,
                    self.prov, self.path)
            builder.assert_not_called()
        def build_while_reserved(*args):
            self.assertTrue(self.path.with_name(self.path.name + ".lock").exists())
            with mock.patch.object(runner, "build_network") as second_builder:
                with self.assertRaises(FileExistsError):
                    runner.run_graph_preparation(self.args, self.problem, self.prices,
                        self.prov, self.root / "second.json")
                second_builder.assert_not_called()
            return self.network
        with mock.patch.object(runner, "build_network", side_effect=build_while_reserved):
            runner.run_graph_preparation(self.args, self.problem, self.prices,
                self.prov, self.root / "first.json")

    def test_cg_missing_cache_and_old_pool_fail_before_graph_or_solver(self):
        with mock.patch.object(runner, "ExactCapacityMaster") as master, \
             mock.patch.object(runner, "build_network") as builder:
            with self.assertRaises(FileNotFoundError):
                runner.run_cg(self.args, self.problem, self.prices, self.prov,
                    self.root / "cg.json", self.root / "pool.jsonl")
            self.args.resume = True
            pool = self.root / "pool.jsonl"
            pool.write_text(json.dumps({"trips": [0], "cost": 1,
                                       "cg_checkpoint_id": "old-commit"}) + "\n")
            with self.assertRaisesRegex(ValueError, "old execution commits are blocked"):
                runner.run_cg(self.args, self.problem, self.prices, self.prov,
                    self.root / "cg.json", pool)
            builder.assert_not_called()
            master.assert_not_called()

    def test_cached_load_excluded_from_allowance_and_legacy_build_included(self):
        self.write()
        clock = SimpleNamespace(t=0.0)
        class Master:
            def __init__(inner, *args, **kwargs):
                inner.routes = []
            def add_route(inner, route):
                inner.routes.append(route)
            def solve(inner):
                return {"objective": 100000.0, "runtime_s": 0.0, "rows": 2,
                        "columns": len(inner.routes), "nonzeros": 2,
                        "artificial_total": 0.0, "route_weight": 1.0,
                        "trip_duals": {0: 0.0, 1: 0.0}, "capacity_duals": {}}
        load = runner.load_graph_cache
        def slow_load(*args):
            clock.t += 70.0
            return load(*args)
        def slow_build(*args):
            clock.t += 70.0
            return self.network
        with mock.patch.object(runner, "ExactCapacityMaster", Master), \
             mock.patch.object(runner, "load_graph_cache", side_effect=slow_load):
            cached = runner.run_cg(self.args, self.problem, self.prices, self.prov,
                self.root / "cached.json", self.root / "cached.pool", clock=lambda: clock.t)
        self.assertEqual(len(cached["iterations"]), 1)
        self.assertEqual(cached["network_load_s"], 70.0)
        self.assertEqual(cached["runtime_s"], 70.0)
        self.assertEqual(cached["solver_runtime_s"], 0.0)
        self.assertEqual(cached["cg_allowance_scope"], "solver_after_verified_graph_load")
        self.args.graph_cache = None
        with mock.patch.object(runner, "ExactCapacityMaster", Master), \
             mock.patch.object(runner, "build_network", side_effect=slow_build):
            legacy = runner.run_cg(self.args, self.problem, self.prices, self.prov,
                self.root / "legacy.json", self.root / "legacy.pool", clock=lambda: clock.t)
        self.assertEqual(legacy["iterations"], [])
        self.assertEqual(legacy["stop_reason"], "cg_wall_limit")
        self.assertEqual(legacy["cg_allowance_scope"], "legacy_graph_inclusive")


class PinnedParentLineageTests(unittest.TestCase):
    def test_exact_k17_commit_gate_preserves_replay_hash_physics_and_remapping(self):
        fixture = inheritance_fixture.InheritanceTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        pin = "35770aae2c08e7d5a356cc3b673e67608e5b1036"
        fixture.doc["provenance"]["git_commit"] = pin
        physics = {"battery_kwh": 239.01, "reserve_kwh": 35.8515,
            "soc_step_kwh": 2.5, "event_block_min": 5, "non_parx_kw": 240.0,
            "parx_kw": 60.0, "capacity_enforced": False, "max_station_wait_min": 1560.0}
        fixture.doc["physics"] = physics
        fixture.status.write_text(json.dumps(fixture.doc))
        with self.assertRaisesRegex(ValueError, "git_commit"):
            fixture.load(expected_physics=physics)
        replayed = []
        def replay(route):
            replayed.append(route["trips"])
            return None
        routes, meta = fixture.load(expected_physics=physics,
            compatible_parent_commit=pin, route_validator=replay)
        self.assertEqual(replayed, [[1, 2]])
        self.assertEqual(routes[0]["trips"], [1, 2])
        self.assertEqual(routes[0]["cg_checkpoint_id"], "child")
        self.assertEqual(routes[0]["inheritance"]["parent_checkpoint_id"], "parent")
        self.assertEqual(meta["parent_execution_commit"], pin)
        self.assertEqual(meta["audited_compatible_parent_commit"], pin)
        self.assertTrue(meta["every_inherited_route_replayed"])
        for bad in ("not-a-commit", "50ceb6c095a580f79f87b53bef536cac31f81963"):
            with self.assertRaisesRegex(ValueError, "git_commit|unaudited"):
                fixture.load(expected_physics=physics, compatible_parent_commit=bad)
        with self.assertRaisesRegex(ValueError, "without shared capacity"):
            fixture.load(expected_physics={**physics, "capacity_enforced": True},
                compatible_parent_commit=pin)
        with self.assertRaisesRegex(ValueError, "physics"):
            fixture.load(expected_physics={**physics, "battery_kwh": 240.0},
                compatible_parent_commit=pin)
        with self.assertRaisesRegex(ValueError, "physical replay"):
            fixture.load(expected_physics=physics, compatible_parent_commit=pin,
                route_validator=lambda route: "invalid SOC")
        with self.assertRaisesRegex(ValueError, "provenance"):
            fixture.load(expected_physics=physics, compatible_parent_commit=pin,
                child_provenance={**fixture.prov, "prices_sha256": "wrong"})
        fixture.pool.write_text(fixture.pool.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "pool hash"):
            fixture.load(expected_physics=physics, compatible_parent_commit=pin)


if __name__ == "__main__":
    unittest.main()
