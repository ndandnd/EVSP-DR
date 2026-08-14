import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import profile_exact_pool_prefixes as profiler  # noqa: E402
import benchmark_exact_pricing as pricing_benchmark  # noqa: E402
from durable_io import read_jsonl_records  # noqa: E402
from exact_cg_telemetry import PhaseTelemetry  # noqa: E402


class ExactCgTelemetryTests(unittest.TestCase):
    def test_shortest_path_materializes_duals_once_per_trip(self):
        class CountingDuals(dict):
            def __init__(self, values):
                super().__init__(values)
                self.lookups = []

            def get(self, key, default=None):
                self.lookups.append(key)
                return super().get(key, default)

        network = pricing_benchmark.ExpandedNetwork.__new__(
            pricing_benchmark.ExpandedNetwork
        )
        network.problem = SimpleNamespace(trips=[10, 20])
        network.topo = [0, 2, 1]
        network.out = [
            [(2, 100.0, 0)],
            [],
            [(1, 0.0, -1)],
        ]
        network.node_meta = [
            ("source", None, None),
            ("sink", None, None),
            ("trip", 10, 0),
        ]
        network.grid = [0.0]
        duals = CountingDuals({10: 25.0, 20: 50.0})

        route = network.min_reduced_cost_route(duals)

        self.assertEqual(duals.lookups, [10, 20])
        self.assertEqual(route["trips"], [10])
        self.assertEqual(route["rc"], 75.0)

    def test_pair_routes_match_preoptimization_golden(self):
        args = Namespace(
            data_dir=REPO_ROOT / "data",
            prices_data_dir=REPO_ROOT / "data",
            prices_csv="hourly_prices_flat.csv",
            soc_step=15.0,
            block_min=10,
            g_kwh=300.0,
            charge_kw=300.0,
            reserve=0.0,
            columns=30,
            dual_value=100000.0,
            dual_mode="constant",
            warmup=0,
            repeat=1,
        )
        result = pricing_benchmark.benchmark_case(
            args, "Practice_Custom_TwoDuty_13301_13302.csv"
        )

        self.assertEqual(result["network_nodes"], 21328)
        self.assertEqual(result["network_arcs"], 1077619)
        self.assertEqual(result["sink_arcs"], 21111)
        self.assertEqual(result["returned_routes"], 30)
        self.assertEqual(
            result["route_sha256"],
            "152e5691becae2953166d586827b570b5d3593cf5d5967c77846e92775cd63e5",
        )
        self.assertAlmostEqual(
            result["best_route"]["rc"], -4499913.039999999, places=6
        )

        args.dual_mode = "heterogeneous"
        heterogeneous = pricing_benchmark.benchmark_case(
            args, "Practice_Custom_TwoDuty_13301_13302.csv"
        )
        self.assertEqual(heterogeneous["returned_routes"], 30)
        self.assertEqual(
            heterogeneous["route_sha256"],
            "5a4c6434a2dc70cfb2195b3cfaf516848217e01e6e409a57b98ec65deb19fc2c",
        )
        self.assertAlmostEqual(
            heterogeneous["best_route"]["rc"],
            -4530091.575999999,
            places=6,
        )

    def test_sidecar_repairs_only_trailing_row_and_resumes_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phases.jsonl"
            identity = {"output": "run.json", "instance_sha256": "abc"}
            first = PhaseTelemetry(path, identity=identity)
            first.phase("network_build", 1.25, network_nodes=10)
            with path.open("ab") as handle:
                handle.write(b'{"schema":')

            second = PhaseTelemetry(path, identity=identity)
            second.phase("master_attempt", 2.5, iteration=1)

            records = read_jsonl_records(path, repair_trailing=False)
            self.assertEqual(
                [record["record_type"] for record in records],
                ["session_start", "phase", "session_start", "phase"],
            )
            self.assertEqual(records[-1]["phase"], "master_attempt")
            self.assertEqual(records[-1]["session"], 2)

    def test_foreign_truncated_sidecar_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "foreign.jsonl"
            PhaseTelemetry(path, identity={"output": "foreign.json"})
            with path.open("ab") as handle:
                handle.write(b'{"schema":')
            original = path.read_bytes()

            with self.assertRaisesRegex(
                ValueError, "belongs to different work"
            ):
                PhaseTelemetry(path, identity={"output": "current.json"})

            self.assertEqual(path.read_bytes(), original)

    def test_ordered_prefixes_preserve_first_reached_pool_semantics(self):
        records = [
            {"trips": [1], "cost": 5.0},
            {"trips": [2], "cost": 6.0},
            {"trips": [1], "cost": 4.0},
            {"trips": [1, 2], "cost": 7.0},
        ]
        prefixes = profiler.ordered_unique_prefixes(
            records, [1, 2], [1, 2, 3]
        )
        self.assertEqual([r["cost"] for r in prefixes[1]], [5.0])
        self.assertEqual([r["cost"] for r in prefixes[2]], [5.0, 6.0])
        self.assertEqual(
            sorted(r["cost"] for r in prefixes[3]), [4.0, 6.0, 7.0]
        )

    def test_profiler_is_read_only_and_runs_all_requested_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            instance = data / "instance.csv"
            prices = data / "prices.csv"
            instance.write_text("instance\n")
            prices.write_text("prices\n")
            result = root / "pool.snapshot.json"
            journal = Path(str(result) + ".columns.jsonl")
            journal.write_text(
                json.dumps({"trips": [1], "cost": 100000.0}) + "\n"
                + json.dumps({"trips": [2], "cost": 100000.0}) + "\n"
            )
            result.write_text(json.dumps({
                "csv": "instance.csv",
                "prices_csv": "prices.csv",
                "trip_ids": [1, 2],
                "master_sense": "partition",
                "columns_journal": str(journal),
                "provenance": {
                    "instance_sha256": hashlib.sha256(
                        instance.read_bytes()
                    ).hexdigest(),
                    "prices_sha256": hashlib.sha256(
                        prices.read_bytes()
                    ).hexdigest(),
                },
            }))
            source_bytes = {
                path: path.read_bytes()
                for path in (result, journal, instance, prices)
            }
            methods = []

            def solve(**kwargs):
                methods.append(kwargs["method"])
                return SimpleNamespace(
                    runtime_s=0.01,
                    objective=200000.0,
                    route_weight=2.0,
                    artificial_total=0.0,
                    max_row_violation=0.0,
                    max_bound_violation=0.0,
                )

            args = Namespace(
                result=result,
                prefixes=[1, 2, 5],
                methods=["highs", "highs-ds", "highs-ipm"],
                time_limit_s=None,
                out=None,
            )
            with (
                patch.object(profiler, "DATA_DIR", data),
                patch.object(
                    profiler, "solve_restricted_master_lp",
                    side_effect=solve,
                ),
            ):
                payload = profiler.profile(args)

            self.assertEqual(
                methods,
                ["highs", "highs-ds", "highs-ipm"] * 2,
            )
            self.assertTrue(payload["source_unchanged"])
            self.assertEqual(
                [row["available"] for row in payload["profiles"]],
                [True, True, False],
            )
            for path, original in source_bytes.items():
                self.assertEqual(path.read_bytes(), original)

            args.out = journal
            with (
                patch.object(profiler, "DATA_DIR", data),
                self.assertRaisesRegex(
                    ValueError, "must not overwrite"
                ),
            ):
                profiler.profile(args)
            self.assertEqual(journal.read_bytes(), source_bytes[journal])


if __name__ == "__main__":
    unittest.main()
