import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import audit_matching_cover_pricing as matching_audit  # noqa: E402
from audit_giro_known_columns import ProblemData  # noqa: E402


def _provenance(route_count, *, direct_only=False):
    return {
        "compatibility_mode": "direct_only" if direct_only else "full",
        "matching_attempt_index": 0,
        "matching_retry_count": 0,
        "matching_attempts_considered": 1,
        "unique_matchings_tried": 1,
        "matching_order_seed": 0,
        "matching_order": "natural",
        "matching_cardinality": 1,
        "path_count": route_count,
        "relaxed_minimum_path_count": route_count,
        "resource_feasible_path_count": route_count,
        "resource_repair_mode": "none",
        "is_exact_minimum_path_cover": True,
        "contiguous_splits_added": 0,
        "max_successor_charge_targets": 64,
    }


def _route(trips, cost, *, provenance=None):
    route = {
        "route": ["PARX_0", *trips, "PARX_0"],
        "cost": float(cost),
        "charging_stops": {"stations": [], "cst": [], "cet": [], "kwh": []},
        "charging_activities": 0,
        "deadhead_kwh": 0.0,
        "type": "truck",
    }
    if provenance is not None:
        route["_matching_init"] = dict(provenance)
    return route


def _column(trips, cost):
    route = _route(trips, cost)
    return matching_audit.CostedColumn(
        route=route,
        trips=tuple(trips),
        incidence=frozenset(trips),
        master_cost=float(cost),
    )


class SavedInputValidationTests(unittest.TestCase):
    def test_nested_cluster_path_is_resolved_by_suffix_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory) / "data"
            nested = data_dir / "random" / "instance.csv"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"instance-bytes")
            expected = hashlib.sha256(nested.read_bytes()).hexdigest()

            resolved, actual = matching_audit.resolve_saved_data_path(
                data_dir,
                "/cluster/work/EVSP-DR/data/random/instance.csv",
                expected_sha256=expected,
                label="instance",
            )

        self.assertEqual(resolved, nested.resolve())
        self.assertEqual(actual, expected)

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            path = data_dir / "instance.csv"
            path.write_bytes(b"wrong")
            with self.assertRaisesRegex(
                matching_audit.MatchingCoverAuditInputError,
                "SHA-256 mismatch",
            ):
                matching_audit.resolve_saved_data_path(
                    data_dir,
                    "instance.csv",
                    expected_sha256="0" * 64,
                    label="instance",
                )

    def test_saved_pool_rejects_invalid_seed_boundary(self):
        pool = {
            "csv_name": "instance.csv",
            "prices_csv": "prices.csv",
            "trip_ids": [0],
            "routes": [],
            "seed_route_count": 1,
        }
        with self.assertRaisesRegex(
            matching_audit.MatchingCoverAuditInputError,
            "between zero and len",
        ):
            matching_audit.validate_saved_pool(pool)


class MatchingCoverValidationTests(unittest.TestCase):
    def test_validates_partition_and_shared_resource_provenance(self):
        provenance = _provenance(2)
        routes = [
            _route([0, 2], 100.0, provenance=provenance),
            _route([1], 100.0, provenance=provenance),
        ]

        result = matching_audit.validate_matching_cover(
            routes, [0, 1, 2], direct_only=False
        )

        self.assertTrue(result["exact_trip_partition"])
        self.assertTrue(result["resource_provenance_validated"])
        self.assertEqual(result["partition_trip_count"], 3)
        self.assertEqual(result["unique_incidence_count"], 2)

    def test_rejects_duplicate_trip_across_matching_routes(self):
        provenance = _provenance(2)
        routes = [
            _route([0, 1], 100.0, provenance=provenance),
            _route([1], 100.0, provenance=provenance),
        ]
        with self.assertRaisesRegex(
            matching_audit.MatchingCoverAuditInputError,
            "not an exact trip partition",
        ):
            matching_audit.validate_matching_cover(
                routes, [0, 1], direct_only=False
            )


class NegativeWaveTests(unittest.TestCase):
    def test_adds_only_negative_matching_column_and_disclaims_optimality(self):
        base = [_column([0], 100.0), _column([1], 100.0)]
        matching = [_column([0, 1], 150.0)]

        result = matching_audit.run_negative_matching_waves(
            [0, 1], base, matching
        )

        self.assertEqual(result["waves"][0]["negative_matching_routes"], 1)
        self.assertAlmostEqual(
            result["waves"][0]["added"][0]["reduced_cost_at_selected_dual"],
            -50.0,
        )
        self.assertEqual(result["waves"][1]["negative_matching_routes"], 0)
        self.assertAlmostEqual(result["final_master"]["objective"], 150.0)
        self.assertAlmostEqual(result["final_master"]["route_weight"], 1.0)
        self.assertFalse(result["pricing_optimality_certified"])
        self.assertIn("finite matching cover", result["scope_warning"])

    def test_equal_incidence_cheaper_candidate_is_not_mistaken_for_present(self):
        expensive = [_column([0], 120.0)]
        cheaper = [_column([0], 100.0)]

        result = matching_audit.run_negative_matching_waves(
            [0], expensive, cheaper
        )

        self.assertEqual(result["waves"][0]["negative_matching_routes"], 1)
        self.assertAlmostEqual(result["final_master"]["objective"], 100.0)


class EndToEndAuditTests(unittest.TestCase):
    def test_audit_rebuilds_costs_and_runs_seed_and_full_pool_waves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            instance = data_dir / "nested" / "instance.csv"
            prices = data_dir / "prices.csv"
            instance.parent.mkdir(parents=True)
            instance.write_text("trip input\n", encoding="utf-8")
            prices.write_text("price input\n", encoding="utf-8")

            saved_routes = [_route([0], 100.0), _route([1], 100.0)]
            pool = {
                "csv_name": "/cluster/repo/data/nested/instance.csv",
                "prices_csv": "/cluster/repo/data/prices.csv",
                "instance_sha256": hashlib.sha256(instance.read_bytes()).hexdigest(),
                "price_sha256": hashlib.sha256(prices.read_bytes()).hexdigest(),
                "trip_ids": [0, 1],
                "routes": saved_routes,
                "seed_route_count": 2,
                "mode": "GREEDY",
                "battery_kwh": 300,
                "run_arguments": {
                    "G": 300,
                    "max_charge2trip": 1560.0,
                    "successor_charge_targets": True,
                    "max_successor_charge_targets": 64,
                    "matching_direct_only": False,
                    "matching_attempts": 32,
                    "matching_order_seed": 0,
                },
            }
            pool_path = root / "routes_colgen_final_test.json"
            pool_path.write_text(json.dumps(pool), encoding="utf-8")

            problem = ProblemData(
                frame=None,
                trips=(0, 1),
                adjacency={},
                start_min={0: 0.0, 1: 10.0},
                end_min={0: 5.0, 1: 15.0},
                trip_energy={0: 1.0, 1: 1.0},
            )
            matching_routes = [
                _route([0, 1], 150.0, provenance=_provenance(1))
            ]
            station_prices = {
                base: {0: 0.1}
                for base in matching_audit.STATION_NODE_BY_BASE
            }

            with (
                patch.object(matching_audit, "build_problem", return_value=problem),
                patch.object(
                    matching_audit,
                    "load_station_hourly_prices",
                    return_value=station_prices,
                ),
                patch.object(
                    matching_audit,
                    "build_matching_initial_routes",
                    return_value=matching_routes,
                ) as build_matching,
                patch.object(
                    matching_audit,
                    "_route_cost",
                    side_effect=lambda route, _hourly, _station: route["cost"],
                ) as route_cost,
            ):
                report = matching_audit.audit_matching_cover_pool(
                    pool_path, data_dir
                )

        self.assertTrue(report["inputs"]["instance_hash_matches_saved"])
        self.assertTrue(report["inputs"]["price_hash_matches_saved"])
        self.assertTrue(report["inputs"]["current_model_only"])
        self.assertFalse(report["inputs"]["incumbent_assignment_dependency"])
        self.assertEqual(route_cost.call_count, 3)
        self.assertEqual(
            build_matching.call_args.kwargs["matching_order_seed"], 0
        )
        self.assertEqual(
            build_matching.call_args.kwargs["deadhead_cost_per_kwh"], 0.0
        )
        self.assertEqual(
            report["negative_only_waves"]["from_saved_seeds"]["waves"][0][
                "negative_matching_routes"
            ],
            1,
        )
        self.assertEqual(
            report["negative_only_waves"]["from_saved_full_pool"]["waves"][0][
                "negative_matching_routes"
            ],
            1,
        )
        self.assertFalse(report["interpretation"]["pricing_optimality_certified"])

    def test_main_prints_json_even_when_output_is_requested(self):
        report = {"answer": 42}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "audit.json"
            stdout = io.StringIO()
            with (
                patch.object(
                    matching_audit,
                    "_parse_args",
                    return_value=Namespace(
                        pool=Path("unused.json"),
                        data_dir=Path("unused-data"),
                        output=output,
                    ),
                ),
                patch.object(
                    matching_audit,
                    "audit_matching_cover_pool",
                    return_value=report,
                ),
                contextlib.redirect_stdout(stdout),
            ):
                return_code = matching_audit.main()

            written = json.loads(output.read_text(encoding="utf-8"))
            printed = json.loads(stdout.getvalue())

        self.assertEqual(return_code, 0)
        self.assertEqual(written, report)
        self.assertEqual(printed, report)


if __name__ == "__main__":
    unittest.main()
