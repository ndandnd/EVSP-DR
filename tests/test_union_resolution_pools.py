import sys
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from target_pool_feasibility import solve_target_feasibility  # noqa: E402
from union_resolution_pools import (  # noqa: E402
    merge_route_sets,
    route_sha256,
)


def route(trips, cost=100000.0):
    return {
        "trips": trips,
        "route_nodes": ["D", *trips, "D"],
        "charging_stops": {
            "stations": [], "cst": [], "cet": [], "kwh": [],
        },
        "expanded_grid_charging_stops": {
            "stations": [], "cst": [], "cet": [], "kwh": [],
        },
        "continuous_realized_charging_blocks": [],
        "cost": cost,
    }


def identity(instance="i", g_kwh=300.0):
    return {
        "instance_sha256": instance,
        "prices_sha256": "p",
        "reference_sha256": "r",
        "deadhead_sha256": "d",
        "g_kwh": g_kwh,
        "charge_kw": 300.0,
        "min_soc_frac": 0.0,
        "csv": "instance.csv",
        "prices_csv": "prices.csv",
        "trip_ids": [0, 1],
    }


class ResolutionPoolUnionTests(unittest.TestCase):
    def test_union_deduplicates_hashes_and_is_source_superset(self):
        shared = route([0])
        sources = [
            {
                "identity": identity(),
                "journal_sha256": "a",
                "routes": [shared, route([1])],
            },
            {
                "identity": identity(),
                "journal_sha256": "b",
                "routes": [shared, route([0, 1])],
            },
        ]
        merged, proof = merge_route_sets(sources)
        self.assertEqual(len(merged), 3)
        self.assertTrue(proof["verified"])
        merged_hashes = {route_sha256(item) for item in merged}
        for source in sources:
            self.assertTrue(
                {route_sha256(item) for item in source["routes"]}
                <= merged_hashes
            )

    def test_union_refuses_instance_and_physics_mismatch(self):
        base = {
            "identity": identity(),
            "journal_sha256": "a",
            "routes": [route([0]), route([1])],
        }
        for changed in (
            identity(instance="foreign"),
            identity(g_kwh=240.0),
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    merge_route_sets([
                        base,
                        {
                            "identity": changed,
                            "journal_sha256": "b",
                            "routes": [route([0, 1])],
                        },
                    ])

    def test_union_target_result_is_no_worse_than_best_input(self):
        first = [route([0]), route([1])]
        second = [route([0, 1])]
        sources = [
            {"identity": identity(), "journal_sha256": "a", "routes": first},
            {"identity": identity(), "journal_sha256": "b", "routes": second},
        ]
        merged, _proof = merge_route_sets(sources)
        first_result = solve_target_feasibility(
            first, [0, 1], 1, timelimit=30, threads=1,
        )
        best_result = solve_target_feasibility(
            second, [0, 1], 1, timelimit=30, threads=1,
        )
        union_result = solve_target_feasibility(
            merged, [0, 1], 1, timelimit=30, threads=1,
        )
        self.assertEqual(first_result["outcome"], "INFEASIBLE")
        self.assertEqual(best_result["outcome"], "FEASIBLE")
        self.assertEqual(union_result["outcome"], "FEASIBLE")

    def test_route_hash_distinguishes_realizations(self):
        self.assertNotEqual(
            route_sha256(route([0], cost=100000.0)),
            route_sha256(route([0], cost=100001.0)),
        )


if __name__ == "__main__":
    unittest.main()
