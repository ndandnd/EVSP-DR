import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import exact_pricer_expanded as exact  # noqa: E402
import exact_initial_pools as pools  # noqa: E402
from exact_initial_pools import (  # noqa: E402
    build_heuristic_initial_pool,
    pool_sha256,
)


class ExactInitialPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = exact.build_problem(
            exact.DATA_DIR,
            "Practice_Selected_1buses.csv",
            max_station_to_trip_wait_min=exact.HORIZON_MIN,
        )
        cls.prices = exact.load_station_hourly_prices(
            exact.DATA_DIR / "hourly_prices_flat.csv",
            exact.CHARGING_STATIONS,
        )

    def _build(self, mode):
        return build_heuristic_initial_pool(
            self.problem,
            self.prices,
            mode=mode,
            depot=exact.DEPOT,
            stations=exact.STATIONS,
            g_kwh=300.0,
            charge_kw=300.0,
            reserve_kwh=0.0,
            soc_step=15.0,
            block_min=10,
            tariff_sha256="a" * 64,
            instance_sha256="b" * 64,
        )

    def test_matching_and_greedy_are_deterministic_exact_partitions(self):
        hashes = {}
        for mode in ("matching", "greedy"):
            records, provenance = self._build(mode)
            repeated, repeated_provenance = self._build(mode)
            covered = [trip for route in records for trip in route["trips"]]
            self.assertEqual(sorted(covered), sorted(self.problem.trips))
            self.assertEqual(len(covered), len(set(covered)))
            self.assertTrue(all(
                route["origin"] == f"exact_{mode}_initial_seed"
                for route in records
            ))
            self.assertFalse(provenance["uses_giro_partition"])
            self.assertEqual(provenance["generated_pool_sha256"],
                             pool_sha256(records))
            self.assertEqual(records, repeated)
            self.assertEqual(provenance, repeated_provenance)
            hashes[mode] = provenance["generated_pool_sha256"]
        self.assertNotEqual(hashes["matching"], hashes["greedy"])

    def test_matching_internal_split_provenance_is_retained(self):
        matching = {
            "relaxed_minimum_path_count": 1,
            "resource_feasible_path_count": 2,
            "resource_repair_mode": "contiguous_split",
            "contiguous_splits_added": 1,
        }
        proposed = [
            {"route": ["D", 0, "D"], "_matching_init": matching},
            {"route": ["D", 1, "D"], "_matching_init": matching},
        ]
        problem = SimpleNamespace(
            trips=(0, 1), adjacency={}, start_min={0: 0, 1: 1},
            end_min={0: 1, 1: 2}, trip_energy={0: 1, 1: 1},
        )
        with patch.object(
            pools, "build_matching_initial_routes", return_value=proposed,
        ):
            sequences, _generator, details = pools._trip_sequences(
                problem, "matching", depot="D", stations=(),
                g_kwh=300, charge_kw=300, reserve_kwh=0,
                soc_step=15, block_min=10,
            )
        self.assertEqual(sequences, [(0,), (1,)])
        self.assertEqual(details["heuristic_relaxed_route_count"], 1)
        self.assertEqual(details["heuristic_realized_route_count"], 2)
        self.assertEqual(details["heuristic_contiguous_splits_added"], 1)

    def test_certified_optimum_is_invariant_to_initial_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            observed = {}
            for mode in ("singletons", "matching", "greedy"):
                output = Path(tmp) / f"{mode}.json"
                exact.main([
                    "--csv", "Practice_Selected_1buses.csv",
                    "--prices_csv", "hourly_prices_flat.csv",
                    "--soc-step", "15", "--block-min", "10",
                    "--max-iters", "400", "--columns_per_iter", "30",
                    "--master-sense", "partition",
                    "--initial-pool", mode,
                    "--out", str(output),
                ])
                payload = json.loads(output.read_text())
                self.assertTrue(payload["certified_rc_optimal"])
                self.assertEqual(payload["initial_pool"], mode)
                self.assertEqual(
                    payload["initial_pool_sha256"],
                    payload["initial_pool_provenance"][
                        "generated_pool_sha256"
                    ],
                )
                observed[mode] = payload["final"]["lp_obj"]
            self.assertAlmostEqual(
                observed["singletons"], observed["matching"], places=6,
            )
            self.assertAlmostEqual(
                observed["singletons"], observed["greedy"], places=6,
            )

    def test_model_warm_pool_cannot_be_mixed_with_external_seed(self):
        with self.assertRaises(SystemExit):
            exact.main([
                "--csv", "unused.csv",
                "--initial-pool", "matching",
                "--validated-seed-routes", "external.json",
                "--augmentation-label", "GIRO-AUGMENTED",
            ])

    def test_lexicographic_hash_matches_final_journaled_seed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("matching", "greedy"):
                output = Path(tmp) / f"{mode}.json"
                exact.main([
                    "--objective", "lexicographic-fleet",
                    "--csv", "Practice_Selected_1buses.csv",
                    "--prices_csv", "hourly_prices_flat.csv",
                    "--initial-pool", mode,
                    "--max-iters", "0",
                    "--out", str(output),
                ])
                payload = json.loads(output.read_text())
                records = [
                    json.loads(line)
                    for line in Path(
                        str(output) + ".columns.jsonl"
                    ).read_text().splitlines()
                ]
                self.assertTrue(all(
                    record["found_lexicographic_phase"] == 0
                    for record in records
                ))
                self.assertEqual(
                    payload["initial_pool_sha256"],
                    pool_sha256(records),
                )

    def test_diversification_round_count_scales_and_is_instrumented(self):
        with tempfile.TemporaryDirectory() as tmp:
            observed = {}
            for rounds in (3, 10):
                output = Path(tmp) / f"diversify_{rounds}.json"
                exact.main([
                    "--csv", "Practice_Selected_1buses.csv",
                    "--prices_csv", "hourly_prices_flat.csv",
                    "--max-iters", "1",
                    "--columns_per_iter", "5",
                    "--diversify-rounds", str(rounds),
                    "--out", str(output),
                ])
                detail = json.loads(output.read_text())["diversification"]
                self.assertEqual(detail["requested_rounds"], rounds)
                self.assertEqual(detail["executed_rounds"], rounds)
                self.assertEqual(len(detail["rounds"]), rounds)
                observed[rounds] = detail
            self.assertEqual(
                observed[10]["rounds"][:3],
                observed[3]["rounds"],
            )


if __name__ == "__main__":
    unittest.main()
