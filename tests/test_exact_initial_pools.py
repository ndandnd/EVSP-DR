import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import exact_pricer_expanded as exact  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
