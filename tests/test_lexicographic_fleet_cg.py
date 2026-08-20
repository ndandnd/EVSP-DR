import csv
import hashlib
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
import lexicographic_fleet_cg as lex  # noqa: E402
from config import BUS_COST_KX  # noqa: E402


class LexicographicFleetCGTests(unittest.TestCase):
    def test_default_combined_cost_path_is_bit_identical_to_golden(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run.json"
            clock = SimpleNamespace(
                time=lambda: 1000.0,
                perf_counter=lambda: 1000.0,
            )
            provenance = {
                "instance_sha256": "fixture",
                "prices_sha256": "fixture",
            }
            with (
                patch.object(exact, "time", clock),
                patch.object(exact, "_provenance", return_value=provenance),
            ):
                exact.main([
                    "--csv", "Practice_Selected_1buses.csv",
                    "--prices_csv", "hourly_prices_flat.csv",
                    "--soc-step", "15",
                    "--block-min", "10",
                    "--max-iters", "400",
                    "--columns_per_iter", "30",
                    "--master-sense", "partition",
                    "--initial-pool", "singletons",
                    "--out", str(output),
                ])

            journal = Path(str(output) + ".columns.jsonl").read_bytes()
            iteration_csv = Path(str(output) + ".iters.csv").read_bytes()
            route_hashes = [
                hashlib.sha256(line).hexdigest()
                for line in journal.splitlines(keepends=True)
            ]
            rows = list(csv.DictReader(
                iteration_csv.decode().splitlines()
            ))
            reduced_costs = [row["min_rc"] for row in rows]

            self.assertEqual(
                hashlib.sha256(journal).hexdigest(),
                "6115e8ef1a0aba4cb32dda14d6e24a58877ab0bcd06f744b9b93367e744c7508",
            )
            self.assertEqual(
                hashlib.sha256(json.dumps(
                    route_hashes, separators=(",", ":")
                ).encode()).hexdigest(),
                "a027e1efc8e5ce7600e1c072acc95c6782e1ce728d3fec35bb5961be90b41016",
            )
            self.assertEqual(
                hashlib.sha256(json.dumps(
                    reduced_costs, separators=(",", ":")
                ).encode()).hexdigest(),
                "1ac2c395c7ba5bbf87a6b5db267d92127e638b1664bac659d24f3e9eebf6096b",
            )
            self.assertEqual(
                hashlib.sha256(iteration_csv).hexdigest(),
                "c33325f94a9c7df3e5036be4b88b86215c7b1a659a80edbb6f9a75ca26452fcf",
            )
            self.assertEqual(len(route_hashes), 614)
            status = json.loads(output.read_text())
            self.assertEqual(status["stop_reason"], "certified")
            self.assertTrue(status["certified_rc_optimal"])

    def test_phase_masters_use_distinct_exact_objectives(self):
        trips = [0, 1, 2]
        routes = [
            {"trips": [0, 1], "cost": BUS_COST_KX + 2.0},
            {"trips": [1, 2], "cost": BUS_COST_KX + 4.0},
            {"trips": [0, 2], "cost": BUS_COST_KX + 6.0},
        ]
        phase_1 = lex._solve_master(trips, routes, 1)
        phase_2 = lex._solve_master(trips, routes, 2)
        phase_3 = lex._solve_master(
            trips, routes, 3, fleet_optimum=phase_2.objective,
        )
        self.assertEqual(phase_1.objective, 0.0)
        self.assertEqual(phase_1.artificial_total, 0.0)
        self.assertAlmostEqual(phase_2.objective, 1.5)
        self.assertAlmostEqual(phase_2.route_weight, 1.5)
        self.assertAlmostEqual(phase_3.route_weight, 1.5)
        self.assertAlmostEqual(phase_3.objective, 6.0)
        self.assertIsNotNone(phase_3.fleet_dual)

    def test_opt_in_run_emits_three_separate_scoped_certificates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "lex.json"
            exact.main([
                "--objective", "lexicographic-fleet",
                "--csv", "Practice_Selected_1buses.csv",
                "--prices_csv", "hourly_prices_flat.csv",
                "--soc-step", "15", "--block-min", "10",
                "--max-iters", "400", "--columns_per_iter", "30",
                "--master-sense", "partition",
                "--initial-pool", "singletons",
                "--out", str(output),
            ])
            status = json.loads(output.read_text())
            phases = status["phases"]
            self.assertEqual([row["phase"] for row in phases], [1, 2, 3])
            self.assertTrue(status["all_phases_certified"])
            self.assertEqual(
                [row["phase"] for row in phases
                 if "fleet_lp_lower_bound" in row],
                [2],
            )
            self.assertEqual(phases[1]["real_route_objective_coefficient"], 1.0)
            self.assertFalse(phases[1]["charging_terms_in_objective"])
            self.assertEqual(
                phases[2]["fixed_optima"]["phase_2_fleet_optimum"],
                phases[1]["fleet_lp_lower_bound"],
            )
            for certificate in phases:
                observed = certificate["certificate_sha256"]
                payload = dict(certificate)
                del payload["certificate_sha256"]
                self.assertEqual(
                    observed, hashlib.sha256(
                        json.dumps(
                            payload, sort_keys=True, separators=(",", ":"),
                            allow_nan=False,
                        ).encode()
                    ).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
