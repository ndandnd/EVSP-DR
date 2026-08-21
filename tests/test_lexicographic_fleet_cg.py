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
    @staticmethod
    def _lex_args(output, *extra):
        return ["--objective","lexicographic-fleet","--csv","Practice_Selected_1buses.csv",
                "--prices_csv","hourly_prices_flat.csv","--soc-step","15","--block-min","10",
                "--max-iters","400","--columns_per_iter","30","--master-sense","partition",
                "--initial-pool","singletons","--out",str(output),*extra]

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
            exact.main(self._lex_args(output, "--rc-eps", "1e9"))
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
            self.assertGreater(phases[1]["iterations"], 1)
            self.assertLessEqual(phases[1]["fleet_lp_lower_bound"],
                                 phases[1]["route_weight"])
            self.assertEqual(
                phases[2]["fixed_optima"]["phase_2_fleet_optimum"],
                phases[1]["route_weight"],
            )
            for key in ("instance_sha256","prices_sha256","reference_sha256","deadhead_sha256"):self.assertEqual(phases[1]["identity"][key],status["provenance"][key])
            self.assertFalse(phases[1]["identity"]["git_dirty"]);self.assertFalse(phases[1]["identity"]["strict_tariff_coverage"])
            self.assertIn("phase_iteration_log_sha256", status)
            for certificate in phases:
                payload=dict(certificate);observed=payload.pop("certificate_sha256")
                encoded=json.dumps(payload,sort_keys=True,separators=(",",":"),
                                   allow_nan=False).encode()
                self.assertEqual(observed,hashlib.sha256(encoded).hexdigest())

    def test_uncertified_phase_exports_no_fleet_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "short.json"
            exact.main(self._lex_args(output, "--max-iters", "1"))
            status=json.loads(output.read_text());phase=status["phases"][-1]
            self.assertFalse(status["all_phases_certified"])
            self.assertEqual(phase["phase"],2);self.assertNotIn("fleet_lp_lower_bound",phase)
            self.assertIsNone(status["phase_2_fleet_lp_bound"])
            self.assertIsNone(status["phase_3_charging_cost"])

    def test_wall_stop_and_no_path_are_serializable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "wall.json"
            exact.main(self._lex_args(output, "--wall-limit-s", "0"))
            phase=json.loads(output.read_text())["phases"][0]
            self.assertEqual(phase["stop_reason"],"wall_limit");self.assertFalse(phase["certified"]);self.assertEqual(phase["iterations"],0)
        args=SimpleNamespace(max_iters=1,wall_limit_s=None,columns_per_iter=1,rc_eps=1e-4)
        with tempfile.TemporaryFile(mode="w+") as handle:
            certificate=lex._run_phase(args,1,[0],{frozenset({0}):{"trips":[0],"cost":BUS_COST_KX}},
              SimpleNamespace(k_best_routes=lambda *_a,**_k:[]),{},"hash",None,0.0,None,
              lex._IterationWriter(handle),{})
        self.assertEqual(certificate["stop_reason"],"no_path");json.dumps(certificate,allow_nan=False)

    def test_short_out_alias_remains_unambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "alias.json"
            with patch.object(exact, "run_cg", return_value={"alias": True}):
                exact.main(["--csv", "unused.csv", "--o", str(output)])
            self.assertEqual(json.loads(output.read_text()), {"alias": True})


if __name__ == "__main__":
    unittest.main()
