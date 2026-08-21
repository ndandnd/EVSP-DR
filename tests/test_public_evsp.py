import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arcflow_oracle import build_network_from_problem  # noqa: E402
from convert_utrecht_evsp import convert_instance, load_problem  # noqa: E402
from generate_public_synthetic_evsp import generate_family  # noqa: E402
from run_public_synthetic_evsp import (  # noqa: E402
    solve_grid,
    solve_pair_pool,
)


def trip_row(line, number, start, end):
    fields = [
        "T", line, str(number), "0", "0", "HUB", str(start), str(end),
        "HUB", "HUB", "HUB", "E", "", "", "0", "2", "12345",
        "10.0", "", "",
    ]
    return ";".join(fields)


class PublicEVSPTests(unittest.TestCase):
    def write_utrecht_fixture(self, folder: Path):
        (folder / "parameters.txt").write_text(
            "U;g;100;0;0;0;0;0;J;2;1;0;60\n"
            "G;DEPOT\n"
            "E;HUB;99;0;1;1\n"
        )
        (folder / "trips.txt").write_text(
            trip_row("L", 1, 30, 50) + "\n"
            + trip_row("L", 2, 80, 100) + "\n"
        )
        (folder / "dhd.txt").write_text(
            "G;0;0;99\n"
            "G;1;100;199\n"
            "D;DEPOT-HUB;5;8;5;5;2\n"
            "D;HUB-DEPOT;5;8;5;5;2\n"
        )

    def test_utrecht_converter_preserves_profiles_and_energy(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            self.write_utrecht_fixture(source)
            payload = convert_instance(
                source, name="fixture", upstream_commit="abc123"
            )
            self.assertEqual(payload["vehicle"]["battery_kwh"], 60)
            self.assertEqual(payload["trips"][0]["energy_kwh"], 20)
            problem = load_problem(payload)
            self.assertEqual(problem.transition_at("DEPOT", 0, 10), (5.0, 4.0))
            self.assertEqual(problem.transition_at("DEPOT", 0, 120), (8.0, 4.0))
            prices = {"HUB": {hour: 0.0 for hour in range(4)}}
            network = build_network_from_problem(
                "fixture", problem, prices, soc_step=5, block_min=5,
                g_kwh=60, charge_kw=60,
            )
            self.assertEqual(network.network.stations, ("HUB",))
            self.assertEqual(network.network.depot, "DEPOT")
            self.assertEqual(network.network.horizon, 200)

    def test_public_synthetic_family_is_deterministic_and_structural(self):
        first_problem, first_pool = generate_family(20260821)
        second_problem, second_pool = generate_family(20260821)
        self.assertEqual(first_problem, second_problem)
        self.assertEqual(first_pool, second_pool)
        fine = solve_grid(
            first_problem, **first_problem["features"]["fine_grid"]
        )
        coarse = solve_grid(
            first_problem, **first_problem["features"]["coarse_grid"]
        )
        pool = solve_pair_pool(first_pool, len(first_problem["trips"]))
        self.assertEqual(round(fine["lp"]["vehicles"], 8), 2)
        self.assertEqual(round(fine["integer"]["vehicles"]), 2)
        self.assertEqual(round(coarse["integer"]["vehicles"]), 3)
        self.assertEqual(round(pool["fleet"]), 3)

    def test_converted_json_round_trip(self):
        problem, _pool = generate_family(7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "problem.json"
            path.write_text(json.dumps(problem))
            loaded = load_problem(path)
            self.assertEqual(len(loaded.trips), 6)
            self.assertEqual(loaded.depot, "DEPOT")


if __name__ == "__main__":
    unittest.main()
