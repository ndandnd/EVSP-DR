"""Re-realization must repair GIRO seeds into injection-valid routes.

Uses the tracked two-duty instance (13301+13302): its raw GIRO seed is
exactly the route class that fails injection (Hastus minute-rounded
recharges), and its duration-repaired variant demonstrated the band-aid
ceiling ("arrives after trip start"). Re-realization must produce routes
that pass validate_injected_route under both the convention physics
(300/300/0) and the realism physics (240/220/20%), and must shift charging
out of the peak window under a peaked tariff.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_giro_known_columns import HORIZON_MIN, build_problem  # noqa: E402
from config import CHARGING_STATIONS  # noqa: E402
from make_giro_seed_routes import build_seeds, load_master  # noqa: E402
from rerealize_routes import _arc_map, rerealize_route  # noqa: E402
from run_exact_pool_mip import validate_injected_route  # noqa: E402
from utils_v2 import load_station_hourly_prices  # noqa: E402

INSTANCE = "Practice_Custom_TwoDuty_13301_13302.csv"
DATA_DIR = REPO_ROOT / "data"


class TestRerealizeRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as tmp:
            seed_path = Path(tmp) / "seed.json"
            build_seeds(load_master(), Path(INSTANCE), seed_path)
            with open(seed_path) as fh:
                cls.seed = json.load(fh)
        cls.problem = build_problem(DATA_DIR, INSTANCE,
                                    max_station_to_trip_wait_min=HORIZON_MIN)
        cls.arc = _arc_map(cls.problem)
        cls.trip_seqs = [[n for n in r["route"] if isinstance(n, int)]
                         for r in cls.seed["routes"]]

    def rerealize_all(self, prices_name, g_kwh, charge_kw, reserve_frac):
        prices = load_station_hourly_prices(DATA_DIR / prices_name,
                                            CHARGING_STATIONS)
        records = []
        for seq in self.trip_seqs:
            record, cost, reason = rerealize_route(
                seq, self.problem, self.arc, prices, g_kwh, charge_kw,
                reserve_frac * g_kwh)
            self.assertIsNone(reason, f"route infeasible: {reason}")
            verdict = validate_injected_route(
                self.problem,
                {"route_nodes": record["route"],
                 "charging_stops": record["charging_stops"]},
                g_kwh, charge_kw, reserve_frac * g_kwh, HORIZON_MIN)
            self.assertIsNone(verdict, f"validator rejected: {verdict}")
            records.append((record, cost))
        return records

    def test_convention_physics_valid_by_construction(self):
        records = self.rerealize_all("hourly_prices_flat.csv", 300.0, 300.0, 0.0)
        self.assertEqual(len(records), 2)
        covered = sorted(n for rec, _ in records
                         for n in rec["route"] if isinstance(n, int))
        self.assertEqual(covered, list(range(len(self.problem.trips))))

    def test_realism_physics_with_reserve(self):
        self.rerealize_all("hourly_prices_transdev_sek.csv", 240.0, 220.0, 0.2)

    def test_peak_tariff_shifts_charging_out_of_window(self):
        def peak_kwh(records, hours):
            total = 0.0
            for rec, _ in records:
                stops = rec["charging_stops"]
                for cst, kwh in zip(stops["cst"], stops["kwh"]):
                    if int(cst // 60) in hours:
                        total += kwh
            return total

        flat = self.rerealize_all("hourly_prices_flat.csv", 300.0, 300.0, 0.0)
        peaked = self.rerealize_all("hourly_prices_single_peak_12.csv",
                                    300.0, 300.0, 0.0)
        window = {11, 12, 13}
        self.assertLess(peak_kwh(peaked, window),
                        peak_kwh(flat, window) * 0.5,
                        "peaked tariff should shift charging out of the peak")


if __name__ == "__main__":
    unittest.main()
