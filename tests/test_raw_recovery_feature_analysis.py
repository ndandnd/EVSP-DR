import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audit_giro_known_columns import DEPOT, STATIONS  # noqa: E402
from build_raw_recovery_feature_analysis import (  # noqa: E402
    _instance_metrics,
    _normalize_results,
    descriptive_associations,
    feature_distribution,
    publish,
)


class RawRecoveryFeatureAnalysisTests(unittest.TestCase):
    def test_instance_metric_definitions(self):
        station = STATIONS[0]
        problem = SimpleNamespace(
            trips=(0, 1, 2),
            start_min={0: 0.0, 1: 20.0, 2: 50.0},
            end_min={0: 10.0, 1: 30.0, 2: 60.0},
            trip_energy={0: 5.0, 1: 5.0, 2: 5.0},
            adjacency={
                DEPOT: [(0, 2.0, 1.0, "depot_trip")],
                0: [
                    (1, 5.0, 2.0, "trip_trip"),
                    (station, 3.0, 1.0, "trip_station"),
                ],
                1: [
                    (2, 10.0, 3.0, "trip_trip"),
                    (station, 2.0, 1.0, "trip_station"),
                ],
                2: [(DEPOT, 4.0, 2.0, "trip_depot")],
                station: [(2, 2.0, 1.0, "station_trip")],
            },
        )
        metrics = _instance_metrics(
            problem, [{"duty_id": "d", "trips": [0, 1, 2]}],
        )
        self.assertAlmostEqual(metrics["deadhead_density"], 2 / 3)
        self.assertEqual(metrics["service_kwh_per_trip"], 5.0)
        self.assertEqual(metrics["service_kwh_per_duty"], 15.0)
        self.assertEqual(metrics["layover_slack_min"], 5.0)
        self.assertEqual(metrics["layover_slack_median"], 7.5)
        self.assertEqual(metrics["layover_slack_max"], 10.0)
        self.assertAlmostEqual(
            metrics["station_reachability_fraction"], 2 / 3,
        )
        self.assertAlmostEqual(
            metrics["station_only_bridge_fraction"], 1 / 3,
        )
        distribution = feature_distribution(pd.DataFrame([metrics]))
        slack = distribution[
            distribution["feature"] == "layover_slack_median"
        ].iloc[0]
        self.assertEqual(slack["n_instances"], 1)
        self.assertEqual(slack["median"], 7.5)

    def test_recovery_requires_a_proven_finite_pool_optimum(self):
        instances = pd.DataFrame([
            {
                "cell_id": "k02_s1", "instance_file_sha256": "a",
                "target_fleet": 2, "trip_count": 10, "duty_count": 2,
                "deadhead_density": 0.1,
            },
            {
                "cell_id": "k02_s2", "instance_file_sha256": "b",
                "target_fleet": 2, "trip_count": 20, "duty_count": 2,
                "deadhead_density": 0.2,
            },
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.csv"
            pd.DataFrame([
                {
                    "cell_id": "k02_s1_c1", "arm": "RAW",
                    "buses": 2, "fleet_proven": True,
                },
                {
                    "cell_id": "k02_s2_c1", "arm": "RAW",
                    "buses": 4, "fleet_proven": True,
                },
                {
                    "cell_id": "k02_s1_c2", "arm": "RAW",
                    "buses": 2, "fleet_proven": False,
                },
                {
                    "cell_id": "k02_s1_c1", "arm": "KNOWN-PARTITION",
                    "buses": 2, "fleet_proven": True,
                },
            ]).to_csv(path, index=False)
            cells = _normalize_results((path,), instances)
        self.assertEqual(
            cells["recovery_status"].tolist(),
            ["recovered", "missed", "unknown"],
        )
        associations = descriptive_associations(instances, cells)
        trip = associations[associations["feature"] == "trip_count"].iloc[0]
        self.assertEqual(trip["n_cells"], 2)
        self.assertEqual(trip["recovered_cells"], 1)
        self.assertEqual(trip["missed_cells"], 1)

    def test_missing_raw_results_is_explicit_not_an_empty_claim(self):
        features = pd.DataFrame([{
            "cell_id": "k02_s1", "instance_file_sha256": "a",
            "target_fleet": 2, "trip_count": 10, "duty_count": 2,
        }])
        with tempfile.TemporaryDirectory() as tmp, patch(
            "build_raw_recovery_feature_analysis.build_instance_features",
            return_value=features,
        ):
            manifest = Path(tmp) / "manifest.csv"
            preflight = Path(tmp) / "preflight.json"
            manifest.write_text("x\n")
            preflight.write_text("{}\n")
            out = Path(tmp) / "out"
            summary = publish(
                out, manifest=manifest, preflight=preflight,
            )
            self.assertEqual(
                summary["analysis_status"],
                "not_estimable_no_auditable_raw_results",
            )
            self.assertEqual(summary["raw_result_rows"], 0)
            self.assertIn(
                "No auditable normalized RAW integer-result rows",
                (out / "README.md").read_text(),
            )
            with (out / "raw_cell_outcomes.csv").open(newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])

    def test_synthetic_random_manifest_is_rejected(self):
        from build_raw_recovery_feature_analysis import build_instance_features

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("relative_path", "generator_family"),
                )
                writer.writeheader()
                writer.writerow({
                    "relative_path": "Practice_SyntheticRandom_x.csv",
                    "generator_family":
                        "generate_random_goal1_instances_v1_seed20260821",
                })
            with self.assertRaisesRegex(ValueError, "SyntheticRandom"):
                build_instance_features(manifest, Path(tmp) / "unused.json")


if __name__ == "__main__":
    unittest.main()
