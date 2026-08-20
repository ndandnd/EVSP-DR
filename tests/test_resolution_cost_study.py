import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import exact_pricer_expanded as exact  # noqa: E402
from audit_giro_known_columns import HORIZON_MIN, build_problem  # noqa: E402
from resolution_cost_study import (  # noqa: E402
    _failure_status,
    build_plan,
)
from summarize_resolution_cost import summarize  # noqa: E402
from utils_v2 import load_station_hourly_prices  # noqa: E402
from config import CHARGING_STATIONS  # noqa: E402


class ResolutionCostStudyTests(unittest.TestCase):
    def test_plan_has_168_hash_bound_cells_and_anchor_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_plan(Path(temporary) / "artifacts")
        self.assertEqual(len(plan["jobs"]), 168)
        self.assertEqual(
            sum(job["physics_profile"] == "p240" for job in plan["jobs"]),
            126,
        )
        self.assertEqual(
            sum(job["physics_profile"] == "p300_bridge"
                for job in plan["jobs"]),
            42,
        )
        self.assertTrue(all(job["max_iters"] == 1_000_000_000
                            for job in plan["jobs"]))
        p240 = [job for job in plan["jobs"]
                if job["physics_profile"] == "p240"]
        anchors = [job for job in p240 if job["grid_role"] == "historical_anchor"]
        self.assertEqual(len(anchors), 18)
        self.assertTrue(all(not job["commensurate"] for job in anchors))
        self.assertTrue(all(
            job["commensurate"] for job in p240
            if job["grid_role"] != "historical_anchor"
        ))
        self.assertEqual(
            {job["selection_replicate"] for job in p240}, {1, 2, 3},
        )

    def test_expanded_network_accepts_two_point_five_minute_blocks(self):
        problem = build_problem(
            exact.DATA_DIR, "Practice_Selected_1buses.csv",
            max_station_to_trip_wait_min=HORIZON_MIN,
        )
        prices = load_station_hourly_prices(
            exact.DATA_DIR / "hourly_prices_flat.csv", CHARGING_STATIONS,
        )
        network = exact.ExpandedNetwork(
            problem, prices, soc_step=15.0, block_min=2.5,
            g_kwh=240.0, charge_kw=240.0,
        )
        self.assertEqual(network.block_min, 2.5)
        self.assertEqual(network.n_blocks, 624)
        self.assertGreater(len(network.node_meta), 0)
        self.assertGreater(network.n_arcs, 0)

    @staticmethod
    def _run_exact(output, *extra):
        return exact.main([
            "--csv", "Practice_Selected_1buses.csv",
            "--prices_csv", "hourly_prices_flat.csv",
            "--soc-step", "15", "--block-min", "10",
            "--g-kwh", "240", "--charge-kw", "240",
            "--master-sense", "partition", "--initial-pool", "singletons",
            "--columns_per_iter", "30", "--checkpoint-every", "1",
            "--out", str(output), *extra,
        ])

    def test_status_contains_cost_metrics_and_distinct_stop_reasons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            certified_path = root / "certified.json"
            self._run_exact(certified_path, "--max-iters", "400")
            status = json.loads(certified_path.read_text())
            self.assertEqual(status["stop_reason"], "certified")
            self.assertTrue(status["certified"])
            for key in (
                "dag_nodes", "dag_arcs", "dag_build_wall_s",
                "iterations_to_certificate", "wall_to_certificate",
                "peak_rss_mb", "pool_columns_final",
            ):
                self.assertIsNotNone(status[key], key)
            self.assertEqual(
                len(status["iteration_metrics"]), status["iterations"],
            )
            for metric in status["iteration_metrics"]:
                self.assertTrue({
                    "master_wall_s", "pricing_wall_s", "columns_added",
                } <= set(metric))

            max_path = root / "max.json"
            self._run_exact(max_path, "--max-iters", "1")
            self.assertEqual(
                json.loads(max_path.read_text())["stop_reason"], "max_iters",
            )
            wall_path = root / "wall.json"
            self._run_exact(
                wall_path, "--max-iters", "1000000000", "--wall-limit-s", "1",
            )
            self.assertEqual(
                json.loads(wall_path.read_text())["stop_reason"], "wall_limit",
            )
            memory_path = root / "memory.json"
            self._run_exact(
                memory_path, "--max-iters", "1000000000",
                "--memory-limit-mb", "1",
            )
            self.assertEqual(
                json.loads(memory_path.read_text())["stop_reason"], "memory",
            )

    def test_failure_wrapper_preserves_estimated_dag_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "oom.json"
            job = {
                "job_key": "oom", "estimated_dag_nodes_upper": 123456,
            }
            _failure_status(job, output, -9)
            status = json.loads(output.read_text())
            self.assertEqual(status["stop_reason"], "memory")
            self.assertEqual(status["estimated_dag_nodes_upper"], 123456)

    def test_summarizer_fits_known_exponents_and_joins_mip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_plan(root / "artifacts")
            model_jobs = [
                job for job in plan["jobs"]
                if job["physics_profile"] == "p240"
                and job["scale"] in {2, 3}
            ]
            for job in model_jobs:
                output = Path(job["output"])
                output.parent.mkdir(parents=True, exist_ok=True)
                trips = job["trip_count"]
                inv_soc = 1.0 / job["soc_step"]
                inv_block = 1.0 / job["block_min"]
                nodes = (
                    1e6 * trips ** 1.2 * inv_soc ** 0.8
                    * inv_block ** 0.6
                )
                wall = (
                    100.0 * trips ** 1.5 * inv_soc ** 1.1
                    * inv_block ** 0.9
                )
                output.write_text(json.dumps({
                    "certified": True, "certified_rc_optimal": True,
                    "stop_reason": "certified",
                    "dag_nodes": nodes, "dag_arcs": nodes * 4,
                    "dag_build_wall_s": wall * 0.05,
                    "peak_rss_mb": nodes / 1e5,
                    "iterations_to_certificate": 10,
                    "wall_to_certificate": wall,
                    "pool_columns_final": 100,
                    "iteration_metrics": [{
                        "master_wall_s": wall * 0.4,
                        "pricing_wall_s": wall * 0.5,
                        "columns_added": 10,
                    }],
                    "final_lp": {
                        "route_weight": job["target_fleet"] + 0.5,
                    },
                }))
            first = model_jobs[0]
            mip_root = root / "mips"
            mip_root.mkdir()
            (mip_root / "one.json").write_text(json.dumps({
                "buses": first["target_fleet"] + 1,
                "fleet_proven": True, "mip_gap": 0.0,
                "pricer_provenance": {
                    "instance_sha256": first["instance_sha256"],
                },
                "physics": {
                    key: first[key] for key in (
                        "g_kwh", "charge_kw", "min_soc_frac",
                        "soc_step", "block_min",
                    )
                },
            }))
            long_path, _prediction, result = summarize(
                plan, root / "summary", [mip_root],
            )
            self.assertEqual(result["rows"], 168)
            self.assertEqual(result["structural_dag_nodes_upper"], 679381)
            self.assertEqual(result["node_model"]["status"], "fit")
            self.assertEqual(
                result["bridge_models"]["node_model"]["status"],
                "insufficient_data",
            )
            self.assertAlmostEqual(
                result["node_model"]["trips_exponent"], 1.2, places=6,
            )
            self.assertAlmostEqual(
                result["node_model"]["inverse_soc_step_exponent"], 0.8,
                places=6,
            )
            self.assertAlmostEqual(
                result["wall_model"]["inverse_block_min_exponent"], 0.9,
                places=6,
            )
            self.assertEqual(
                result["wall_model"]["training_ranges"]["trips"], [29, 71],
            )
            with long_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            joined = next(row for row in rows if row["job_key"] == first["job_key"])
            self.assertEqual(
                float(joined["integer_fleet"]), first["target_fleet"] + 1,
            )
            self.assertEqual(joined["grid_id"], first["grid_id"])
            self.assertEqual(len(rows), 168)


if __name__ == "__main__":
    unittest.main()
