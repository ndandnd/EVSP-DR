import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import exact_pricer_expanded as exact  # noqa: E402
from audit_giro_known_columns import HORIZON_MIN, build_problem  # noqa: E402
from resolution_cost_study import (  # noqa: E402
    _failure_status,
    build_plan,
    run_job,
)
from summarize_resolution_cost import summarize  # noqa: E402
from utils_v2 import load_station_hourly_prices  # noqa: E402
from config import CHARGING_STATIONS  # noqa: E402
import resolution_cost_arcflow as arc_cost  # noqa: E402
import resolution_cost_pool_mip as pool_cost  # noqa: E402
from tests.test_arcflow_oracle import fake_data  # noqa: E402


class ResolutionCostStudyTests(unittest.TestCase):
    def test_plan_has_168_hash_bound_cells_and_anchor_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_plan(Path(temporary) / "artifacts")
        self.assertEqual(len(plan["jobs"]), 336)
        self.assertEqual(plan["scientific_cells"], 168)
        self.assertEqual(
            {job["method_arm"] for job in plan["jobs"]},
            {"exact_cg", "arc_flow"},
        )
        self.assertEqual(
            sum(job["physics_profile"] == "p240"
                and job["method_arm"] == "exact_cg"
                for job in plan["jobs"]),
            126,
        )
        self.assertEqual(
            sum(job["physics_profile"] == "p300_bridge"
                and job["method_arm"] == "exact_cg"
                for job in plan["jobs"]),
            42,
        )
        self.assertTrue(all(job["max_iters"] == 1_000_000_000
                            for job in plan["jobs"]))
        p240 = [job for job in plan["jobs"]
                if job["physics_profile"] == "p240"
                and job["method_arm"] == "exact_cg"]
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
        pairs = {}
        for job in plan["jobs"]:
            pairs.setdefault(job["paired_cell_key"], set()).add(
                job["method_arm"]
            )
        self.assertEqual(len(pairs), 168)
        self.assertTrue(all(
            arms == {"exact_cg", "arc_flow"} for arms in pairs.values()
        ))

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

    def test_arcflow_parent_timeout_is_an_honest_wall_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_plan(Path(temporary) / "artifacts")
            job = next(
                row for row in plan["jobs"]
                if row["method_arm"] == "arc_flow"
            )
            with patch(
                "resolution_cost_study.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["arc"], 1),
            ):
                self.assertEqual(run_job(job, budget_override_s=1), 0)
            status = json.loads(Path(job["output"]).read_text())
            self.assertEqual(status["stop_reason"], "wall_limit")
            self.assertEqual(
                status["estimated_dag_nodes_upper"],
                job["estimated_dag_nodes_upper"],
            )

    def test_arcflow_arm_records_lp_mip_and_sparse_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fake.csv").write_text("fixture")
            output = root / "arc.json"
            args = SimpleNamespace(
                out=output, resume=False, csv="fake.csv",
                prices_csv="prices.csv", data_dir=root,
                g_kwh=300.0, charge_kw=300.0, reserve_kwh=0.0,
                soc_step=15.0, block_min=10,
                memory_limit_mb=None, time_limit_s=10.0,
                lp_time_limit_s=5.0, lp_time_fraction=0.4,
                mip_rel_gap=0.0,
            )
            with (
                patch.object(arc_cost, "build_network", return_value=fake_data()),
                patch.object(
                    arc_cost, "gate_g4",
                    return_value=({"passed": True, "routes": 1}, [{}]),
                ),
            ):
                result = arc_cost.run(args)
            self.assertEqual(result["lp_bound"], 1.0)
            self.assertEqual(result["integer_result"], 1.0)
            self.assertTrue(result["integer_proven"])
            self.assertEqual(result["variables"], 2)
            self.assertEqual(result["constraints"], 3)
            self.assertEqual(result["nonzeros"], 4)

    def test_cg_arm_records_integer_pool_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            status_path = root / "cg.json"
            journal = root / "cg.json.columns.jsonl"
            journal.write_text("\n".join(json.dumps(route) for route in (
                {"trips": [0], "cost": 1.0},
                {"trips": [1], "cost": 1.0},
                {"trips": [0, 1], "cost": 1.0},
            )) + "\n")
            status_path.write_text(json.dumps({
                "trip_ids": [0, 1], "columns_journal": str(journal),
                "certified": True,
            }))
            args = SimpleNamespace(
                cg_status=status_path, time_limit_s=10.0,
                mip_rel_gap=0.0, out=root / "mip.json",
            )
            with patch.object(
                pool_cost, "validate_final_selected_routes",
                return_value=None,
            ):
                result = pool_cost.run(args)
            self.assertEqual(result["integer_result"], 1)
            self.assertTrue(result["integer_proven"])
            self.assertEqual(result["variables"], 3)
            self.assertEqual(result["constraints"], 2)
            self.assertEqual(result["nonzeros"], 4)

    def test_summarizer_fits_known_exponents_and_joins_mip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = build_plan(root / "artifacts")
            model_jobs = [
                job for job in plan["jobs"]
                if job["physics_profile"] == "p240"
                and job["scale"] in {2, 3}
                and job["method_arm"] == "exact_cg"
            ]
            by_pair = {
                job["paired_cell_key"]: job for job in plan["jobs"]
                if job["method_arm"] == "arc_flow"
            }
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
                arc_job = by_pair[job["paired_cell_key"]]
                arc_output = Path(arc_job["output"])
                arc_output.parent.mkdir(parents=True, exist_ok=True)
                variables = (
                    2e6 * trips ** 1.4 * inv_soc ** 1.0
                    * inv_block ** 1.2
                )
                arc_output.write_text(json.dumps({
                    "integer_proven": job is not model_jobs[0],
                    "certified": job is not model_jobs[0],
                    "stop_reason": (
                        "wall_limit" if job is model_jobs[0] else "certified"
                    ),
                    "dag_nodes": nodes, "dag_arcs": nodes * 4,
                    "dag_build_wall_s": wall * 0.05,
                    "active_arcs": variables,
                    "variables": variables,
                    "constraints": variables * 0.7,
                    "nonzeros": variables * 2.4,
                    "model_build_wall_s": wall * 0.1,
                    "lp_bound": job["target_fleet"] + 0.5,
                    "integer_result": job["target_fleet"] + 1,
                    "lp_wall_s": wall * 0.3,
                    "mip_wall_s": wall * 0.6,
                    "wall_s": wall,
                    "peak_rss_mb": variables / 1e5,
                    "lp": {"status": "optimal", "solve_s": wall * 0.3},
                    "mip": {"status": "limit_reached", "mip_gap": 0.1},
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
            self.assertEqual(result["rows"], 336)
            self.assertEqual(result["structural_dag_nodes_upper"], 679381)
            self.assertEqual(result["node_model"]["status"], "fit")
            self.assertEqual(
                result["bridge_models"]["node_model"]["status"],
                "insufficient_data",
            )
            self.assertAlmostEqual(
                result["arcflow_models"]["variables"]["trips_exponent"],
                1.4, places=6,
            )
            self.assertTrue(
                result["paired_diagnostics"][
                    "lp_equality_holds_everywhere_checked"
                ]
            )
            self.assertEqual(
                len(result["paired_diagnostics"][
                    "arcflow_intractable_while_cg_certified"
                ]), 1,
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
            self.assertEqual(len(rows), 336)


if __name__ == "__main__":
    unittest.main()
