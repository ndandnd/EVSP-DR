import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "scripts" / "event_uniform_envelope"


class CgAccelerationToolingTests(unittest.TestCase):
    def test_prepare_freezes_six_replicates_at_k13_and_k20(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            input_repo = base / "input"
            solver_repo = base / "solver"
            instances = input_repo / "data" / "scale_ladder" / "instances"
            instances.mkdir(parents=True)
            manifest_rows = []
            for scale in (8, 13, 20):
                for replicate in range(1, 7):
                    relative = f"data/scale_ladder/instances/k{scale}_{replicate}.csv"
                    path = input_repo / relative
                    path.write_text(f"scale,replicate\n{scale},{replicate}\n")
                    manifest_rows.append({
                        "scale": scale,
                        "selection_replicate": replicate,
                        "trip_count": scale * 10 + replicate,
                        "relative_path": relative,
                        "instance_file_sha256": hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest(),
                    })
            manifest = instances / (
                "scale_ladder_instance_manifest_6sel_seed20260803.csv"
            )
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(manifest_rows[0])
                )
                writer.writeheader()
                writer.writerows(manifest_rows)
            for relative in (
                "src/exact_pricer_expanded.py",
                "src/event_pricer_network.py",
                "src/exact_cg_telemetry.py",
            ):
                path = solver_repo / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)
            root = base / "campaign"
            subprocess.run([
                sys.executable,
                str(TOOLS / "prepare_cg_acceleration.py"),
                "--input-repo", str(input_repo),
                "--solver-repo", str(solver_repo),
                "--root", str(root),
                "--input-commit", "a" * 40,
                "--solver-commit", "b" * 40,
                "--wrapper-commit", "c" * 40,
            ], check=True)
            with (root / "matrix.tsv").open() as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(len(rows), 12)
            self.assertEqual({int(row[2]) for row in rows}, {13, 20})
            plan = json.loads((root / "execution_plan.json").read_text())
            self.assertEqual(len(plan["arms"]), 6)
            self.assertEqual(
                {arm["columns_per_iter"] for arm in plan["arms"]},
                {30, 60, 120, 200},
            )

    def test_audit_writes_phase_and_slurm_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cg" / "b030_reduced").mkdir(parents=True)
            (root / "network_cache").mkdir()
            (root / "matrix.tsv").write_text(
                "0\tk13_s1\t13\t1\t100\t/input.csv\tsha\t"
                "event_2p5_event5\t2.5\t5\t43200\n"
            )
            (root / "execution_plan.json").write_text(json.dumps({
                "arms": [{
                    "arm": "b030_reduced",
                    "columns_per_iter": 30,
                    "selection": "reduced_cost",
                    "diversity_weight": 0.0,
                }]
            }))
            (root / "jobs.tsv").write_text(
                "stage\tarm\tarray_job_id\tindices\n"
                "cg\tb030_reduced\t123\t0\n"
            )
            cache = root / "network_cache" / (
                "M__k13_s1__event_2p5_event5.pkl"
            )
            Path(str(cache) + ".manifest.json").write_text(json.dumps({
                "original_build_s": 60.0
            }))
            status = root / "cg" / "b030_reduced" / (
                "M__k13_s1__event_2p5_event5.json"
            )
            status.write_text(json.dumps({
                "certified_rc_optimal": True,
                "stop_reason": "certified",
                "wall_s": 120.0,
                "iterations": 10,
                "columns": 250,
                "peak_rss_mb": 500,
                "network_metrics": {"cache_hit": True, "cache_io_s": 2.0},
                "final": {
                    "route_weight": 13.0,
                    "lp_obj": 1300000.0,
                    "artificials": 0.0,
                    "min_rc": 0.0,
                },
            }))
            Path(str(status) + ".phase-telemetry.jsonl").write_text(
                json.dumps({
                    "record_type": "phase",
                    "phase": "master_attempt",
                    "duration_s": 70.0,
                }) + "\n" + json.dumps({
                    "record_type": "phase",
                    "phase": "pricing_extra_columns",
                    "duration_s": 30.0,
                }) + "\n"
            )
            sacct = root / "sacct.psv"
            sacct.write_text(
                "123_0|123_0|ca03_30r|COMPLETED|0:0|00:02:00|"
                "12:15:00|00:01:50|1|500M|1G|96Gn|node01\n"
            )
            subprocess.run([
                sys.executable,
                str(TOOLS / "audit_cg_acceleration.py"),
                str(root), "--sacct", str(sacct),
            ], check=True)
            with (root / "cg_acceleration_rows.csv").open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["outcome"], "certified")
            self.assertEqual(row["slurm_state"], "COMPLETED")
            self.assertEqual(float(row["end_to_end_s"]), 180.0)
            self.assertEqual(float(row["pricing_batch_s"]), 30.0)
            self.assertEqual(float(row["master_lp_s"]), 70.0)

    def test_cg_workers_are_requeue_safe_and_mips_remain_nonpreemptible(self):
        worker = (TOOLS / "cg_acceleration.sub").read_text()
        launcher = (TOOLS / "submit_cg_acceleration_ladder.sh").read_text()
        medium = (TOOLS / "medium_event_cg.sub").read_text()
        self.assertIn("--requeue", launcher)
        self.assertIn("--open-mode=append", launcher)
        self.assertIn("aftercorr:$CACHE_JOB", launcher)
        self.assertIn("--resume", worker)
        self.assertIn("SLURM_RESTART_COUNT", worker)
        self.assertIn("--event-network-cache-mode require", worker)
        self.assertIn("--resume", medium)
        mip = (REPO / "src" / "submit_exact_pool_mip.sub").read_text()
        self.assertIn("#SBATCH --partition=scaglione", mip)
        self.assertIn("#SBATCH --no-requeue", mip)

    def test_shell_scripts_parse(self):
        for relative in (
            "event_network_cache.sub",
            "cg_acceleration.sub",
            "submit_cg_acceleration_ladder.sh",
            "audit_cg_acceleration.sh",
            "medium_event_cg.sub",
            "recover_cg_acceleration_cache_timeout.sh",
            "submit_small_threshold_preempted_recovery.sh",
            "inspect_active_event_campaigns.sh",
            "submit_small_threshold_resume48h.sh",
            "audit_small_threshold_resume48h.sh",
        ):
            subprocess.run(
                ["/bin/bash", "-n", str(TOOLS / relative)], check=True
            )

    def test_missing_preempted_selector_is_strict_by_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "medium_event_summary.csv"
            rows = [
                {"index": "30", "scale": "9", "slurm_state": "PREEMPTED",
                 "result_present": "False"},
                {"index": "31", "scale": "9", "slurm_state": "COMPLETED",
                 "result_present": "True"},
            ]
            with summary.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run([
                sys.executable,
                str(TOOLS / "select_missing_preempted_event_indices.py"),
                "--root", str(root), "--expected-scale", "9",
            ], check=True, capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), "30")

    def test_frontier_summary_preserves_preemption_without_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "medium_event_summary.csv"
            rows = [{
                "index": "30", "cell_id": "k9_p1", "scale": "9",
                "slurm_state": "PREEMPTED", "result_present": "False",
                "configuration_match": "False",
            }]
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            subprocess.run([
                sys.executable, str(TOOLS / "summarize_cg_frontier.py"),
                str(root),
            ], check=True, capture_output=True, text=True)
            with (root / "cg_frontier_rows.csv").open() as handle:
                result = next(csv.DictReader(handle))
            self.assertEqual(result["outcome"], "preempted")

    def test_live_inspector_prioritizes_active_and_scheduler_states(self):
        import importlib.util
        path = TOOLS / "inspect_active_event_campaigns.py"
        spec = importlib.util.spec_from_file_location("live_inspector", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.path.insert(0, str(TOOLS))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        self.assertEqual(
            module.classify({"certified_rc_optimal": True}, "RUNNING"),
            "running",
        )
        self.assertEqual(module.classify(None, "PREEMPTED"),
                         "execution_preempted")
        self.assertEqual(
            module.classify({"certified_rc_optimal": True}, "COMPLETED"),
            "certified",
        )

    def test_small_threshold_long_resume_is_separate_and_requeue_safe(self):
        launcher = (TOOLS / "submit_small_threshold_resume48h.sh").read_text()
        worker = (TOOLS / "cg_resume_extended.sub").read_text()
        self.assertIn("--expected-cells 23", launcher)
        self.assertIn("CHILD_CAP=172800", launcher)
        self.assertIn("-t 1-12:30:00 --requeue", launcher)
        self.assertIn("cg_resume48h_20260904", launcher)
        self.assertIn("SLURM_RESTART_COUNT", worker)
        self.assertIn("--resume", worker)

    def test_acceleration_recovery_overrides_only_selected_index(self):
        import importlib.util
        path = TOOLS / "audit_cg_acceleration.py"
        spec = importlib.util.spec_from_file_location("acceleration_audit", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.path.insert(0, str(TOOLS))
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs.tsv").write_text(
                "stage\tarm\tarray_job_id\tindices\n"
                "cg\tb030_reduced\t100\t0-11\n"
            )
            (root / "jobs_recovery_200.tsv").write_text(
                "stage\tarm\tarray_job_id\tindices\n"
                "cg\tb030_reduced\t200\t9\n"
            )
            jobs = module.arm_jobs(root)
            self.assertEqual(jobs[("b030_reduced", 1)], "100")
            self.assertEqual(jobs[("b030_reduced", 9)], "200")


if __name__ == "__main__":
    unittest.main()
