import csv
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cross_generation_schema as schemas  # noqa: E402
from build_cross_generation_evidence import (  # noqa: E402
    _scale_progress_rows,
    build as _build,
)
from executable_identity import sha256_file  # noqa: E402


def build(manifest, output, **kwargs):
    kwargs.setdefault(
        "approved_manifest_sha256",
        hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
    )
    git_executable = Path(shutil.which("git")).resolve()
    kwargs.setdefault("git_executable", git_executable)
    kwargs.setdefault(
        "expected_git_sha256", sha256_file(git_executable)
    )
    return _build(manifest, output, **kwargs)


class CrossGenerationEvidenceTests(unittest.TestCase):
    @staticmethod
    def _sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_current_csv(self, path):
        fields = sorted(schemas.CURRENT_HEURISTIC_REQUIRED)
        values = {field: "0" for field in fields}
        values.update({
            field: "" for field in fields
            if field.startswith(("Tier2_", "Tier3_"))
        })
        values.update({
            "Iteration": "1",
            "Master_Obj_Before_Add": "800000",
            "Master_Improvement_Before_Add": "1000",
            "Master_Time_s": "2",
            "LP_Route_Weight_Before_Add": "8",
            "Artificial_Trips_Before_Add": "0",
            "Artificial_Total_Before_Add": "0",
            "Pricing_Time_s": "8",
            "Cumulative_Master_Time_s": "2",
            "Cumulative_Pricing_Time_s": "8",
            "Cols_Added": "4",
            "Best_RC": "-1",
            "Timed_Out": "False",
            "Deepest_Tier_Hit_Timelimit": "False",
            "Pricing_Labels_Used": "100",
            "Pricing_Label_Cap_Configured": "1000",
            "Pricing_Completed_Routes": "10",
            "Pricing_Negative_Completed": "4",
            "Pricing_Label_Cap_Evictions": "0",
            "Pricing_Exhaustive_Deepest_Tier": "True",
            "Pricing_Queue_Order": "best_bound",
            "Pricing_Output_Selection": "incidence",
            "Pricing_Dominance_Mode": "exact",
            "Pricing_Eligible_Negative_Incidences": "4",
            "Pricing_Returned_Trip_Count_Min": "1",
            "Pricing_Returned_Trip_Count_Mean": "2",
            "Pricing_Returned_Trip_Count_Max": "3",
            "Highest_Tier_Reached": "1",
            "Recent_Window_Sum": "",
            "Total_Runtime_s": "10",
            "Tier1_Time_s": "8",
            "Tier1_Exhaustive": "True",
        })
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(values)

    def _fixture(self, root):
        root.mkdir(parents=True, exist_ok=True)
        historical = root / "historical.csv"
        historical.write_text(
            ",".join(schemas.LEGACY_HEURISTIC_HEADER) + "\n"
            "1,900000,inf,1,9,1,9,3,-2,False,60,0,10\n"
        )
        current = root / "current.csv"
        self._write_current_csv(current)
        exact = root / "exact.iters.csv"
        exact.write_text(
            ",".join(schemas.EXACT_ITER_HEADER) + "\n"
            "10,1,800000,8,0,-0.5,20\n"
            "20,2,799000,7.9,0,0,24\n"
        )
        exact_endpoint = root / "exact.json"
        trip_ids = list(range(8))
        trip_sha = hashlib.sha256(json.dumps(
            trip_ids, separators=(",", ":")
        ).encode()).hexdigest()
        exact_endpoint.write_text(json.dumps({
            "stop_reason": "certified",
            "certified_rc_optimal": True,
            "wall_s": 20,
            "iterations": 2,
            "final": {"min_rc": 0.0},
            "final_lp": {
                "objective": 799000,
                "route_weight": 7.9,
                "artificial_total": 0,
            },
            "trip_ids": trip_ids,
            "provenance": {
                "git_commit": "a" * 40,
                "git_dirty": False,
                "instance_sha256": "b" * 64,
                "prices_sha256": "c" * 64,
                "rc_eps": 0.0001,
            },
        }))
        telemetry = root / "telemetry.jsonl"
        telemetry_identity = {
            "run": "exact-run",
            "output": "exact.json",
            "csv": "instance.csv",
            "prices_csv": "prices.csv",
            "git_commit": "a" * 40,
            "instance_sha256": "b" * 64,
            "prices_sha256": "c" * 64,
            "soc_step": 15,
            "block_min": 10,
            "g_kwh": 300,
            "charge_kw": 300,
            "min_soc_frac": 0,
            "master_sense": "cover",
            "initial_pool": "singletons",
        }
        telemetry_sha = hashlib.sha256(json.dumps(
            telemetry_identity, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        telemetry.write_text(
            json.dumps({
                "schema": schemas.TELEMETRY_SCHEMA,
                "record_type": "session_start",
                "session": 1,
                "identity_sha256": telemetry_sha,
                "identity": telemetry_identity,
            }) + "\n"
            + json.dumps({
                "schema": schemas.TELEMETRY_SCHEMA,
                "record_type": "phase",
                "session": 1,
                "identity_sha256": telemetry_sha,
                "phase": "pricing",
                "duration_s": 5,
                "elapsed_session_s": 5,
                "iteration": 1,
                "outcome": "ok",
            }) + "\n"
        )
        exact_journal = root / "exact.json.columns.jsonl"
        exact_journal.write_text("".join(
            json.dumps({"trips": [index], "cost": 100012.5}) + "\n"
            for index in range(8)
        ))
        pool_status_sha = self._sha(exact_endpoint)
        pool_journal_sha = self._sha(exact_journal)
        checkpoint = root / "checkpoint.json"
        checkpoint.write_text(json.dumps({
            "schema": schemas.MIP_CHECKPOINT_SCHEMA,
            "kind": "checkpoint",
            "observational_only": True,
            "gurobi_tree_restart_supported": False,
            "checkpoint_elapsed_s": 300,
            "observed_total_elapsed_s": 300,
            "latest_statistics_observed_s": 300,
            "stage": "fleet",
            "incumbent_state": "reused_most_recent_earlier_incumbent",
            "incumbent": {
                "total_elapsed_s": 0,
                "stage_elapsed_s": 0,
                "fleet": 8,
                "objective": 800100,
                "route_vector_sha256": "e" * 64,
            },
            "latest_statistics": {
                    "statistics_incumbent_fleet": 8,
                "fleet_bound": 8,
                "objective_bound": None,
                "fleet_gap": 0,
                "node_count": 1,
                "solution_count": 1,
            },
            "first_feasible_incumbent_s": 0,
            "solver_ended_before_checkpoint": False,
            "metadata": {
                "source_result_sha256": pool_status_sha,
                "source_journal_sha256": pool_journal_sha,
                "source_initial_partition_sha256": None,
                "git_commit": "2" * 40,
                "experiment_arm": "B",
            },
        }))
        mip_final = root / "mip.json"
        mip_final.write_text(json.dumps({
            "partitioning": True,
            "experiment_arm": "B",
            "incumbent_found": True,
            "buses": 8,
            "mip_obj": 800100,
            "mip_bound": 800100,
            "mip_bound_scope": "fleet_count_only_coarse_cost_bound",
            "mip_gap": 0,
            "fleet_bound": 8,
            "fleet_proven": True,
            "status_name": "OPTIMAL",
            "optimal_scope": "fleet_only",
            "runtime_s": 300,
            "selected_routes": [
                {
                    "trips": [index],
                    "cost": 100012.5,
                    "route_nodes": ["PARX_0", index, "PARX_0"],
                    "charging_stops": {},
                }
                for index in range(8)
            ],
            "source_result_sha256": pool_status_sha,
            "source_journal_sha256": pool_journal_sha,
            "mip_provenance": {
                "observed_git_commit": "2" * 40,
                "arguments": {
                    "cover": False,
                    "two_stage": True,
                },
            },
        }))
        endpoint_current = root / "current-endpoint.json"
        endpoint_current.write_text(json.dumps({
            "Termination_Reason": "stagnation_rolling_window",
            "Total_Runtime_s": 10,
            "Final_LP_Route_Weight": 8,
            "Final_LP_Artificial_Total": 0,
            "Instance_SHA256": "b" * 64,
            "Git": {"commit": "a" * 40, "dirty": False},
        }))
        manifest_artifact = root / "release-manifest.json"
        manifest_artifact.write_text(json.dumps({
            "schema": "release",
            "files": {"artifact.bin": "5" * 64},
        }))
        replay_artifact = root / "replay.json"
        replay_routes = [{"route": ["PARX_0", index, "PARX_0"]}
                         for index in range(8)]
        replay_artifact.write_text(json.dumps({
            "routes": replay_routes,
            "infeasible": [],
            "physics": {
                "g_kwh": 300,
                "charge_kw": 300,
                "reserve_frac": 0,
            },
            "instance_csv": "instance.csv",
            "provenance": {"instance_sha256": "b" * 64},
        }))
        replay_artifact_sha = self._sha(replay_artifact)
        replay_projection = [{
            "route_nodes": route["route"],
            "charging_stops": {},
        } for route in replay_routes]
        replay_projection.sort(key=lambda value: json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ))
        replay_vector_sha = hashlib.sha256(json.dumps(
            replay_projection, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        common_exact = {
            "algorithm_family": "exact_expanded_network",
            "implementation": "exact_pricer",
            "scale_family": "union",
            "scale": 8,
            "replicate": "r1",
            "seed": 1,
            "git_commit": "a" * 40,
            "git_dirty": False,
            "instance_sha256": "b" * 64,
            "trip_set_sha256": trip_sha,
            "tariff_sha256": "c" * 64,
            "target_lp_weight": 8,
            "target_fleet": 8,
            "model": "evsp",
            "charging_discretization": "15kwh_10min",
            "battery_kwh": 300,
            "charge_kw": 300,
            "reserve_fraction": 0,
            "master_sense": "cover",
            "initializer": "singletons",
        }
        artifacts = [
            self._spec("hist", "hist-run", historical,
                       "heuristic_dp_historical_csv", {
                           "algorithm_family": "heuristic_dp_historical",
                           "implementation": "legacy_dp",
                           "scale_family": "union", "scale": 8,
                           "replicate": "r1",
                           "trip_count": 8,
                           "trip_set_sha256": trip_sha,
                           "instance_sha256": "b" * 64,
                           "tariff_sha256": "c" * 64,
                       }),
            self._spec("current", "current-run", current,
                       "heuristic_dp_current_csv", {
                           "algorithm_family": "heuristic_dp_current",
                           "implementation": "instrumented_dp",
                           "scale_family": "union", "scale": 8,
                           "replicate": "r1",
                           "instance_sha256": "b" * 64,
                           "trip_set_sha256": "3" * 64,
                           "tariff_sha256": "c" * 64,
                           "model": "evsp",
                           "charging_discretization": "15kwh_10min",
                           "battery_kwh": 300,
                           "charge_kw": 300,
                           "reserve_fraction": 0,
                           "master_sense": "cover",
                           "initializer": "singletons",
                           "target_fleet": 8,
                       }),
            self._spec("current-endpoint", "current-run", endpoint_current,
                       "endpoint_json", {
                           "algorithm_family": "heuristic_dp_current",
                           "implementation": "instrumented_dp",
                           "scale_family": "union", "scale": 8,
                           "replicate": "r1",
                           "git_commit": "a" * 40,
                           "git_dirty": False,
                           "instance_sha256": "b" * 64,
                           "trip_set_sha256": "3" * 64,
                           "tariff_sha256": "c" * 64,
                           "model": "evsp",
                           "charging_discretization": "15kwh_10min",
                           "battery_kwh": 300,
                           "charge_kw": 300,
                           "reserve_fraction": 0,
                           "master_sense": "cover",
                           "initializer": "singletons",
                           "target_fleet": 8,
                       }),
            self._spec("exact", "exact-run", exact,
                       "exact_cg_iterations_csv", common_exact),
            self._spec("exact-endpoint", "exact-run", exact_endpoint,
                       "endpoint_json", common_exact),
            self._spec("exact-journal", "exact-run", exact_journal,
                       "exact_cg_column_journal_jsonl", common_exact),
            self._spec("telemetry", "exact-run", telemetry,
                       "exact_cg_phase_telemetry_jsonl", common_exact),
            self._spec("mip-checkpoint", "mip-run", checkpoint,
                       "mip_checkpoint", {
                           "algorithm_family": "mip_finite_pool",
                           "implementation": "raw",
                           "scale_family": "union", "scale": 8,
                           "replicate": "r1", "treatment": "RAW",
                           "augmentation_kind": "none",
                           "git_commit": "2" * 40,
                           "pool_status_sha256": pool_status_sha,
                           "pool_journal_sha256": pool_journal_sha,
                           "instance_sha256": "b" * 64,
                           "tariff_sha256": "c" * 64,
                           "trip_set_sha256": trip_sha,
                           "trip_count": 8,
                           "physical_replay_validated": True,
                           "physical_replay_artifact_sha256": replay_artifact_sha,
                           "physical_replay_route_vector_sha256":
                               replay_vector_sha,
                           "model": "evsp",
                           "charging_discretization": "15kwh_10min",
                           "battery_kwh": 300,
                           "charge_kw": 300,
                           "reserve_fraction": 0,
                           "master_sense": "partition",
                           "initializer": "singletons",
                           "target_fleet": 8,
                       }),
            self._spec("mip-final", "mip-run", mip_final,
                       "mip_final", {
                           "algorithm_family": "mip_finite_pool",
                           "implementation": "raw",
                           "scale_family": "union", "scale": 8,
                           "replicate": "r1", "treatment": "RAW",
                           "augmentation_kind": "none",
                           "git_commit": "2" * 40,
                           "pool_status_sha256": pool_status_sha,
                           "pool_journal_sha256": pool_journal_sha,
                           "instance_sha256": "b" * 64,
                           "tariff_sha256": "c" * 64,
                           "trip_set_sha256": trip_sha,
                           "trip_count": 8,
                           "physical_replay_validated": True,
                           "physical_replay_artifact_sha256": replay_artifact_sha,
                           "physical_replay_route_vector_sha256":
                               replay_vector_sha,
                           "model": "evsp",
                           "charging_discretization": "15kwh_10min",
                           "battery_kwh": 300,
                           "charge_kw": 300,
                           "reserve_fraction": 0,
                           "master_sense": "partition",
                           "initializer": "singletons",
                           "target_fleet": 8,
                       }),
            self._spec("release", "release-run", manifest_artifact,
                       "artifact_manifest_json", {
                           "algorithm_family": "artifact_manifest",
                           "implementation": "release",
                           "scale_family": None, "scale": None,
                       }),
            self._spec("replay", "replay-run", replay_artifact,
                       "route_validation_json", {
                           "algorithm_family": "mip_finite_pool",
                           "implementation": "validated_replay",
                           "scale_family": "union", "scale": 8,
                           "replicate": "r1",
                           "trip_count": 8,
                           "trip_set_sha256": trip_sha,
                           "instance_sha256": "b" * 64,
                           "battery_kwh": 300,
                           "charge_kw": 300,
                           "reserve_fraction": 0,
                       }),
        ]
        expectations = [
            self._expect("heuristic_dp_current", "instrumented_dp",
                         "trajectory"),
            self._expect("exact_expanded_network", "exact_pricer",
                         "trajectory"),
            self._expect("mip_finite_pool", "raw",
                         "mip_checkpoint_and_final", treatment="RAW"),
        ]
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "evsp-dr-cross-generation-input-manifest-v1",
            "relative_paths": "manifest",
            "artifacts": artifacts,
            "coverage_expectations": expectations,
        }, indent=2))
        return manifest

    def _spec(self, artifact_id, run_id, path, kind, metadata):
        return {
            "artifact_id": artifact_id,
            "run_id": run_id,
            "artifact_role": artifact_id,
            "path": path.name,
            "artifact_type": kind,
            "expected_sha256": self._sha(path),
            "required": True,
            "metadata": metadata,
        }

    @staticmethod
    def _expect(family, implementation, evidence, treatment=None):
        return {
            "algorithm_family": family,
            "implementation": implementation,
            "treatment": treatment,
            "scale_family": "union",
            "scale": 8,
            "comparison_group": "synthetic-k8",
            "model_difference": "synthetic controlled fixture",
            "required_evidence": evidence,
            "minimum_replicates": 1,
        }

    def test_synthetic_fixture_produces_all_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = self._fixture(root)
            output = root / "output"
            result = build(
                manifest, output, repo_root=REPO_ROOT,
                command=["synthetic-test"],
            )
            self.assertEqual(result["verified_artifacts"], 11)
            required = (
                "artifact_inventory.csv", "cg_iteration_long.csv",
                "cg_run_summary.csv", "mip_checkpoint_long.csv",
                "mip_run_summary.csv", "phase_telemetry_long.csv",
                "scale_progress_summary.csv",
                "data_dictionary.csv", "SCHEMA.md",
                "provenance.json", "missing_data_and_rerun_plan.csv",
                "benchmark_rerun_plan.json", "figure_manifest.json",
                "completion.json",
            )
            required += tuple(
                f"{stem}.{extension}"
                for stem in (
                    "lp_route_weight_by_scale",
                    "normalized_target_gap_by_scale",
                    "artificial_mass_by_scale",
                    "reduced_cost_columns_by_scale",
                    "master_pricing_time_shares_by_scale",
                    "mip_incumbent_bound_by_scale",
                    "final_fleet_gap_by_scale",
                )
                for extension in ("png", "pdf")
            )
            self.assertTrue(all((output / name).is_file() for name in required))
            with (output / "cg_iteration_long.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            legacy = next(row for row in rows if row["run_id"] == "hist-run")
            current = next(
                row for row in rows if row["run_id"] == "current-run"
            )
            exact = next(row for row in rows if row["run_id"] == "exact-run")
            self.assertNotEqual(legacy["legacy_master_objective"], "")
            self.assertEqual(legacy["master_objective_before_add"], "")
            self.assertNotEqual(current["master_objective_before_add"], "")
            self.assertEqual(current["legacy_master_objective"], "")
            self.assertEqual(exact["master_time_s"], "")
            with (output / "cg_run_summary.csv").open(newline="") as handle:
                summaries = list(csv.DictReader(handle))
            exact_summary = next(
                row for row in summaries if row["run_id"] == "exact-run"
            )
            self.assertEqual(exact_summary["pricing_certified"], "True")
            self.assertEqual(
                exact_summary["phase_telemetry_available"], "True"
            )
            with (output / "scale_progress_summary.csv").open(
                    newline="") as handle:
                progress = list(csv.DictReader(handle))
            self.assertTrue(any(
                row["run_id"] == "exact-run"
                and row["certified_lp_bound"] == "7.9"
                and row["target_gap_basis"]
                == "certified_lp_route_weight_bound"
                for row in progress
            ))
            self.assertTrue(any(
                row["run_id"] == "mip-run"
                and row["mip_incumbent_fleet"] == "8"
                and row["mip_proof_scope"].startswith("finite_pool")
                for row in progress
            ))
            with (output / "mip_checkpoint_long.csv").open(
                    newline="") as handle:
                mip_checkpoints = list(csv.DictReader(handle))
            self.assertEqual(
                mip_checkpoints[0]["statistics_incumbent_fleet"], "8.0"
            )
            figure_manifest = json.loads(
                (output / "figure_manifest.json").read_text()
            )
            self.assertEqual(
                {row["status"] for row in figure_manifest["figures"]},
                {"available"},
            )
            with (output / "artifact_coverage_matrix.csv").open(
                    newline="") as handle:
                coverage = list(csv.DictReader(handle))
            designed = [
                row for row in coverage
                if row["comparison_group"] == "synthetic-k8"
            ]
            self.assertEqual(len(designed), 3)
            self.assertTrue(all(
                row["coverage_status"] == "available" for row in designed
            ))
            historical_outside = next(
                row for row in coverage
                if row["algorithm_family"] == "heuristic_dp_historical"
            )
            self.assertEqual(
                historical_outside["size_class"],
                "observed_outside_design",
            )
            second = root / "output-second"
            build(
                manifest, second, repo_root=REPO_ROOT,
                command=["synthetic-test"],
            )
            for path in sorted(output.iterdir()):
                counterpart = second / path.name
                self.assertTrue(counterpart.is_file(), path.name)
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    hashlib.sha256(counterpart.read_bytes()).hexdigest(),
                    path.name,
                )
            with self.assertRaises(FileExistsError):
                build(
                    manifest, output, repo_root=REPO_ROOT,
                    command=["synthetic-test"],
                )

    def test_hash_corruption_duplicate_and_mixed_provenance_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._fixture(root)
            with self.assertRaisesRegex(
                ValueError, "SHA-256 mismatch"
            ):
                _build(
                    manifest_path,
                    root / "unbound-git",
                    repo_root=REPO_ROOT,
                    command=["test"],
                    approved_manifest_sha256=hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                    git_executable=Path(shutil.which("git")).resolve(),
                    expected_git_sha256="0" * 64,
                )
            with self.assertRaisesRegex(ValueError, "approved SHA"):
                build(
                    manifest_path, root / "bad-manifest-approval",
                    repo_root=REPO_ROOT, command=["test"],
                    approved_manifest_sha256="0" * 64,
                )
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"][0]["expected_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "listed artifacts"):
                build(
                    manifest_path, root / "bad-hash", repo_root=REPO_ROOT,
                    command=["test"],
                )

            manifest_path = self._fixture(root / "duplicate")
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"].append(dict(manifest["artifacts"][0]))
            manifest["artifacts"][-1]["artifact_id"] = "duplicate-id"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "duplicate run/schema"):
                build(
                    manifest_path, root / "duplicate-output",
                    repo_root=REPO_ROOT, command=["test"],
                )

            manifest_path = self._fixture(root / "mixed")
            manifest = json.loads(manifest_path.read_text())
            exact_endpoint = next(
                item for item in manifest["artifacts"]
                if item["artifact_id"] == "exact-endpoint"
            )
            exact_endpoint["metadata"]["instance_sha256"] = "9" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "mixed instance_sha256"):
                build(
                    manifest_path, root / "mixed-output",
                    repo_root=REPO_ROOT, command=["test"],
                )

    def test_tail_safe_final_row_and_interior_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._fixture(root)
            manifest = json.loads(manifest_path.read_text())
            historical = root / "historical.csv"
            historical.write_text(
                historical.read_text() + "2,broken"
            )
            manifest["artifacts"][0]["expected_sha256"] = self._sha(historical)
            manifest_path.write_text(json.dumps(manifest))
            output = root / "tail-output"
            build(
                manifest_path, output, repo_root=REPO_ROOT, command=["test"]
            )
            with (output / "artifact_inventory.csv").open(
                    newline="") as handle:
                rows = list(csv.DictReader(handle))
            row = next(item for item in rows if item["artifact_id"] == "hist")
            self.assertEqual(row["tail_dropped"], "True")

            manifest_path = self._fixture(root / "interior")
            manifest = json.loads(manifest_path.read_text())
            historical = root / "interior/historical.csv"
            lines = historical.read_text().splitlines()
            historical.write_text(
                lines[0] + "\n2,broken\n" + lines[1] + "\n"
            )
            manifest["artifacts"][0]["expected_sha256"] = self._sha(historical)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "listed artifacts"):
                build(
                    manifest_path, root / "interior-output",
                    repo_root=REPO_ROOT, command=["test"],
                )

    def test_tracked_historical_manifest_is_ingested(self):
        manifest = REPO_ROOT / (
            "CROSS_GENERATION_EVIDENCE_INPUT_MANIFEST_20260816.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "tracked"
            result = build(
                manifest, output, repo_root=REPO_ROOT,
                command=["tracked-history-test"],
            )
            self.assertEqual(result["verified_artifacts"], 5)
            self.assertGreater(result["cg_iterations"], 0)
            with (output / "artifact_inventory.csv").open(
                    newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(
                row["validation_status"] == "verified" for row in rows
            ))
            expected_hashes = {
                "legacy-pricing-20bus":
                    "577221ac9115af65e77bf5fc7baabf6a55e45d886c3a5bd8dc763d162c6ca9ed",
                "legacy-pricing-30bus":
                    "8a15fcf53b840c386b15f63b1e092f3f16828807d8bbb49078dc0b899dee1540",
                "legacy-pricing-43bus":
                    "5f46e1e793282eeb0f91b24e7f60dee8aa97edcda414483f638031b64ad62e8f",
                "legacy-pricing-10b-rnd001":
                    "29206f9318de13ca3787a9e6cd39b6e952a9514e34f1f68b76b9b8c46b4cae96",
                "legacy-pricing-10b-rnd002":
                    "49793c62b64eef9dc57c9248104d58f27fa3349fb3c01165011c9bb7b93cdb52",
            }
            self.assertEqual(
                {row["artifact_id"]: row["observed_sha256"] for row in rows},
                expected_hashes,
            )
            with (output / "cg_run_summary.csv").open(newline="") as handle:
                summaries = {
                    row["run_id"]: row for row in csv.DictReader(handle)
                }
            expected_iterations = {
                "legacy-dp-20bus-200cols": "187",
                "legacy-dp-30bus-200cols": "33",
                "legacy-dp-43bus-200cols": "19",
                "legacy-dp-10b-rnd001": "565",
                "legacy-dp-10b-rnd002": "529",
            }
            self.assertEqual(
                {
                    run_id: summaries[run_id]["iteration_count"]
                    for run_id in expected_iterations
                },
                expected_iterations,
            )
            figure_manifest = json.loads(
                (output / "figure_manifest.json").read_text()
            )
            omitted = [
                row for row in figure_manifest["figures"]
                if row["status"] == "omitted"
            ]
            self.assertTrue(omitted)
            for row in omitted:
                self.assertFalse((output / f"{row['stem']}.png").exists())
                self.assertFalse((output / f"{row['stem']}.pdf").exists())
            self.assertTrue(
                (output / "artifact_coverage_matrix.csv").is_file()
            )
            self.assertTrue(
                (output / "missing_data_and_rerun_plan.csv").is_file()
            )

    def test_telemetry_tail_and_disabled_availability_are_explicit(self):
        spec = {
            "artifact_id": "telemetry",
            "run_id": "run",
            "artifact_type": "exact_cg_phase_telemetry_jsonl",
            "metadata": {},
        }
        identity = {
            "output": "result.json",
            "csv": "instance.csv",
            "prices_csv": "prices.csv",
            "instance_sha256": "b" * 64,
            "prices_sha256": "c" * 64,
            "git_commit": "a" * 40,
            "soc_step": 15,
            "block_min": 10,
            "g_kwh": 300,
            "charge_kw": 300,
            "min_soc_frac": 0,
            "master_sense": "cover",
            "initial_pool": "singletons",
        }
        identity_sha = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        payload = (
            json.dumps({
                "schema": schemas.TELEMETRY_SCHEMA,
                "record_type": "session_start",
                "session": 1,
                "identity_sha256": identity_sha,
                "identity": identity,
            }) + "\n"
            + json.dumps({
                "schema": schemas.TELEMETRY_SCHEMA,
                "record_type": "phase",
                "session": 1,
                "identity_sha256": identity_sha,
                "phase": "master",
                "duration_s": 1,
                "elapsed_session_s": 1,
            }) + "\n{\"broken\":"
        ).encode()
        parsed = schemas.parse_artifact(payload, spec)
        self.assertTrue(parsed["tail"]["tail_dropped"])
        self.assertEqual(len(parsed["telemetry_rows"]), 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._fixture(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["artifacts"] = [
                item for item in manifest["artifacts"]
                if item["artifact_id"] != "telemetry"
            ]
            manifest_path.write_text(json.dumps(manifest))
            output = root / "without-telemetry"
            build(
                manifest_path, output, repo_root=REPO_ROOT,
                command=["test"],
            )
            with (output / "cg_run_summary.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            exact = next(row for row in rows if row["run_id"] == "exact-run")
            self.assertEqual(exact["phase_telemetry_available"], "False")
            self.assertEqual(
                exact["phase_telemetry_reason"],
                "telemetry_disabled_or_not_supplied",
            )

    def test_legacy_mip_checkpoint_without_statistics_incumbent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._fixture(root)
            manifest = json.loads(manifest_path.read_text())
            checkpoint_path = root / "checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            checkpoint["latest_statistics"].pop(
                "statistics_incumbent_fleet"
            )
            checkpoint_path.write_text(json.dumps(checkpoint))
            checkpoint_spec = next(
                artifact for artifact in manifest["artifacts"]
                if artifact["artifact_id"] == "mip-checkpoint"
            )
            checkpoint_spec["expected_sha256"] = self._sha(checkpoint_path)
            manifest_path.write_text(json.dumps(manifest))
            output = root / "legacy-checkpoint"
            build(
                manifest_path, output, repo_root=REPO_ROOT,
                command=["test"],
            )
            with (output / "mip_checkpoint_long.csv").open(
                    newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["statistics_incumbent_fleet"], "")

    def test_missing_current_field_and_mip_semantic_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = self._fixture(root)
            manifest = json.loads(manifest_path.read_text())
            current = root / "current.csv"
            with current.open(newline="") as handle:
                rows = list(csv.reader(handle))
            removed = rows[0].index("Pricing_Dominance_Mode")
            for row in rows:
                row.pop(removed)
            with current.open("w", newline="") as handle:
                csv.writer(handle).writerows(rows)
            next(
                item for item in manifest["artifacts"]
                if item["artifact_id"] == "current"
            )["expected_sha256"] = self._sha(current)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "listed artifacts"):
                build(
                    manifest_path, root / "missing-field",
                    repo_root=REPO_ROOT, command=["test"],
                )

            manifest_path = self._fixture(root / "mip-mismatch")
            manifest = json.loads(manifest_path.read_text())
            next(
                item for item in manifest["artifacts"]
                if item["artifact_id"] == "mip-final"
            )["metadata"]["pool_status_sha256"] = "9" * 64
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "listed artifacts"):
                build(
                    manifest_path, root / "mip-mismatch-output",
                    repo_root=REPO_ROOT, command=["test"],
                )

    def test_scale_progress_keeps_instance_families_separate(self):
        mip_summaries = []
        specs = {}
        for family, digest in (
            ("first_n_stress", "1" * 64),
            ("duty_union", "2" * 64),
        ):
            run_id = f"{family}-run"
            mip_summaries.append({
                "run_id": run_id,
                "algorithm_family": "mip_finite_pool",
                "implementation": "two_stage_pool_mip",
                "scale_family": family,
                "scale": 10,
                "replicate": "r1",
                "treatment": "RAW",
                "integer_fleet": 10,
                "fleet_bound": 10,
                "fleet_proven": True,
                "optimal_scope": "fleet_only",
                "runtime_s": 1,
                "physically_validated_schedule": True,
                "proof_censored": False,
                "status_name": "OPTIMAL",
            })
            specs[run_id] = [{
                "metadata": {
                    "target_fleet": 10,
                    "trip_count": 100,
                    "instance_sha256": digest,
                    "trip_set_sha256": digest,
                    "tariff_sha256": "3" * 64,
                    "initializer": "singletons",
                },
            }]
        rows = _scale_progress_rows(
            [], [], [], mip_summaries, specs
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["instance_family"] for row in rows},
            {"first_n_stress", "duty_union"},
        )

    def test_positive_artificials_block_lp_bound_and_target_gap(self):
        run_id = "artificial-run"
        rows = _scale_progress_rows(
            [{
                "run_id": run_id,
                "wall_time_s": 10,
                "iteration": 1,
            }],
            [{
                "run_id": run_id,
                "algorithm_family": "exact_expanded_network",
                "implementation": "exact_pricer",
                "scale_family": "duty_union",
                "scale": 8,
                "replicate": "r1",
                "seed": 1,
                "final_wall_time_s": 10,
                "final_lp_route_weight": 7.5,
                "final_artificial_total": 1.0,
                "final_best_reduced_cost": 0.0,
                "pricing_certified": True,
                "termination_raw": "certified",
                "certification_censored": False,
                "source_artifact_ids": "status",
            }],
            [], [], {
                run_id: [{
                    "metadata": {
                        "target_fleet": 8,
                        "artificial_tolerance": 1e-6,
                    },
                }],
            },
        )
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["certified_lp_bound"])
        self.assertIsNone(rows[0]["target_gap"])
        self.assertEqual(
            rows[0]["target_gap_basis"],
            "unavailable_artificial_mass_positive_or_unknown",
        )

    def test_scale_progress_keeps_pool_treatments_separate(self):
        summaries = []
        specs = {}
        for treatment in ("RAW", "MATCHING", "GIRO"):
            run_id = f"run-{treatment.lower()}"
            summaries.append({
                "run_id": run_id,
                "algorithm_family": "mip_finite_pool",
                "implementation": "two_stage_pool_mip",
                "scale_family": "duty_union",
                "scale": 8,
                "replicate": "r1",
                "treatment": treatment,
                "integer_fleet": 8,
                "fleet_bound": 8,
                "fleet_proven": True,
                "partitioning": True,
                "optimal_scope": "fleet_only",
                "runtime_s": 1,
                "physically_validated_schedule": True,
                "proof_censored": False,
                "status_name": "OPTIMAL",
            })
            specs[run_id] = [{
                "metadata": {
                    "target_fleet": 8,
                    "trip_count": 80,
                    "instance_sha256": "1" * 64,
                    "trip_set_sha256": "2" * 64,
                    "tariff_sha256": "3" * 64,
                    "initializer": "singletons",
                },
            }]
        rows = _scale_progress_rows([], [], [], summaries, specs)
        self.assertEqual(
            {row["treatment"] for row in rows},
            {"RAW", "MATCHING", "GIRO"},
        )


if __name__ == "__main__":
    unittest.main()
