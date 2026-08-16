import hashlib
import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import cross_generation_schema as schemas  # noqa: E402


class CrossGenerationSchemaAdversarialTests(unittest.TestCase):
    def test_mip_checkpoint_rejects_final_kind_and_bad_hashes(self):
        spec = {
            "artifact_id": "checkpoint",
            "run_id": "run",
            "artifact_type": "mip_checkpoint",
            "metadata": {
                "implementation": "two_stage_pool_mip",
                "scale_family": "union",
                "scale": 8,
                "replicate": "r1",
                "treatment": "RAW",
                "git_commit": "a" * 40,
                "pool_status_sha256": "b" * 64,
                "pool_journal_sha256": "c" * 64,
            },
        }
        payload = {
            "schema": schemas.MIP_CHECKPOINT_SCHEMA,
            "kind": "final",
            "observational_only": True,
            "gurobi_tree_restart_supported": False,
            "stage": "fleet",
            "checkpoint_elapsed_s": 0,
            "observed_total_elapsed_s": 0,
            "solver_ended_before_checkpoint": False,
            "incumbent_state": "no_incumbent_yet",
            "incumbent": None,
            "latest_statistics": {},
            "metadata": {
                "git_commit": "a" * 40,
                "source_result_sha256": "b" * 64,
                "source_journal_sha256": "c" * 64,
                "source_initial_partition_sha256": None,
                "experiment_arm": "B",
            },
        }
        with self.assertRaisesRegex(ValueError, "stage/state/kind"):
            schemas.parse_artifact(json.dumps(payload).encode(), spec)
        payload["kind"] = "checkpoint"
        payload["metadata"]["source_result_sha256"] = None
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            schemas.parse_artifact(json.dumps(payload).encode(), spec)

    def test_mip_final_rejects_cover_nan_and_unvalidated_schedule(self):
        spec = {
            "artifact_id": "final",
            "run_id": "run",
            "artifact_type": "mip_final",
            "metadata": {
                "algorithm_family": "mip_finite_pool",
                "implementation": "two_stage_pool_mip",
                "scale_family": "union",
                "scale": 8,
                "replicate": "r1",
                "treatment": "RAW",
                "trip_count": 1,
                "trip_set_sha256": hashlib.sha256(b"[0]").hexdigest(),
                "git_commit": "a" * 40,
                "pool_status_sha256": "b" * 64,
                "pool_journal_sha256": "c" * 64,
                "physical_replay_validated": False,
            },
        }
        payload = {
            "partitioning": False,
            "experiment_arm": "B",
            "incumbent_found": True,
            "buses": 1,
            "selected_routes": [{"trips": [0], "cost": 100000}],
            "mip_obj": float("nan"),
            "mip_bound": 100000,
            "mip_bound_scope": "fleet_count_only_coarse_cost_bound",
            "mip_gap": 0,
            "runtime_s": 1,
            "fleet_bound": 1,
            "fleet_proven": True,
            "status_name": "OPTIMAL",
            "optimal_scope": "fleet_only",
            "source_result_sha256": "b" * 64,
            "source_journal_sha256": "c" * 64,
            "mip_provenance": {
                "observed_git_commit": "a" * 40,
                "arguments": {"cover": False, "two_stage": True},
            },
        }
        with self.assertRaisesRegex(ValueError, "covering"):
            schemas.parse_artifact(json.dumps(payload).encode(), spec)
        payload["partitioning"] = True
        with self.assertRaisesRegex(ValueError, "non-finite"):
            schemas.parse_artifact(json.dumps(payload).encode(), spec)

    def test_telemetry_recomputes_identity_and_rejects_bad_types(self):
        identity = {"run": "one"}
        wrong = "d" * 64
        rows = [
            {
                "schema": schemas.TELEMETRY_SCHEMA,
                "record_type": "session_start",
                "session": 1,
                "identity": identity,
                "identity_sha256": wrong,
            }
        ]
        spec = {
            "artifact_id": "telemetry",
            "run_id": "run",
            "artifact_type": "exact_cg_phase_telemetry_jsonl",
            "metadata": {},
        }
        with self.assertRaisesRegex(ValueError, "session identity"):
            schemas.parse_artifact(
                (json.dumps(rows[0]) + "\n").encode(), spec
            )
        digest = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        rows[0]["identity_sha256"] = digest
        rows.append({
            "schema": schemas.TELEMETRY_SCHEMA,
            "record_type": "phase",
            "session": 1,
            "identity_sha256": digest,
            "phase": "pricing",
            "duration_s": True,
            "elapsed_session_s": 1,
        })
        with self.assertRaisesRegex(ValueError, "boolean"):
            schemas.parse_artifact(
                ("\n".join(json.dumps(row) for row in rows) + "\n").encode(),
                spec,
            )

    def test_sequence_validation_rejects_decreasing_iterations(self):
        payload = (
            ",".join(schemas.EXACT_ITER_HEADER) + "\n"
            "1,2,10,1,0,-1,2\n"
            "2,1,9,1,0,-1,3\n"
        ).encode()
        spec = {
            "artifact_id": "exact",
            "run_id": "run",
            "artifact_type": "exact_cg_iterations_csv",
            "metadata": {},
        }
        with self.assertRaisesRegex(ValueError, "duplicated/decreasing"):
            schemas.parse_artifact(payload, spec)


if __name__ == "__main__":
    unittest.main()
