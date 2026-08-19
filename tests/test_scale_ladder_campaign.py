import csv
import copy
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import launch_scale_ladder as ladder  # noqa: E402
from tariff_response_environment import (  # noqa: E402
    NUMPY_BUILD_IDENTITY_SCHEMA,
    PORTABLE_FIELDS,
    _runtime_simd_metadata,
    _stable_build_config,
    compare_portable,
    identity as environment_identity,
)
from build_scale_ladder_inputs import build as build_inputs  # noqa: E402
from build_tariff_response_manifest import sha256_file  # noqa: E402
from scale_ladder_trip_identity import (  # noqa: E402
    classify_legacy_trip_hash,
    identity,
    require_compatible,
)
from summarize_scale_ladder import (  # noqa: E402
    CG_FIELDS,
    PROGRESS_FIELDS,
    _validate_completion,
    summarize,
    target_gap_interpretation,
    _rename_noreplace,
)
from audit_scale_ladder_known_membership import audit  # noqa: E402
from recover_scale_ladder_mip_progress import recover  # noqa: E402
from run_scale_ladder_local_diagnostics import (  # noqa: E402
    LOCAL_CODE_PATHS,
)
from reconcile_scale_ladder_gate import (  # noqa: E402
    _dependency_semantics,
    _discover_intended_science_jobs,
    _discover_held_science_jobs,
    _discover_probe_job,
    _hard_probe_mismatch,
    _resolve_gate_state,
    _reconcile_locked,
    _validate_held_array_controller,
    reconcile as reconcile_gate,
)


INSTANCE_MANIFEST = (
    REPO_ROOT
    / "data/scale_ladder/instances/scale_ladder_instance_manifest.csv"
)


class ScaleLadderCampaignTests(unittest.TestCase):
    def _probe_specs(self, root, *, artifacts=False, plan_sha=None):
        plan_sha = plan_sha or "p" * 64
        specs = {}
        for partition, job_id in (
            ("default_partition", "101"),
            ("scaglione", "102"),
        ):
            spec = ladder._probe_spec(plan_sha, partition, root, 1)
            spec["job_id"] = job_id
            specs[partition] = spec
            if not artifacts:
                continue
            output = Path(spec["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                "schema": "evsp-dr-scale-ladder-environment-probe-v1",
                "compatible": True,
                "plan_sha256": plan_sha,
                "probe_id": spec["probe_id"],
                "probe_attempt": 1,
                "slurm_job_id": job_id,
                "slurm_partition": partition,
                "differences": [],
                "observed_node_metadata": {},
                "planned_portable_identity_sha256": "a" * 64,
                "observed_portable_identity_sha256": "a" * 64,
            }))
            Path(str(output) + ".sha256").write_text(
                f"{sha256_file(output)}  {output.name}\n"
            )
        return specs

    def _portable_identity(self):
        portable = {
            field: f"value-{field}" for field in PORTABLE_FIELDS
        }
        encoded = json.dumps(
            portable, sort_keys=True, separators=(",", ":")
        ).encode()
        return {
            "schema": "evsp-dr-portable-environment-v1",
            "portable": portable,
            "portable_identity_sha256":
                hashlib.sha256(encoded).hexdigest(),
            "node_metadata": {
                "platform": "login",
                "hostname": "login01",
                "kernel_release": "login-kernel",
            },
        }

    def _write_activation_campaign(self, root, *, activation_job="303"):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        plan = {
            "campaign_root": str(root),
            "submission_protocol": "probe_first_activation_v1",
        }
        raw = ladder.canonical(plan)
        plan_sha = hashlib.sha256(raw).hexdigest()
        (root / "approved-plan.json").write_bytes(raw)
        specs = {
            partition: ladder._probe_spec(
                plan_sha, partition, root, attempt=1
            )
            for partition in ladder.PROBE_PARTITIONS
        }
        specs["default_partition"]["job_id"] = "301"
        specs["scaglione"]["job_id"] = "302"
        activation = ladder._activation_spec(
            plan_sha, root, specs, attempt=1
        )
        activation["job_id"] = activation_job
        manifest = {
            **plan,
            "approval_sha256": plan_sha,
            "submitted": False,
            "submission_state": "activation_released",
            "probe_state": "submitted",
            "reservation_state": "not_created",
            "reservations": [],
            "gate_state": "not_created",
            "gate_job_id": None,
            "submitted_arrays": {},
            "infrastructure_probes": specs,
            "activation": activation,
        }
        (root / "campaign.json").write_text(json.dumps(manifest))
        return plan, plan_sha, specs, activation, manifest

    def _probe_result(
        self, spec, *, compatible, state, resolution,
        differences=None,
    ):
        return {
            "job_id": spec["job_id"],
            "state": state,
            "state_resolution": resolution,
            "output": spec["output"],
            "compatible": compatible,
            "probe_id": spec["probe_id"],
            "partition": spec["partition"],
            "attempt": spec["attempt"],
            "comment": spec["comment"],
            "job_name": spec["job_name"],
            "path_bound": True,
            "differences": list(differences or []),
        }

    def _write_pre_gate_retry_campaign(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        plan = {
            "campaign_root": str(root),
            "submission_protocol": "probe_first_activation_v1",
        }
        raw = ladder.canonical(plan)
        plan_sha = hashlib.sha256(raw).hexdigest()
        (root / "approved-plan.json").write_bytes(raw)
        specs = {
            partition: ladder._probe_spec(
                plan_sha, partition, root, attempt=1
            )
            for partition in ladder.PROBE_PARTITIONS
        }
        specs["default_partition"].update({
            "job_id": "101", "released": True,
        })
        specs["scaglione"].update({
            "job_id": "102", "released": True,
        })
        activation = ladder._activation_spec(
            plan_sha, root, specs, attempt=1,
            dependency_job_ids=["101", "102"],
        )
        activation.update({"job_id": "103", "released": True})
        results = {
            "default_partition": self._probe_result(
                specs["default_partition"], compatible=True,
                state="COMPLETED", resolution="accounting_terminal",
            ),
            "scaglione": self._probe_result(
                specs["scaglione"], compatible=False,
                state="PREEMPTED", resolution="scheduler_failure",
            ),
        }
        manifest = {
            **plan,
            "approval_sha256": plan_sha,
            "submitted": False,
            "submission_state": "probe_failed",
            "probe_state": "failed",
            "probe_results": results,
            "reservation_state": "not_created",
            "reservations": [],
            "gate_state": "not_created",
            "gate_job_id": None,
            "submitted_arrays": {},
            "infrastructure_probes": specs,
            "activation": activation,
        }
        (root / "campaign.json").write_text(json.dumps(manifest))
        return plan, plan_sha, specs, activation, results

    def _held_science_fixture(self, root, *, preflight_tasks=3):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        tools = {}
        for name in ("squeue", "scontrol", "sacct"):
            path = root / name
            path.write_text(name)
            tools[name] = {
                "available": True,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        groups = (
            "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
            "MIP_RAW", "MIP_KNOWN",
        )
        plan = {
            "campaign_root": str(root),
            "runtime_environment": {"USER": "nc437"},
            "task_groups": {
                group: [
                    f"{group.lower()}_{index}"
                    for index in range(
                        preflight_tasks if group == "PREFLIGHT" else 1
                    )
                ]
                for group in groups
            },
            **tools,
        }
        raw = ladder.canonical(plan)
        plan_sha = hashlib.sha256(raw).hexdigest()
        manifest = {
            "gate_job_id": "100",
            "submitted_arrays": {"PREFLIGHT": "401"},
        }
        return plan, plan_sha, manifest, tools

    def test_portable_environment_ignores_node_metadata(self):
        planned = self._portable_identity()
        observed = copy.deepcopy(planned)
        observed["node_metadata"] = {
            "platform": "compute",
            "hostname": "worker99",
            "kernel_release": "different-kernel",
        }
        self.assertEqual(compare_portable(planned, observed), [])

    def test_runtime_simd_capabilities_are_not_portable_build_identity(self):
        base = {
            "Compilers": {
                "c": {"name": "gcc", "version": "10.2.1"},
            },
            "Build Dependencies": {
                "blas": {
                    "name": "openblas64",
                    "found": True,
                    "version": "0.3.23.dev",
                },
            },
            "SIMD Extensions": {
                "baseline": ["SSE", "SSE2", "SSE3"],
                "found": ["SSSE3", "SSE41", "POPCNT", "AVX2"],
                "not found": ["AVX512F", "AVX512_SKX"],
            },
        }
        snavely = copy.deepcopy(base)
        snavely["SIMD Extensions"].update({
            "found": ["SSSE3", "SSE41"],
            "not found": ["POPCNT", "AVX2", "AVX512F", "AVX512_SKX"],
        })
        scaglione = copy.deepcopy(base)
        scaglione["SIMD Extensions"].update({
            "found": [
                "SSSE3", "SSE41", "POPCNT", "AVX2",
                "AVX512F", "AVX512_SKX",
            ],
            "not found": [],
        })

        planned = self._portable_identity()
        planned["portable"]["numpy_build_identity_schema"] = (
            NUMPY_BUILD_IDENTITY_SCHEMA
        )
        planned["portable"]["numpy_build"] = _stable_build_config(base)
        observed_snavely = copy.deepcopy(planned)
        observed_snavely["portable"]["numpy_build"] = (
            _stable_build_config(snavely)
        )
        observed_scaglione = copy.deepcopy(planned)
        observed_scaglione["portable"]["numpy_build"] = (
            _stable_build_config(scaglione)
        )

        self.assertEqual(compare_portable(planned, observed_snavely), [])
        self.assertEqual(compare_portable(planned, observed_scaglione), [])
        self.assertNotEqual(
            _runtime_simd_metadata(snavely),
            _runtime_simd_metadata(scaglione),
        )
        self.assertEqual(
            _runtime_simd_metadata(scaglione)["not found"], []
        )

        cleaned_all_found = copy.deepcopy(scaglione)
        del cleaned_all_found["SIMD Extensions"]["not found"]
        self.assertEqual(
            _stable_build_config(scaglione),
            _stable_build_config(cleaned_all_found),
        )
        self.assertEqual(
            _runtime_simd_metadata(cleaned_all_found)["not found"], []
        )

        cleaned_none_found = copy.deepcopy(base)
        cleaned_none_found["SIMD Extensions"]["not found"] = (
            cleaned_none_found["SIMD Extensions"].pop("found")
        ) + cleaned_none_found["SIMD Extensions"]["not found"]
        explicit_none_found = copy.deepcopy(cleaned_none_found)
        explicit_none_found["SIMD Extensions"]["found"] = []
        self.assertEqual(
            _stable_build_config(cleaned_none_found),
            _stable_build_config(explicit_none_found),
        )
        self.assertEqual(
            _runtime_simd_metadata(cleaned_none_found)["found"], []
        )

        incompatible = copy.deepcopy(base)
        incompatible["SIMD Extensions"]["baseline"] = ["AVX2"]
        observed_incompatible = copy.deepcopy(planned)
        observed_incompatible["portable"]["numpy_build"] = (
            _stable_build_config(incompatible)
        )
        differences = compare_portable(planned, observed_incompatible)
        self.assertEqual(
            [item["field"] for item in differences],
            ["portable.numpy_build"],
        )

        different_dispatch = copy.deepcopy(base)
        different_dispatch["SIMD Extensions"].update({
            "found": ["SSSE3", "SSE41", "POPCNT", "AVX2"],
            "not found": ["AVX512F"],
        })
        observed_dispatch = copy.deepcopy(planned)
        observed_dispatch["portable"]["numpy_build"] = (
            _stable_build_config(different_dispatch)
        )
        self.assertEqual(
            [
                item["field"]
                for item in compare_portable(planned, observed_dispatch)
            ],
            ["portable.numpy_build"],
        )

        different_build = copy.deepcopy(base)
        different_build["Compilers"]["c"]["version"] = "11.4.0"
        self.assertNotEqual(
            _stable_build_config(base),
            _stable_build_config(different_build),
        )

        different_dependency = copy.deepcopy(base)
        different_dependency["Build Dependencies"]["blas"]["found"] = False
        self.assertNotEqual(
            _stable_build_config(base),
            _stable_build_config(different_dependency),
        )

        reordered = copy.deepcopy(base)
        for field in ("baseline", "found", "not found"):
            reordered["SIMD Extensions"][field].reverse()
        self.assertEqual(
            _stable_build_config(base),
            _stable_build_config(reordered),
        )

        malformed = copy.deepcopy(base)
        malformed["SIMD Extensions"]["found"] = "AVX2"
        with self.assertRaisesRegex(ValueError, "must be a string list"):
            _stable_build_config(malformed)

        old_policy = copy.deepcopy(planned)
        del old_policy["portable"]["numpy_build_identity_schema"]
        self.assertEqual(
            compare_portable(old_policy, planned)[0]["field"],
            "portable.numpy_build_identity_schema",
        )

    def test_probe_entrypoint_imports_reviewed_sibling_in_isolated_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "reviewed-src"
            source.mkdir()
            for name in (
                "run_scale_ladder_environment_probe.py",
                "tariff_response_environment.py",
            ):
                (source / name).write_bytes(
                    (REPO_ROOT / "src" / name).read_bytes()
                )
            # A broad insertion of the sibling directory would let this file
            # shadow the standard library.  The exact-path loader must not.
            (source / "platform.py").write_text(
                "raise RuntimeError('sibling shadow module was imported')\n"
            )
            ambient = Path(tmp) / "ambient"
            ambient.mkdir()
            (ambient / "tariff_response_environment.py").write_text(
                "raise RuntimeError('ambient module was imported')\n"
            )
            environment = dict(os.environ)
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTHONHOME", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    str(source / "run_scale_ladder_environment_probe.py"),
                    "--help",
                ],
                cwd=ambient,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--plan", completed.stdout)

    def test_probe_worker_executes_isolated_entrypoint_and_publishes_artifact(
        self,
    ):
        worker_path = shutil.which(
            "sha256sum", path="/usr/local/bin:/usr/bin:/bin"
        )
        if worker_path is None:
            self.skipTest("probe worker requires Linux-style sha256sum")
        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            root = temporary / "repo"
            source = root / "src"
            source.mkdir(parents=True)
            for name in (
                "run_scale_ladder_environment_probe.py",
                "submit_scale_ladder_probe.sub",
            ):
                (source / name).write_bytes(
                    (REPO_ROOT / "src" / name).read_bytes()
                )
            (source / "tariff_response_environment.py").write_text(
                "def identity():\n"
                "    return {'portable_identity_sha256': 'test', "
                "'node_metadata': {}}\n"
                "def compare_portable(planned, observed):\n"
                "    return []\n"
            )
            subprocess.run(
                ["git", "init", "-q"], cwd=root, check=True
            )
            subprocess.run(
                ["git", "add", "src"], cwd=root, check=True
            )
            subprocess.run(
                [
                    "git", "-c", "user.name=EVSP Test", "-c",
                    "user.email=evsp-test@example.invalid", "commit", "-qm",
                    "fixture",
                ],
                cwd=root,
                check=True,
            )
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            code_hashes = {
                f"src/{name}": sha256_file(source / name)
                for name in (
                    "run_scale_ladder_environment_probe.py",
                    "submit_scale_ladder_probe.sub",
                    "tariff_response_environment.py",
                )
            }
            plan = {
                "checkout_identity": {"commit": commit},
                "code_hashes": code_hashes,
                "python_identity": {
                    "portable_identity_sha256": "test",
                },
            }
            plan_path = temporary / "approved-plan.json"
            plan_path.write_text(
                json.dumps(plan, sort_keys=True, separators=(",", ":"))
            )
            plan_sha = sha256_file(plan_path)
            output = temporary / "artifacts/default.attempt1.json"
            home = temporary / "home"
            home.mkdir()
            ambient = temporary / "ambient"
            ambient.mkdir()
            (ambient / "tariff_response_environment.py").write_text(
                "raise RuntimeError('ambient module was imported')\n"
            )
            python = Path(sys.executable).resolve()
            environment = dict(os.environ)
            environment.update({
                "SLURM_JOB_ID": "1234",
                "SLURM_JOB_PARTITION": "default_partition",
            })
            completed = subprocess.run(
                [
                    "/bin/bash",
                    str(source / "submit_scale_ladder_probe.sub"),
                    str(plan_path),
                    plan_sha,
                    "default",
                    "1",
                    str(python),
                    sha256_file(python),
                    str(root),
                    str(home),
                    str(output),
                    sha256_file(source / "submit_scale_ladder_probe.sub"),
                ],
                cwd=ambient,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text())
            self.assertTrue(payload["compatible"])
            self.assertEqual(payload["plan_sha256"], plan_sha)
            self.assertEqual(payload["probe_id"], "default")
            self.assertEqual(payload["probe_attempt"], 1)
            self.assertEqual(payload["slurm_job_id"], "1234")
            self.assertEqual(
                payload["slurm_partition"], "default_partition"
            )
            sidecar = Path(str(output) + ".sha256")
            self.assertTrue(sidecar.is_file())
            self.assertEqual(sidecar.read_text().split()[0], sha256_file(output))

    def test_launcher_reads_portable_environment_schema(self):
        payload = self._portable_identity()
        payload["portable"]["python"] = "3.12.3"
        with patch.object(
            ladder.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps(payload), stderr=""
            ),
        ):
            observed = ladder._environment(Path(sys.executable).resolve())
        self.assertEqual(observed["portable"]["python"], "3.12.3")

    def test_top_level_submit_is_probe_first_and_restart_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            plan = {
                "campaign_root": str(root),
                "submission_protocol": "probe_first_activation_v1",
            }
            plan_sha = hashlib.sha256(ladder.canonical(plan)).hexdigest()
            released = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="released", stderr=""
            )

            def release_probe(_plan, spec):
                current = json.loads((root / "campaign.json").read_text())
                recorded = current["infrastructure_probes"][
                    spec["partition"]
                ]
                self.assertEqual(recorded["job_id"], spec["job_id"])
                self.assertEqual(current["activation"]["job_id"], "103")
                return released

            def release_activation(_plan, spec):
                current = json.loads((root / "campaign.json").read_text())
                self.assertEqual(
                    current["activation"]["job_id"], spec["job_id"]
                )
                return released

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_recover_probe_job_id", return_value=None
            ), patch.object(
                ladder, "_submit_probe", side_effect=["101", "102"]
            ) as submit_probe, patch.object(
                ladder, "_release_held_probe", side_effect=release_probe
            ) as release_probe, patch.object(
                ladder, "_discover_bound_job", return_value=None
            ), patch.object(
                ladder, "_submit_activation", return_value="103"
            ) as submit_activation, patch.object(
                ladder, "_release_held_activation",
                side_effect=release_activation,
            ) as release_activation, patch.object(
                ladder, "_stage_scientific_inputs"
            ) as stage_inputs, patch.object(
                ladder, "_ensure_reservations"
            ) as reserve, patch.object(
                ladder, "_submit_array"
            ) as submit_array:
                first = ladder.submit(plan, plan_sha)
                with patch.object(
                    ladder, "_resolve_bound_job",
                    return_value={"state": "PENDING", "live": True},
                ):
                    second = ladder.submit(plan, plan_sha)
            self.assertEqual(submit_probe.call_count, 2)
            self.assertTrue(all(
                call.kwargs == {"held": True}
                for call in submit_probe.call_args_list
            ))
            self.assertEqual(release_probe.call_count, 2)
            submit_activation.assert_called_once()
            release_activation.assert_called_once()
            stage_inputs.assert_not_called()
            reserve.assert_not_called()
            submit_array.assert_not_called()
            self.assertEqual(first["reservations"], [])
            self.assertEqual(first["submitted_arrays"], {})
            self.assertIsNone(first["gate_job_id"])
            self.assertEqual(first["gate_state"], "not_created")
            self.assertTrue(first["activation"]["released"])
            self.assertEqual(second["activation"]["job_id"], "103")
            self.assertFalse((root / "input").exists())

    def test_top_level_recovers_probe_accepted_before_id_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            plan = {
                "campaign_root": str(root),
                "submission_protocol": "probe_first_activation_v1",
            }
            plan_sha = hashlib.sha256(ladder.canonical(plan)).hexdigest()
            released = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="released", stderr=""
            )
            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_recover_probe_job_id", return_value=None
            ), patch.object(
                ladder, "_submit_probe",
                side_effect=RuntimeError("sbatch outcome ambiguous"),
            ):
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    ladder.submit(plan, plan_sha)
            recorded = json.loads((root / "campaign.json").read_text())
            self.assertIsNone(
                recorded["infrastructure_probes"]
                ["default_partition"]["job_id"]
            )

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_recover_probe_job_id",
                side_effect=["501", "502"],
            ) as recover, patch.object(
                ladder, "_submit_probe"
            ) as submit_probe, patch.object(
                ladder, "_release_held_probe", return_value=released
            ), patch.object(
                ladder, "_discover_bound_job", return_value=None
            ), patch.object(
                ladder, "_submit_activation", return_value="503"
            ), patch.object(
                ladder, "_release_held_activation", return_value=released
            ):
                resumed = ladder.submit(plan, plan_sha)
            self.assertEqual(recover.call_count, 2)
            submit_probe.assert_not_called()
            self.assertEqual(
                resumed["infrastructure_probes"]["default_partition"]
                ["job_id"],
                "501",
            )
            self.assertEqual(
                resumed["infrastructure_probes"]["scaglione"]["job_id"],
                "502",
            )

    def test_top_level_recovers_activation_accepted_before_id_publication(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            plan = {
                "campaign_root": str(root),
                "submission_protocol": "probe_first_activation_v1",
            }
            plan_sha = hashlib.sha256(ladder.canonical(plan)).hexdigest()
            released = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="released", stderr=""
            )
            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_recover_probe_job_id", return_value=None
            ), patch.object(
                ladder, "_submit_probe", side_effect=["601", "602"]
            ), patch.object(
                ladder, "_release_held_probe", return_value=released
            ) as premature_release, patch.object(
                ladder, "_discover_bound_job", return_value=None
            ), patch.object(
                ladder, "_submit_activation",
                side_effect=RuntimeError("sbatch outcome ambiguous"),
            ):
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    ladder.submit(plan, plan_sha)
            premature_release.assert_not_called()
            recorded = json.loads((root / "campaign.json").read_text())
            self.assertIsNone(recorded["activation"]["job_id"])

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_discover_bound_job", return_value="603"
            ) as discover, patch.object(
                ladder, "_submit_activation"
            ) as submit_activation, patch.object(
                ladder, "_release_held_probe", return_value=released
            ) as release_probe, patch.object(
                ladder, "_release_held_activation", return_value=released
            ):
                resumed = ladder.submit(plan, plan_sha)
            discover.assert_called_once()
            submit_activation.assert_not_called()
            self.assertEqual(release_probe.call_count, 2)
            self.assertEqual(resumed["activation"]["job_id"], "603")
            self.assertTrue(resumed["activation"]["released"])

    def test_pre_gate_retry_replaces_only_scheduler_failed_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            plan, plan_sha, _specs, _activation, results = (
                self._write_pre_gate_retry_campaign(root)
            )
            released = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="released", stderr=""
            )

            def resolved(_plan, spec):
                job_id = str(spec["job_id"])
                if job_id == "103":
                    return {"state": "FAILED", "live": False,
                            "exit_code": "1:0"}
                if job_id == "101":
                    return {"state": "COMPLETED", "live": False,
                            "exit_code": "0:0"}
                if job_id == "102":
                    return {"state": "PREEMPTED", "live": False,
                            "exit_code": "0:15"}
                raise AssertionError(spec)

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_wait_for_probes", return_value=results
            ), patch.object(
                ladder, "_resolve_bound_job", side_effect=resolved
            ), patch.object(
                ladder, "_recover_probe_job_id", return_value=None
            ), patch.object(
                ladder, "_submit_probe", return_value="104"
            ) as submit_probe, patch.object(
                ladder, "_discover_bound_job", return_value=None
            ), patch.object(
                ladder, "_submit_activation", return_value="105"
            ) as submit_activation, patch.object(
                ladder, "_release_held_probe", return_value=released
            ) as release_probe, patch.object(
                ladder, "_release_held_activation", return_value=released
            ):
                observed = ladder.submit(
                    plan, plan_sha, retry_failed_probes=True
                )

            submit_probe.assert_called_once()
            self.assertEqual(
                submit_probe.call_args.args[3]["partition"], "scaglione"
            )
            submit_activation.assert_called_once()
            self.assertEqual(release_probe.call_count, 1)
            self.assertEqual(
                observed["infrastructure_probes"]["default_partition"]
                ["job_id"],
                "101",
            )
            replacement = observed["infrastructure_probes"]["scaglione"]
            self.assertEqual(replacement["attempt"], 2)
            self.assertEqual(replacement["job_id"], "104")
            self.assertTrue(replacement["released"])
            self.assertEqual(observed["activation"]["attempt"], 2)
            self.assertEqual(observed["activation"]["job_id"], "105")
            self.assertEqual(
                observed["activation"]["dependency_job_ids"], ["104"]
            )
            self.assertEqual(
                observed["probe_attempt_history"]["scaglione"][0]
                ["spec"]["job_id"],
                "102",
            )
            self.assertEqual(
                observed["infrastructure_retry"]["state"], "dispatched"
            )
            self.assertEqual(observed["gate_state"], "not_created")
            self.assertEqual(observed["reservations"], [])
            self.assertFalse((root / "input").exists())

    def test_pre_gate_probe_retry_refuses_live_or_environment_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "live"
            plan, plan_sha, _specs, _activation, results = (
                self._write_pre_gate_retry_campaign(root)
            )
            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_resolve_bound_job",
                return_value={"state": "PENDING", "live": True},
            ), patch.object(ladder, "_submit_probe") as submitted:
                with self.assertRaisesRegex(RuntimeError, "controller is live"):
                    ladder.submit(
                        plan, plan_sha, retry_failed_probes=True
                    )
            submitted.assert_not_called()
            unchanged = json.loads((root / "campaign.json").read_text())
            self.assertEqual(unchanged["activation"]["attempt"], 1)
            self.assertNotIn("infrastructure_retry", unchanged)

            mismatch_root = Path(tmp) / "mismatch"
            plan, plan_sha, _specs, _activation, results = (
                self._write_pre_gate_retry_campaign(mismatch_root)
            )
            mismatch = copy.deepcopy(results)
            mismatch["scaglione"].update({
                "state_resolution": "environment_mismatch",
                "differences": [{"field": "portable.numpy"}],
            })

            def resolved(_plan, spec):
                if spec["job_id"] == "103":
                    return {"state": "FAILED", "live": False,
                            "exit_code": "1:0"}
                return {"state": "COMPLETED", "live": False,
                        "exit_code": "0:0"}

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_resolve_bound_job", side_effect=resolved
            ), patch.object(
                ladder, "_wait_for_probes", return_value=mismatch
            ), patch.object(ladder, "_submit_probe") as submitted:
                with self.assertRaisesRegex(ValueError, "non-retryable"):
                    ladder.submit(
                        plan, plan_sha, retry_failed_probes=True
                    )
            submitted.assert_not_called()
            refused = json.loads(
                (mismatch_root / "campaign.json").read_text()
            )
            self.assertNotIn("infrastructure_retry", refused)
            self.assertEqual(refused["gate_state"], "not_created")

    def test_activation_only_retry_reobserves_when_summary_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            plan, plan_sha, specs, _activation, _results = (
                self._write_pre_gate_retry_campaign(root)
            )
            manifest_path = root / "campaign.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("probe_results")
            manifest_path.write_text(json.dumps(manifest))
            passed = {
                partition: self._probe_result(
                    spec, compatible=True, state="COMPLETED",
                    resolution="accounting_terminal",
                )
                for partition, spec in specs.items()
            }
            released = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="released", stderr=""
            )

            def resolved(_plan, spec):
                if spec["job_id"] == "103":
                    return {"state": "NODE_FAIL", "live": False,
                            "exit_code": "0:1"}
                return {"state": "COMPLETED", "live": False,
                        "exit_code": "0:0"}

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_resolve_bound_job", side_effect=resolved
            ), patch.object(
                ladder, "_wait_for_probes", return_value=passed
            ), patch.object(ladder, "_submit_probe") as submit_probe, \
                    patch.object(
                        ladder, "_discover_bound_job", return_value=None
                    ), \
                    patch.object(
                        ladder, "_submit_activation", return_value="106"
                    ), patch.object(
                        ladder, "_release_held_activation",
                        return_value=released,
                    ):
                observed = ladder.submit(
                    plan, plan_sha, retry_failed_activation=True
                )
            submit_probe.assert_not_called()
            self.assertEqual(observed["activation"]["attempt"], 2)
            self.assertEqual(observed["activation"]["job_id"], "106")
            self.assertEqual(
                observed["activation"]["dependency_job_ids"], []
            )
            self.assertTrue(
                ladder._probes_compatible(observed["probe_results"])
            )

    def test_retry_crash_windows_resume_same_attempt_without_science(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            plan, plan_sha, _specs, _activation, results = (
                self._write_pre_gate_retry_campaign(root)
            )
            # Simulate a probe killed between the no-clobber JSON publication
            # and its checksum sidecar publication.  Attempt 1 remains
            # immutable; recovery must use attempt 2.
            publication_results = copy.deepcopy(results)
            publication_results["scaglione"].update({
                "state": "COMPLETED",
                "state_resolution": "awaiting_artifact",
                "artifact_status": "awaiting_sidecar",
                "observer_deadline_reached": True,
            })
            original_output = Path(
                publication_results["scaglione"]["output"]
            )
            original_output.parent.mkdir(parents=True, exist_ok=True)
            original_output.write_text('{"partial_publication": true}\n')
            manifest_path = root / "campaign.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["probe_results"] = publication_results
            manifest_path.write_text(json.dumps(manifest))

            def resolved(_plan, spec):
                if spec["job_id"] == "103":
                    return {"state": "FAILED", "live": False,
                            "exit_code": "1:0"}
                if spec["job_id"] == "101":
                    return {"state": "COMPLETED", "live": False,
                            "exit_code": "0:0"}
                return {"state": "COMPLETED", "live": False,
                        "exit_code": "0:0"}

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_resolve_bound_job", side_effect=resolved
            ), patch.object(
                ladder, "_wait_for_probes", return_value=publication_results
            ), patch.object(
                ladder, "_recover_probe_job_id", return_value=None
            ), patch.object(
                ladder, "_submit_probe",
                side_effect=RuntimeError("probe sbatch ambiguous"),
            ):
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    ladder.submit(
                        plan, plan_sha, retry_failed_probes=True
                    )
            first_crash = json.loads((root / "campaign.json").read_text())
            self.assertEqual(
                first_crash["infrastructure_retry"]["state"], "authorized"
            )
            self.assertEqual(
                first_crash["infrastructure_probes"]["scaglione"]
                ["attempt"],
                2,
            )
            self.assertIsNone(
                first_crash["infrastructure_probes"]["scaglione"]
                ["job_id"]
            )
            self.assertEqual(
                first_crash["infrastructure_probes"]["scaglione"]
                ["submission_intent"],
                "accepted_or_ambiguous",
            )
            self.assertTrue(original_output.is_file())
            self.assertNotEqual(
                first_crash["infrastructure_probes"]["scaglione"]
                ["output"],
                str(original_output),
            )
            with patch.object(
                ladder, "_validate_submission_contract"
            ):
                with self.assertRaisesRegex(RuntimeError, "explicitly authorized"):
                    ladder.submit(plan, plan_sha)

            released = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="released", stderr=""
            )
            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_recover_probe_job_id", return_value="104"
            ), patch.object(ladder, "_submit_probe") as submit_probe, \
                    patch.object(
                        ladder, "_discover_bound_job", return_value=None
                    ), \
                    patch.object(
                        ladder, "_submit_activation",
                        side_effect=RuntimeError("activation sbatch ambiguous"),
                    ):
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    ladder.submit(
                        plan, plan_sha, retry_failed_probes=True
                    )
            submit_probe.assert_not_called()
            second_crash = json.loads((root / "campaign.json").read_text())
            self.assertEqual(second_crash["activation"]["attempt"], 2)
            self.assertIsNone(second_crash["activation"]["job_id"])
            self.assertEqual(
                second_crash["activation"]["submission_intent"],
                "accepted_or_ambiguous",
            )

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_discover_bound_job", return_value="105"
            ), patch.object(ladder, "_submit_activation") as submit_activation, \
                    patch.object(
                        ladder, "_release_held_probe", return_value=released
                    ), patch.object(
                        ladder, "_release_held_activation",
                        return_value=released,
                    ):
                observed = ladder.submit(
                    plan, plan_sha, retry_failed_probes=True
                )
            submit_activation.assert_not_called()
            self.assertEqual(observed["activation"]["attempt"], 2)
            self.assertEqual(observed["activation"]["job_id"], "105")
            self.assertEqual(
                len(observed["activation_attempt_history"]), 1
            )
            self.assertEqual(
                observed["infrastructure_retry"]["state"], "dispatched"
            )
            self.assertEqual(observed["gate_state"], "not_created")

    def test_dispatched_terminal_controller_advances_once_but_live_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "terminal"
            plan, plan_sha, specs, _activation, _results = (
                self._write_pre_gate_retry_campaign(root)
            )
            retry_spec = ladder._probe_spec(
                plan_sha, "scaglione", root, 2
            )
            retry_spec.update({"job_id": "104", "released": True})
            specs["scaglione"] = retry_spec
            activation = ladder._activation_spec(
                plan_sha, root, specs, 2,
                dependency_job_ids=["104"],
            )
            activation.update({"job_id": "105", "released": True})
            passed = {
                partition: self._probe_result(
                    spec, compatible=True, state="COMPLETED",
                    resolution="accounting_terminal",
                )
                for partition, spec in specs.items()
            }
            manifest_path = root / "campaign.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update({
                "infrastructure_probes": specs,
                "activation": activation,
                "probe_results": passed,
                "infrastructure_retry": {
                    "schema":
                        "evsp-dr-scale-ladder-infrastructure-retry-v1",
                    "kind": "failed_probes",
                    "state": "dispatched",
                    "target_activation_attempt": 2,
                    "source_activation_job_id": "103",
                    "replaced_probe_partitions": ["scaglione"],
                    "activation_job_id": "105",
                },
            })
            manifest_path.write_text(json.dumps(manifest))
            released = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="released", stderr=""
            )

            def terminal(_plan, spec):
                if spec["job_id"] == "105":
                    return {"state": "FAILED", "live": False,
                            "exit_code": "1:0"}
                return {"state": "COMPLETED", "live": False,
                        "exit_code": "0:0"}

            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_resolve_bound_job", side_effect=terminal
            ), patch.object(
                ladder, "_wait_for_probes", return_value=passed
            ), patch.object(
                ladder, "_discover_bound_job", return_value=None
            ), patch.object(
                ladder, "_submit_activation", return_value="106"
            ) as submitted, patch.object(
                ladder, "_release_held_activation", return_value=released
            ):
                observed = ladder.submit(
                    plan, plan_sha, retry_failed_activation=True
                )
            submitted.assert_called_once()
            self.assertEqual(observed["activation"]["attempt"], 3)
            self.assertEqual(observed["activation"]["job_id"], "106")
            self.assertEqual(
                observed["activation"]["dependency_job_ids"], []
            )
            self.assertEqual(
                len(observed["infrastructure_retry_history"]), 1
            )

            live_root = Path(tmp) / "live"
            plan, plan_sha, specs, _activation, _results = (
                self._write_pre_gate_retry_campaign(live_root)
            )
            retry_spec = ladder._probe_spec(
                plan_sha, "scaglione", live_root, 2
            )
            retry_spec.update({"job_id": "104", "released": True})
            specs["scaglione"] = retry_spec
            activation = ladder._activation_spec(
                plan_sha, live_root, specs, 2,
                dependency_job_ids=["104"],
            )
            activation.update({"job_id": "105", "released": True})
            manifest_path = live_root / "campaign.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.update({
                "infrastructure_probes": specs,
                "activation": activation,
                "infrastructure_retry": {
                    "schema":
                        "evsp-dr-scale-ladder-infrastructure-retry-v1",
                    "kind": "failed_probes", "state": "dispatched",
                    "target_activation_attempt": 2,
                    "source_activation_job_id": "103",
                    "replaced_probe_partitions": ["scaglione"],
                    "activation_job_id": "105",
                },
            })
            manifest_path.write_text(json.dumps(manifest))
            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.object(
                ladder, "_resolve_bound_job",
                return_value={"state": "RUNNING", "live": True},
            ), patch.object(ladder, "_submit_activation") as submitted:
                with self.assertRaisesRegex(RuntimeError, "controller is live"):
                    ladder.submit(
                        plan, plan_sha, retry_failed_activation=True
                    )
            submitted.assert_not_called()
            unchanged = json.loads(manifest_path.read_text())
            self.assertEqual(unchanged["activation"]["attempt"], 2)
            self.assertEqual(
                unchanged["infrastructure_retry"]["state"], "dispatched"
            )

    def test_campaign_lock_serializes_instead_of_failing_fast(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ladder.fcntl, "flock"
        ) as flock:
            with ladder._campaign_lock(Path(tmp) / "campaign"):
                pass
        self.assertEqual(
            [call.args[1] for call in flock.call_args_list],
            [ladder.fcntl.LOCK_EX, ladder.fcntl.LOCK_UN],
        )

    def test_public_gate_reconciler_uses_the_campaign_lock(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "reconcile_scale_ladder_gate._campaign_lock"
        ) as campaign_lock, patch(
            "reconcile_scale_ladder_gate._reconcile_locked",
            return_value={"gate_state": "held"},
        ) as locked_reconcile:
            observed = reconcile_gate(
                Path(tmp) / "campaign", "a" * 64,
                release_held_gate=True,
            )
        self.assertEqual(observed["gate_state"], "held")
        campaign_lock.assert_called_once_with(
            (Path(tmp) / "campaign").resolve()
        )
        locked_reconcile.assert_called_once_with(
            (Path(tmp) / "campaign").resolve(), "a" * 64,
            release_held_gate=True,
            resume_missing_arrays=False,
            retry_failed_probes=False,
        )

    def test_activation_controller_is_held_and_uses_afterany(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probes = {
                "default_partition": {"job_id": "201"},
                "scaglione": {"job_id": "202"},
            }
            spec = ladder._activation_spec(
                "a" * 64, root, probes, attempt=1
            )
            plan = {
                "campaign_root": str(root),
                "python": {"path": "/approved/python", "sha256": "p"},
                "runtime_environment": {"HOME": "/approved/home"},
                "activation_worker_sha256": "w",
            }
            with patch.object(
                ladder, "_sbatch", return_value="203"
            ) as submitted:
                observed = ladder._submit_activation(
                    plan, root / "approved-plan.json", "a" * 64,
                    spec, root / "logs",
                )
            self.assertEqual(observed, "203")
            arguments = submitted.call_args.args[1]
            self.assertIn("--hold", arguments)
            self.assertIn("--dependency=afterany:201:202", arguments)
            self.assertNotIn("--signal=B:USR1@60", arguments)
            self.assertNotIn("afterok:201:202", arguments)

    def test_top_level_probe_submission_is_held_but_retry_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ladder._probe_spec(
                "d" * 64, "default_partition", root, 1
            )
            plan = {
                "campaign_root": str(root),
                "python": {"path": "/approved/python", "sha256": "p"},
                "runtime_environment": {"HOME": "/approved/home"},
                "probe_worker_sha256": "w",
            }
            with patch.object(
                ladder, "_sbatch", side_effect=["701", "702"]
            ) as submitted:
                ladder._submit_probe(
                    plan, root / "approved-plan.json", "d" * 64,
                    spec, root / "logs", held=True,
                )
                ladder._submit_probe(
                    plan, root / "approved-plan.json", "d" * 64,
                    spec, root / "logs",
                )
            held_arguments = submitted.call_args_list[0].args[1]
            retry_arguments = submitted.call_args_list[1].args[1]
            self.assertEqual(held_arguments[0], "--hold")
            self.assertNotIn("--hold", retry_arguments)

    def test_release_recovers_exact_terminal_job_from_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ladder._probe_spec(
                "e" * 64, "default_partition", Path(tmp), 1
            )
            spec["job_id"] = "801"
            plan = {
                "squeue": {"path": "/approved/squeue"},
                "scontrol": {"path": "/approved/scontrol"},
                "sacct": {"path": "/approved/sacct"},
                "runtime_environment": {"USER": "nc437"},
            }

            def fake_run(command, **_kwargs):
                if command[0] == "/approved/squeue":
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr=""
                    )
                if command[0] == "/approved/scontrol":
                    return SimpleNamespace(
                        returncode=1, stdout="", stderr="Invalid job id"
                    )
                if command[0] == "/approved/sacct":
                    return SimpleNamespace(
                        returncode=0,
                        stdout="|".join([
                            "801", spec["job_name"], "COMPLETED",
                            spec["partition"], spec["comment"], "0:0",
                        ]) + "\n",
                        stderr="",
                    )
                raise AssertionError(command)

            with patch.object(
                ladder.subprocess, "run", side_effect=fake_run
            ):
                observed = ladder._release_held_probe(plan, spec)
            self.assertEqual(observed.returncode, 0)

            def mismatched_run(command, **_kwargs):
                if command[0] == "/approved/squeue":
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr=""
                    )
                if command[0] == "/approved/scontrol":
                    return SimpleNamespace(
                        returncode=1, stdout="", stderr="Invalid job id"
                    )
                return SimpleNamespace(
                    returncode=0,
                    stdout="|".join([
                        "801", spec["job_name"], "COMPLETED",
                        "wrong_partition", spec["comment"], "0:0",
                    ]) + "\n",
                    stderr="",
                )

            with patch.object(
                ladder.subprocess, "run", side_effect=mismatched_run
            ):
                rejected = ladder._release_held_probe(plan, spec)
            self.assertNotEqual(rejected.returncode, 0)

    def test_job_discovery_requires_comment_partition_and_name(self):
        spec = ladder._probe_spec(
            "f" * 64, "scaglione", Path("/campaign"), 1
        )
        plan = {
            "squeue": {"path": "/approved/squeue"},
            "runtime_environment": {"USER": "nc437"},
        }
        with patch.object(
            ladder.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout=(
                    f"901|{spec['job_name']}|PENDING|{spec['partition']}|"
                    f"JobHeldUser|{spec['comment']}\n"
                ),
                stderr="",
            ),
        ):
            self.assertEqual(ladder._discover_bound_job(plan, spec), "901")
        with patch.object(
            ladder.subprocess,
            "run",
            return_value=SimpleNamespace(
                returncode=0,
                stdout=(
                    f"901|{spec['job_name']}|PENDING|default_partition|"
                    f"JobHeldUser|{spec['comment']}\n"
                ),
                stderr="",
            ),
        ):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                ladder._discover_bound_job(plan, spec)

    def test_ambiguous_probe_submission_waits_for_exact_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ladder._probe_spec(
                "f" * 64, "default_partition", Path(tmp), 1
            )
            spec["submission_intent"] = "accepted_or_ambiguous"
            plan = {
                "squeue": {"path": "/approved/squeue"},
                "runtime_environment": {"USER": "nc437"},
            }
            visible = (
                f"901|{spec['job_name']}|PENDING|{spec['partition']}|"
                f"JobHeldUser|{spec['comment']}\n"
            )
            observations = ["", visible]

            def runner(_command, **_kwargs):
                return SimpleNamespace(
                    returncode=0, stdout=observations.pop(0), stderr=""
                )

            with patch.object(
                ladder.subprocess, "run", side_effect=runner
            ), patch.object(ladder.time, "sleep") as slept:
                recovered = ladder._recover_probe_job_id(
                    plan, "f" * 64, "default_partition", spec
                )
            self.assertEqual(recovered, "901")
            slept.assert_called_once_with(
                ladder.AMBIGUOUS_DISCOVERY_DELAY_S
            )

    def test_sbatch_timeout_is_bounded_and_reported_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            sbatch = Path(tmp) / "sbatch"
            sbatch.write_text("approved")
            plan = {
                "sbatch": {
                    "available": True,
                    "path": str(sbatch),
                    "sha256": sha256_file(sbatch),
                },
            }
            with patch.object(
                ladder.subprocess, "run",
                side_effect=subprocess.TimeoutExpired([str(sbatch)], 30),
            ) as invoked:
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    ladder._sbatch(plan, ["--hold", "worker"])
            self.assertEqual(
                invoked.call_args.kwargs["timeout"], ladder.SBATCH_TIMEOUT_S
            )

    def test_held_array_recovery_collapses_tasks_and_binds_range_dependency(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            plan, plan_sha, manifest, tools = self._held_science_fixture(tmp)
            gate_row = (
                f"100|LDG{plan_sha[:5]}|PENDING|default_partition|"
                f"JobHeldUser|SLADG:{plan_sha[:20]}\n"
            )
            array_row = (
                f"401|LDPF{plan_sha[:4]}|PENDING|default_partition|"
                f"Dependency|SLAD:{plan_sha[:20]}:PREFLIGHT\n"
            )

            def runner(command, **_kwargs):
                if command[0] == tools["squeue"]["path"]:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=gate_row + array_row + array_row,
                        stderr="",
                    )
                if command[0] == tools["scontrol"]["path"]:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"JobId=401 ArrayJobId=401 "
                            f"JobName=LDPF{plan_sha[:4]} "
                            "UserId=nc437(1646707) "
                            "JobState=PENDING Partition=default_partition "
                            "Reason=Dependency RunTime=00:00:00 "
                            f"Comment=SLAD:{plan_sha[:20]}:PREFLIGHT "
                            "ArrayTaskId=0-2 "
                            "Dependency=afterok:100(unfulfilled)\n"
                        ),
                        stderr="",
                    )
                raise AssertionError(command)

            with patch(
                "reconcile_scale_ladder_gate.subprocess.run",
                side_effect=runner,
            ):
                arrays, gate = _discover_held_science_jobs(
                    plan, plan_sha, manifest
                )
            self.assertEqual(arrays, {"PREFLIGHT": "401"})
            self.assertEqual(gate, "100")

            self.assertEqual(
                _dependency_semantics("afterok:100,afterok:401"),
                _dependency_semantics("afterok:401:100"),
            )

    def test_cg_controller_requires_whole_array_dependency_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, plan_sha, _manifest, tools = self._held_science_fixture(tmp)
            plan["task_groups"]["CG"] = ["cg_0"]
            prefix = (
                "JobId=402 ArrayJobId=402 UserId=nc437(1646707) "
                f"JobName={ladder._array_name('CG', plan_sha)} "
                "JobState=PENDING Partition=default_partition "
                "Reason=Dependency RunTime=00:00:00 "
                f"Comment=SLAD:{plan_sha[:20]}:CG "
                "ArrayTaskId=0-0 Dependency="
            )

            accepted_displays = (
                "afterok:100(unfulfilled),afterok:401_*(unfulfilled)",
                "afterok:401_*:100(failed)",
            )
            for dependency in accepted_displays:
                with self.subTest(dependency=dependency), patch(
                    "reconcile_scale_ladder_gate.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout=f"{prefix}{dependency}\n",
                        stderr="",
                    ),
                ):
                    _validate_held_array_controller(
                        plan, plan_sha, "CG", "402", "100",
                        {"PREFLIGHT": "401", "CG": "402"},
                    )

            rejected_displays = (
                # Base-array scope must remain explicit in scontrol output.
                "afterok:100,afterok:401",
                # The scalar gate must not be treated as an array controller.
                "afterok:100_*,afterok:401_*",
                # A dependency on one array task is not a whole-array barrier.
                "afterok:100,afterok:401_2",
                "afterok:100,afterany:401_*",
                "afterok:100,afterok:401_*,afterok:999",
                "afterok:100,afterok:401_*,afterok:401_*",
                "afterok:100?afterok:401_*",
                "afterok:100,afterok:401_*(unexpected)",
                "afterok:100(unfulfilled)0,afterok:401_*",
                "after(unfulfilled)ok:100,afterok:401_*",
                "afterok:100(unfulfilled)(failed),afterok:401_*",
                "afterok:100,afterok:401_*,",
                "afterok:100,afterok:401_*:",
            )
            for dependency in rejected_displays:
                with self.subTest(dependency=dependency), patch(
                    "reconcile_scale_ladder_gate.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout=f"{prefix}{dependency}\n",
                        stderr="",
                    ),
                ):
                    with self.assertRaises(ValueError):
                        _validate_held_array_controller(
                            plan, plan_sha, "CG", "402", "100",
                            {"PREFLIGHT": "401", "CG": "402"},
                        )

    def test_mip_controllers_require_whole_array_dependency_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, plan_sha, _manifest, tools = self._held_science_fixture(tmp)
            cases = (
                (
                    "MIP_RAW", "404",
                    {"CG": "402", "MIP_RAW": "404"},
                    (
                        "afterok:100(unfulfilled),"
                        "aftercorr:402_*(unfulfilled)"
                    ),
                    "afterok:100,aftercorr:402",
                ),
                (
                    "MIP_KNOWN", "405",
                    {
                        "CG": "402", "SEED": "301",
                        "MIP_KNOWN": "405",
                    },
                    (
                        "aftercorr:402_*:301_*(unfulfilled),"
                        "afterok:100(unfulfilled)"
                    ),
                    "afterok:100,aftercorr:402:301",
                ),
            )
            for group, job_id, array_ids, accepted, rejected in cases:
                plan["task_groups"][group] = [f"{group.lower()}_0"]
                prefix = (
                    f"JobId={job_id} ArrayJobId={job_id} "
                    "UserId=nc437(1646707) "
                    f"JobName={ladder._array_name(group, plan_sha)} "
                    "JobState=PENDING Partition=scaglione "
                    "Reason=Dependency RunTime=00:00:00 "
                    f"Comment=SLAD:{plan_sha[:20]}:{group} "
                    "ArrayTaskId=0-0 Dependency="
                )
                with self.subTest(group=group, outcome="accepted"), patch(
                    "reconcile_scale_ladder_gate.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout=f"{prefix}{accepted}\n",
                        stderr="",
                    ),
                ):
                    _validate_held_array_controller(
                        plan, plan_sha, group, job_id, "100", array_ids,
                    )
                with self.subTest(group=group, outcome="rejected"), patch(
                    "reconcile_scale_ladder_gate.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout=f"{prefix}{rejected}\n",
                        stderr="",
                    ),
                ):
                    with self.assertRaises(ValueError):
                        _validate_held_array_controller(
                            plan, plan_sha, group, job_id, "100", array_ids,
                        )

    def test_mip_controller_accepts_captured_aftercorr_split_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, plan_sha, _manifest, _tools = self._held_science_fixture(
                tmp
            )
            plan["task_groups"]["MIP_RAW"] = [
                f"mip_raw_{index}" for index in range(21)
            ]
            dependency = (
                "afterok:100(unfulfilled),"
                "aftercorr:402_*(unfulfilled)"
            )

            def record(raw_job_id, tasks, **replacements):
                fields = {
                    "JobId": str(raw_job_id),
                    "ArrayJobId": "404",
                    "ArrayTaskId": str(tasks),
                    "JobName": ladder._array_name("MIP_RAW", plan_sha),
                    "UserId": "nc437(1646707)",
                    "JobState": "PENDING",
                    "Reason": "Dependency",
                    "RunTime": "00:00:00",
                    "Partition": "scaglione",
                    "Comment": f"SLAD:{plan_sha[:20]}:MIP_RAW",
                    "Dependency": dependency,
                }
                fields.update(replacements)
                return " ".join(
                    f"{key}={value}" for key, value in fields.items()
                )

            rows = [record(404, "10-20")] + [
                record(420 - task, task) for task in range(9, -1, -1)
            ]

            def validate(candidate_rows):
                with patch(
                    "reconcile_scale_ladder_gate.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout="\n".join(candidate_rows) + "\n",
                        stderr="",
                    ),
                ):
                    _validate_held_array_controller(
                        plan, plan_sha, "MIP_RAW", "404", "100",
                        {"CG": "402", "MIP_RAW": "404"},
                    )

            validate(rows)

            invalid = {
                "missing task": rows[:-1],
                "overlap": [record(404, "9-20"), *rows[1:]],
                "wrong parent": [
                    rows[0],
                    record(411, 9, ArrayJobId="999"),
                    *rows[2:],
                ],
                "duplicate physical job": [
                    rows[0], record(404, 9), *rows[2:]
                ],
                "inconsistent dependency": [
                    rows[0],
                    record(411, 9, Dependency="afterok:100"),
                    *rows[2:],
                ],
                "nonzero runtime": [
                    rows[0], record(411, 9, RunTime="00:00:01"), *rows[2:]
                ],
            }
            for label, candidate_rows in invalid.items():
                with self.subTest(label=label):
                    with self.assertRaises(ValueError):
                        validate(candidate_rows)

    def test_held_array_recovery_rejects_spoofed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, plan_sha, _manifest, tools = self._held_science_fixture(tmp)
            gate = (
                f"100|LDG{plan_sha[:5]}|PENDING|default_partition|"
                f"JobHeldUser|SLADG:{plan_sha[:20]}\n"
            )
            valid = (
                f"401|LDPF{plan_sha[:4]}|PENDING|default_partition|"
                f"Dependency|SLAD:{plan_sha[:20]}:PREFLIGHT\n"
            )
            cases = {
                "wrong name": valid.replace(
                    f"LDPF{plan_sha[:4]}", "WRONG"
                ),
                "wrong partition": valid.replace(
                    "default_partition", "scaglione"
                ),
                "wrong reason": valid.replace(
                    "Dependency", "Priority"
                ),
                "unknown group": valid.replace(
                    "PREFLIGHT", "UNKNOWN"
                ),
                "released state": valid.replace("PENDING", "RUNNING"),
                "two parents": valid + valid.replace("401|", "402|", 1),
            }
            for label, row in cases.items():
                with self.subTest(label=label), patch(
                    "reconcile_scale_ladder_gate.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=0, stdout=gate + row, stderr=""
                    ),
                ), patch(
                    "reconcile_scale_ladder_gate._validate_held_array_controller"
                ):
                    with self.assertRaises(ValueError):
                        _discover_held_science_jobs(
                            plan, plan_sha,
                            {"gate_job_id": "100", "submitted_arrays": {}},
                        )

    def test_held_array_recovery_rejects_wrong_range_or_gate_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, plan_sha, manifest, tools = self._held_science_fixture(tmp)
            listing = (
                f"100|LDG{plan_sha[:5]}|PENDING|default_partition|"
                f"JobHeldUser|SLADG:{plan_sha[:20]}\n"
                f"401|LDPF{plan_sha[:4]}|PENDING|default_partition|"
                f"Dependency|SLAD:{plan_sha[:20]}:PREFLIGHT\n"
            )

            def runner_with(*, task_range, dependency):
                def runner(command, **_kwargs):
                    if command[0] == tools["squeue"]["path"]:
                        return SimpleNamespace(
                            returncode=0, stdout=listing, stderr=""
                        )
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"JobId=401 ArrayJobId=401 "
                            f"JobName=LDPF{plan_sha[:4]} "
                            "UserId=nc437(1646707) "
                            "JobState=PENDING Partition=default_partition "
                            "Reason=Dependency RunTime=00:00:00 "
                            f"Comment=SLAD:{plan_sha[:20]}:PREFLIGHT "
                            f"ArrayTaskId={task_range} "
                            f"Dependency={dependency}\n"
                        ),
                        stderr="",
                    )
                return runner

            for label, task_range, dependency in (
                ("range", "0-1", "afterok:100"),
                ("gate", "0-2", "afterok:999"),
                ("multiple controller rows", "0-2", "afterok:100"),
            ):
                runner = runner_with(
                    task_range=task_range, dependency=dependency
                )
                if label == "multiple controller rows":
                    original = runner

                    def runner(command, **kwargs):
                        observed = original(command, **kwargs)
                        if command[0] == tools["scontrol"]["path"]:
                            observed.stdout += observed.stdout
                        return observed
                with self.subTest(label=label), patch(
                    "reconcile_scale_ladder_gate.subprocess.run",
                    side_effect=runner,
                ):
                    with self.assertRaises(ValueError):
                        _discover_held_science_jobs(
                            plan, plan_sha, manifest
                        )

    def test_intended_gate_and_array_are_boundedly_rediscovered(self):
        manifest = {
            "gate_submission_intent": {"comment": "gate"},
            "array_submission_intents": {
                "PREFLIGHT": {"comment": "array"},
            },
        }
        sleeper = Mock()
        with patch(
            "reconcile_scale_ladder_gate._discover_held_science_jobs",
            side_effect=[({}, None), ({"PREFLIGHT": "401"}, "100")],
        ) as discovered:
            arrays, gate = _discover_intended_science_jobs(
                {}, "a" * 64, manifest, sleeper=sleeper
            )
        self.assertEqual(arrays, {"PREFLIGHT": "401"})
        self.assertEqual(gate, "100")
        self.assertEqual(discovered.call_count, 2)
        sleeper.assert_called_once_with(
            ladder.AMBIGUOUS_DISCOVERY_DELAY_S
        )

    def test_unresolved_science_intent_never_resubmits_or_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"campaign_root": str(root), "task_groups": {}}
            raw = ladder.canonical(plan)
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "gate_state": "creating",
                "gate_job_id": None,
                "submitted_arrays": {},
                "gate_submission_intent": {"comment": "gate"},
                "array_submission_intents": {
                    "PREFLIGHT": {"comment": "array"},
                },
            }))
            with patch(
                "reconcile_scale_ladder_gate._discover_intended_science_jobs",
                side_effect=RuntimeError("remains ambiguous"),
            ), patch(
                "reconcile_scale_ladder_gate._sbatch"
            ) as submit_gate, patch(
                "reconcile_scale_ladder_gate._submit_array"
            ) as submit_array, patch(
                "reconcile_scale_ladder_gate._bounded_query"
            ) as release_gate:
                with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                    _reconcile_locked(
                        root, plan_sha,
                        resume_missing_arrays=True,
                        release_held_gate=True,
                    )
            submit_gate.assert_not_called()
            submit_array.assert_not_called()
            release_gate.assert_not_called()

    def test_gate_submission_intent_is_durable_before_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "campaign_root": str(root),
                "task_groups": {group: [group.lower()] for group in (
                    "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
                    "MIP_RAW", "MIP_KNOWN",
                )},
            }
            raw = ladder.canonical(plan)
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "gate_state": "creating",
                "gate_job_id": None,
                "submitted_arrays": {},
            }))

            def interrupted(_plan, _arguments):
                durable = json.loads((root / "campaign.json").read_text())
                self.assertEqual(
                    durable["gate_submission_intent"]["comment"],
                    f"SLADG:{plan_sha[:20]}",
                )
                raise RuntimeError("sbatch interrupted")

            with patch(
                "reconcile_scale_ladder_gate._discover_held_science_jobs",
                return_value=({}, None),
            ), patch(
                "reconcile_scale_ladder_gate._sbatch",
                side_effect=interrupted,
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    _reconcile_locked(
                        root, plan_sha, resume_missing_arrays=True
                    )
            durable = json.loads((root / "campaign.json").read_text())
            self.assertIn("gate_submission_intent", durable)

    def test_array_submission_intent_is_durable_before_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = (
                "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
                "MIP_RAW", "MIP_KNOWN",
            )
            plan = {
                "campaign_root": str(root),
                "task_groups": {group: [group.lower()] for group in groups},
            }
            raw = ladder.canonical(plan)
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "gate_state": "creating",
                "gate_job_id": "100",
                "submitted_arrays": {},
            }))

            def interrupted(
                _plan, _plan_path, _plan_sha, group, *_args, **_kwargs
            ):
                durable = json.loads((root / "campaign.json").read_text())
                self.assertEqual(group, "PREFLIGHT")
                self.assertIn(
                    "PREFLIGHT", durable["array_submission_intents"]
                )
                raise RuntimeError("array sbatch interrupted")

            with patch(
                "reconcile_scale_ladder_gate._discover_held_science_jobs",
                return_value=({}, None),
            ), patch(
                "reconcile_scale_ladder_gate._require_gate_held"
            ), patch(
                "reconcile_scale_ladder_gate._submit_array",
                side_effect=interrupted,
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    _reconcile_locked(
                        root, plan_sha, resume_missing_arrays=True
                    )
            durable = json.loads((root / "campaign.json").read_text())
            self.assertIn(
                "PREFLIGHT", durable["array_submission_intents"]
            )

    def test_gate_completion_requires_exact_zero_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, plan_sha, _manifest, tools = self._held_science_fixture(tmp)
            gate_row = (
                f"100|LDG{plan_sha[:5]}|COMPLETED|default_partition|"
                f"None|SLADG:{plan_sha[:20]}\n"
            )

            def runner_missing_exit(command, **_kwargs):
                if command[0] == tools["squeue"]["path"]:
                    return SimpleNamespace(
                        returncode=0, stdout=gate_row, stderr=""
                    )
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        f"JobId=100 JobName=LDG{plan_sha[:5]} "
                        "JobState=COMPLETED Partition=default_partition "
                        f"Comment=SLADG:{plan_sha[:20]}\n"
                    ),
                    stderr="",
                )

            with self.assertRaisesRegex(ValueError, "exit code"):
                _resolve_gate_state(
                    plan, "100", plan_sha, runner=runner_missing_exit
                )

            def runner_nonzero(command, **_kwargs):
                if command[0] == tools["squeue"]["path"]:
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr=""
                    )
                if command[0] == tools["scontrol"]["path"]:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"JobId=100 JobName=LDG{plan_sha[:5]} "
                            "JobState=COMPLETED Partition=default_partition "
                            f"Comment=SLADG:{plan_sha[:20]} ExitCode=1:0\n"
                        ),
                        stderr="",
                    )
                raise AssertionError(command)

            with self.assertRaisesRegex(ValueError, "nonzero"):
                _resolve_gate_state(
                    plan, "100", plan_sha, runner=runner_nonzero
                )

    def test_activation_runtime_rejects_manifest_fingerprint_spoofs(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan, _plan_sha, _specs, _activation, manifest = (
                self._write_activation_campaign(Path(tmp) / "campaign")
            )
            plan["code_hashes"] = {}
            with patch.object(
                ladder, "_validate_submission_contract"
            ), patch.dict(os.environ, {"SLURM_JOB_ID": "303"}, clear=False), \
                    patch.object(
                        ladder, "_resolve_bound_job",
                        return_value={"state": "RUNNING", "live": True},
                    ):
                self.assertEqual(
                    ladder._require_activation_runtime(plan, manifest), "303"
                )
                for field, value in (
                    ("job_name", "WRONG"),
                    ("partition", "scaglione"),
                    ("comment", "WRONG"),
                ):
                    spoofed = copy.deepcopy(manifest)
                    spoofed["activation"][field] = value
                    with self.subTest(field=field), self.assertRaisesRegex(
                        ValueError, "fingerprint"
                    ):
                        ladder._require_activation_runtime(plan, spoofed)
                spoofed = copy.deepcopy(manifest)
                spoofed["activation"]["job_id"] = "999"
                with self.assertRaisesRegex(ValueError, "runtime identity"):
                    ladder._require_activation_runtime(plan, spoofed)

    def test_failed_probe_activation_has_no_scientific_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            root.mkdir()
            plan = {
                "campaign_root": str(root),
                "submission_protocol": "probe_first_activation_v1",
            }
            raw = ladder.canonical(plan)
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            specs = {
                partition: ladder._probe_spec(
                    plan_sha, partition, root, attempt=1
                )
                for partition in ladder.PROBE_PARTITIONS
            }
            specs["default_partition"]["job_id"] = "301"
            specs["scaglione"]["job_id"] = "302"
            activation = ladder._activation_spec(
                plan_sha, root, specs, attempt=1
            )
            activation["job_id"] = "303"
            manifest = {
                **plan,
                "approval_sha256": plan_sha,
                "submitted": False,
                "submission_state": "activation_released",
                "probe_state": "submitted",
                "reservation_state": "not_created",
                "reservations": [],
                "gate_state": "not_created",
                "gate_job_id": None,
                "submitted_arrays": {},
                "infrastructure_probes": specs,
                "activation": activation,
            }
            (root / "campaign.json").write_text(json.dumps(manifest))
            failed = {
                "default_partition": {"compatible": False, "state": "FAILED"},
                "scaglione": {"compatible": False, "state": "FAILED"},
            }
            with patch.object(
                ladder, "_require_activation_runtime", return_value="303"
            ), patch.object(
                ladder, "_wait_for_probes", return_value=failed
            ), patch.object(
                ladder, "_stage_scientific_inputs"
            ) as stage_inputs, patch.object(
                ladder, "_ensure_reservations"
            ) as reserve, patch.object(
                ladder, "_submit_array"
            ) as submit_array:
                with self.assertRaisesRegex(
                    RuntimeError, "no scientific work"
                ):
                    ladder.activate_existing(root, plan_sha, wait_s=0)
            stage_inputs.assert_not_called()
            reserve.assert_not_called()
            submit_array.assert_not_called()
            observed = json.loads((root / "campaign.json").read_text())
            self.assertEqual(observed["probe_state"], "failed")
            self.assertEqual(observed["reservations"], [])
            self.assertEqual(observed["gate_state"], "not_created")

    def test_probe_observation_deadline_is_not_labeled_probe_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            _plan, plan_sha, _specs, _activation, _manifest = (
                self._write_activation_campaign(root)
            )
            incomplete = {
                "default_partition": {
                    "compatible": False, "state": "ACCOUNTING_ERROR",
                    "state_resolution": "accounting_query_error",
                },
                "scaglione": {
                    "compatible": False, "state": "TIMEOUT_WAITING",
                    "state_resolution": "observer_deadline",
                    "observer_deadline_reached": True,
                },
            }
            with patch.object(
                ladder, "_require_activation_runtime", return_value="303"
            ), patch.object(
                ladder, "_wait_for_probes", return_value=incomplete
            ), patch.object(
                ladder, "_stage_scientific_inputs"
            ) as stage_inputs, patch.object(
                ladder, "_ensure_reservations"
            ) as reserve:
                with self.assertRaisesRegex(
                    RuntimeError, "accounting is incomplete"
                ):
                    ladder.activate_existing(root, plan_sha, wait_s=0)
            stage_inputs.assert_not_called()
            reserve.assert_not_called()
            observed = json.loads((root / "campaign.json").read_text())
            self.assertEqual(
                observed["submission_state"],
                "probe_observation_incomplete",
            )
            self.assertEqual(observed["probe_state"], "observation_incomplete")
            self.assertEqual(observed["gate_state"], "not_created")

    def test_successful_activation_stages_then_resumes_and_releases_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "campaign"
            _plan, plan_sha, specs, _activation, _manifest = (
                self._write_activation_campaign(root)
            )

            def passed_result(partition):
                spec = specs[partition]
                return {
                    "job_id": spec["job_id"],
                    "state": "COMPLETED",
                    "output": spec["output"],
                    "compatible": True,
                    "probe_id": spec["probe_id"],
                    "partition": partition,
                    "attempt": spec["attempt"],
                    "path_bound": True,
                }

            passed = {
                partition: passed_result(partition)
                for partition in ladder.PROBE_PARTITIONS
            }
            events = []

            def stage(_plan):
                events.append("stage")

            def reserve(_plan, observed_sha, activation_lineage):
                self.assertEqual(observed_sha, plan_sha)
                self.assertEqual(
                    activation_lineage,
                    ladder._activation_lineage_id(_plan, plan_sha),
                )
                events.append("reserve")
                return [root / "reservation.json"]

            def reconcile(
                observed_root, observed_sha, *, release_held_gate=False,
                resume_missing_arrays=False, **_kwargs,
            ):
                self.assertEqual(Path(observed_root).resolve(), root.resolve())
                self.assertEqual(observed_sha, plan_sha)
                current = json.loads((root / "campaign.json").read_text())
                if release_held_gate:
                    self.assertTrue(resume_missing_arrays)
                    self.assertEqual(current["gate_state"], "creating")
                    events.append("reconcile_and_release")
                    current.update({
                        "gate_job_id": "900",
                        "gate_state": "release_retry_requested",
                        "submitted_arrays": {
                            group: str(910 + index)
                            for index, group in enumerate((
                                "PREFLIGHT", "SEED", "CG",
                                "CG_SENSITIVITY", "MIP_RAW", "MIP_KNOWN",
                            ))
                        },
                    })
                    return current
                events.append("reconcile_completed")
                current["gate_state"] = "released_reconciled"
                current["submitted"] = True
                return current

            with patch.object(
                ladder, "_require_activation_runtime", return_value="303"
            ), patch.object(
                ladder, "_wait_for_probes", return_value=passed
            ), patch.object(
                ladder, "_stage_scientific_inputs", side_effect=stage
            ), patch.object(
                ladder, "_ensure_reservations", side_effect=reserve
            ), patch(
                "reconcile_scale_ladder_gate._reconcile_locked",
                side_effect=reconcile,
            ), patch(
                "reconcile_scale_ladder_gate._resolve_gate_state",
                return_value={"state": "COMPLETED"},
            ):
                observed = ladder.activate_existing(
                    root, plan_sha, wait_s=1
                )
            self.assertEqual(events, [
                "stage", "reserve", "reconcile_and_release",
                "reconcile_completed",
            ])
            self.assertTrue(observed["submitted"])
            self.assertEqual(observed["gate_state"], "released_reconciled")
            self.assertEqual(observed["activation"]["state"], "complete")
            self.assertEqual(observed["reservations"], [
                str(root / "reservation.json")
            ])

    def test_reservations_resume_across_controllers_with_same_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {
                "reservation_root": tmp,
                "jobs": [{
                    "execution_digest": "d" * 64,
                    "job_key": "cg_k02",
                }],
            }
            lineage = "4" * 64
            first = ladder._ensure_reservations(
                plan, "a" * 64, lineage
            )
            second = ladder._ensure_reservations(
                plan, "a" * 64, lineage
            )
            self.assertEqual(first, second)
            with self.assertRaises(FileExistsError):
                ladder._ensure_reservations(
                    plan, "a" * 64, "5" * 64
                )
            payload = json.loads(first[0].read_text())
            self.assertEqual(payload["activation_lineage_id"], lineage)
            self.assertEqual(
                payload["schema"],
                "evsp-dr-scale-ladder-reservation-v3",
            )
            self.assertEqual(
                payload["submission_protocol"],
                "probe_first_activation_v1",
            )

    def test_cg_portable_identity_does_not_require_gurobi(self):
        observed = environment_identity("cg")
        self.assertEqual(observed["portable_profile"], "cg")
        self.assertNotIn("gurobi", observed["portable"])
        self.assertNotIn(
            "gurobipy_distribution_sha256", observed["portable"]
        )

    def test_portable_environment_reports_exact_required_mismatches(self):
        planned = self._portable_identity()
        for field in (
            "executable_sha256",
            "numpy",
            "scipy_distribution_sha256",
        ):
            observed = copy.deepcopy(planned)
            observed["portable"][field] = "changed"
            differences = compare_portable(planned, observed)
            self.assertEqual(
                [item["field"] for item in differences],
                [f"portable.{field}"],
            )
            self.assertEqual(differences[0]["planned"],
                             planned["portable"][field])
            self.assertEqual(differences[0]["observed"], "changed")
        observed = copy.deepcopy(planned)
        del observed["portable"]["pandas_distribution_sha256"]
        differences = compare_portable(planned, observed)
        self.assertEqual(
            differences[0]["field"],
            "portable.pandas_distribution_sha256",
        )
        self.assertEqual(
            differences[0]["reason"], "missing_required_field"
        )

    def test_failed_probe_never_releases_gate(self):
        compatible = {
            "default_partition": {"compatible": True},
            "scaglione": {"compatible": True},
        }
        for failed in ("default_partition", "scaglione"):
            results = copy.deepcopy(compatible)
            results[failed]["compatible"] = False
            runner = Mock()
            completed = ladder._release_gate_after_probes(
                {"scontrol": {"path": "/approved/scontrol"}},
                "12345",
                results,
                runner=runner,
            )
            self.assertNotEqual(completed.returncode, 0)
            runner.assert_not_called()

    def test_probe_result_binds_job_partition_and_artifact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root, artifacts=True)

            def completed_jobs(command, **_kwargs):
                if command[0] == "/approved/squeue":
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr=""
                    )
                rows = "".join(
                    f"{spec['job_id']}|{spec['job_name']}|COMPLETED|"
                    f"{partition}|{spec['comment']}|0:0\n"
                    for partition, spec in specs.items()
                )
                return SimpleNamespace(
                    returncode=0, stdout=rows, stderr=""
                )
            with patch.object(
                ladder.subprocess,
                "run",
                side_effect=completed_jobs,
            ):
                results = ladder._wait_for_probes(
                    {
                        "squeue": {"path": "/approved/squeue"},
                        "sacct": {"path": "/approved/sacct"},
                        "runtime_environment": {"USER": "nc437"},
                        "python_identity": {
                            "portable_identity_sha256": "a" * 64,
                        },
                        "campaign_root": str(root),
                    },
                    "p" * 64,
                    specs,
                    timeout_s=1,
                )
            self.assertTrue(ladder._probes_compatible(results))
            release = Mock(return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="", stderr=""
            ))
            released = ladder._release_gate_after_probes(
                {"scontrol": {"path": "/approved/scontrol"}},
                "500", results, runner=release,
            )
            self.assertEqual(released.returncode, 0)
            release.assert_called_once_with(
                ["/approved/scontrol", "release", "500"],
                text=True, capture_output=True, check=False,
            )
            bad = json.loads(
                Path(specs["scaglione"]["output"]).read_text()
            )
            bad["slurm_partition"] = "default_partition"
            Path(specs["scaglione"]["output"]).write_text(json.dumps(bad))
            Path(specs["scaglione"]["output"] + ".sha256").write_text(
                f"{sha256_file(Path(specs['scaglione']['output']))}  scaglione.attempt1.json\n"
            )
            with patch.object(
                ladder.subprocess,
                "run",
                side_effect=completed_jobs,
            ):
                results = ladder._wait_for_probes(
                    {
                        "squeue": {"path": "/approved/squeue"},
                        "sacct": {"path": "/approved/sacct"},
                        "runtime_environment": {"USER": "nc437"},
                        "python_identity": {
                            "portable_identity_sha256": "a" * 64,
                        },
                        "campaign_root": str(root),
                    },
                    "p" * 64,
                    specs,
                    timeout_s=1,
                )
            self.assertFalse(ladder._probes_compatible(results))
            spoofed = {
                "default_partition": {
                    "compatible": True, "job_id": "101",
                    "output": specs["default_partition"]["output"],
                },
                "scaglione": {
                    "compatible": True, "job_id": "101",
                    "output": specs["default_partition"]["output"],
                },
            }
            self.assertFalse(ladder._probes_compatible(spoofed))

    def test_live_squeue_precedes_stale_terminal_accounting_and_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root, artifacts=True)
            live_rows = "".join(
                f"{spec['job_id']}|{spec['job_name']}|"
                f"{'PENDING' if partition == 'default_partition' else 'RUNNING'}|"
                f"{partition}|{spec['comment']}\n"
                for partition, spec in specs.items()
            )
            accounting_rows = "".join(
                f"{spec['job_id']}|{spec['job_name']}|"
                f"{'COMPLETED' if partition == 'default_partition' else 'TIMEOUT'}|"
                f"{partition}|{spec['comment']}|"
                f"{'0:0' if partition == 'default_partition' else '1:0'}\n"
                for partition, spec in specs.items()
            )

            def contradictory(command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        live_rows if command[0] == "/approved/squeue"
                        else accounting_rows
                    ),
                    stderr="",
                )

            plan = {
                "squeue": {"path": "/approved/squeue"},
                "sacct": {"path": "/approved/sacct"},
                "runtime_environment": {"USER": "nc437"},
                "python_identity": {
                    "portable_identity_sha256": "a" * 64,
                },
                "campaign_root": str(root),
            }
            observations = ladder._probe_job_states(
                plan, specs, runner=contradictory
            )
            self.assertEqual(
                observations["default_partition"]["state"], "PENDING"
            )
            self.assertEqual(
                observations["scaglione"]["state"], "RUNNING"
            )
            for observed in observations.values():
                self.assertEqual(observed["source"], "squeue")
                self.assertEqual(observed["resolution"], "live")
                self.assertTrue(observed["stale_accounting_conflict"])
                self.assertFalse(observed["terminal"])
            with patch.object(
                ladder.subprocess, "run", side_effect=contradictory
            ):
                results = ladder._wait_for_probes(
                    plan, "p" * 64, specs, timeout_s=0.01
                )
            self.assertFalse(ladder._probes_compatible(results))
            self.assertTrue(ladder._probes_waiting(results))
            for result in results.values():
                self.assertFalse(result["compatible"])
                self.assertTrue(result["observer_deadline_reached"])
                self.assertEqual(result["state_source"], "squeue")
                self.assertIsNotNone(result["artifact_sha256"])
                self.assertNotEqual(result["state"], "TIMEOUT")

    def test_live_probe_fingerprint_mismatch_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root)
            default = specs["default_partition"]
            rows = (
                f"{default['job_id']}|wrong-name|PENDING|"
                f"default_partition|{default['comment']}\n"
            )

            def runner(command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout=rows if command[0] == "/approved/squeue" else "",
                    stderr="",
                )

            observed = ladder._probe_job_states({
                "squeue": {"path": "/approved/squeue"},
                "sacct": {"path": "/approved/sacct"},
                "runtime_environment": {"USER": "nc437"},
            }, specs, runner=runner)
            failure = observed["default_partition"]
            self.assertEqual(failure["resolution"], "identity_mismatch")
            self.assertTrue(failure["terminal"])
            self.assertIn(
                "job_name",
                {error["field"] for error in failure["identity_errors"]},
            )

    def test_accounting_requires_fingerprint_and_zero_completed_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root)
            default = specs["default_partition"]
            scaglione = specs["scaglione"]
            accounting = (
                f"{default['job_id']}|{default['job_name']}|COMPLETED|"
                f"default_partition|wrong-comment|0:0\n"
                f"{scaglione['job_id']}|{scaglione['job_name']}|COMPLETED|"
                f"scaglione|{scaglione['comment']}|1:0\n"
            )

            def runner(command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout="" if command[0] == "/approved/squeue"
                    else accounting,
                    stderr="",
                )

            observed = ladder._probe_job_states({
                "squeue": {"path": "/approved/squeue"},
                "sacct": {"path": "/approved/sacct"},
                "runtime_environment": {"USER": "nc437"},
            }, specs, runner=runner)
            self.assertEqual(
                observed["default_partition"]["state"],
                "ACCOUNTING_IDENTITY_MISMATCH",
            )
            self.assertEqual(
                observed["scaglione"]["state"],
                "ACCOUNTING_OUTCOME_MISMATCH",
            )
            self.assertTrue(all(
                value["resolution"] == "identity_mismatch"
                for value in observed.values()
            ))

    def test_failed_accounting_requires_an_exact_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = self._probe_specs(Path(tmp))
            rows = "".join(
                f"{spec['job_id']}|{spec['job_name']}|FAILED|"
                f"{partition}|{spec['comment']}|"
                f"{'missing' if partition == 'default_partition' else ''}\n"
                for partition, spec in specs.items()
            )

            def runner(command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout="" if command[0] == "/approved/squeue" else rows,
                    stderr="",
                )

            observed = ladder._probe_job_states({
                "squeue": {"path": "/approved/squeue"},
                "sacct": {"path": "/approved/sacct"},
                "runtime_environment": {"USER": "nc437"},
            }, specs, runner=runner)
            for value in observed.values():
                self.assertEqual(
                    value["state"], "ACCOUNTING_OUTCOME_MISMATCH"
                )
                self.assertEqual(value["resolution"], "identity_mismatch")

    def test_squeue_query_error_never_falls_through_to_sacct(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = self._probe_specs(Path(tmp))
            calls = []

            def runner(command, **_kwargs):
                calls.append(command)
                if command[0] == "/approved/squeue":
                    return SimpleNamespace(
                        returncode=1, stdout="", stderr="controller down"
                    )
                raise AssertionError("sacct must not be trusted without squeue")

            observed = ladder._probe_job_states({
                "squeue": {"path": "/approved/squeue"},
                "sacct": {"path": "/approved/sacct"},
                "runtime_environment": {"USER": "nc437"},
            }, specs, runner=runner)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("-j", calls[0])
            self.assertEqual(
                calls[0][calls[0].index("-u") + 1], "nc437"
            )
            for value in observed.values():
                self.assertEqual(
                    value["resolution"], "controller_query_error"
                )
                self.assertFalse(value["terminal"])

    def test_probe_scheduler_queries_are_bounded_and_timeout_is_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = self._probe_specs(Path(tmp))
            timeouts = []

            def controller_timeout(command, **kwargs):
                timeouts.append(kwargs.get("timeout"))
                raise subprocess.TimeoutExpired(
                    command, kwargs.get("timeout")
                )

            plan = {
                "squeue": {"path": "/approved/squeue"},
                "sacct": {"path": "/approved/sacct"},
                "runtime_environment": {"USER": "nc437"},
            }
            observed = ladder._probe_job_states(
                plan, specs, runner=controller_timeout
            )
            self.assertEqual(len(timeouts), 1)
            self.assertGreater(timeouts[0], 0)
            self.assertLessEqual(
                timeouts[0], ladder.PROBE_SLURM_QUERY_TIMEOUT_S
            )
            for value in observed.values():
                self.assertEqual(
                    value["resolution"], "controller_query_error"
                )
                self.assertEqual(value["state"], "CONTROLLER_QUERY_TIMEOUT")
                self.assertFalse(value["terminal"])

            def accounting_timeout(command, **kwargs):
                timeouts.append(kwargs.get("timeout"))
                if command[0] == "/approved/squeue":
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr=""
                    )
                raise subprocess.TimeoutExpired(
                    command, kwargs.get("timeout")
                )

            observed = ladder._probe_job_states(
                plan, specs, runner=accounting_timeout
            )
            self.assertTrue(all(timeout is not None for timeout in timeouts))
            for value in observed.values():
                self.assertEqual(
                    value["resolution"], "accounting_query_error"
                )
                self.assertFalse(value["terminal"])
                self.assertIn("timed out", value["query_error"])

    def test_observer_deadline_is_waiting_not_scheduler_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root)
            live_rows = "".join(
                f"{spec['job_id']}|{spec['job_name']}|PENDING|"
                f"{partition}|{spec['comment']}\n"
                for partition, spec in specs.items()
            )

            def pending(command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout=live_rows
                    if command[0] == "/approved/squeue" else "",
                    stderr="",
                )

            with patch.object(
                ladder.subprocess, "run", side_effect=pending
            ):
                results = ladder._wait_for_probes({
                    "squeue": {"path": "/approved/squeue"},
                    "sacct": {"path": "/approved/sacct"},
                    "runtime_environment": {"USER": "nc437"},
                    "campaign_root": str(root),
                }, "p" * 64, specs, timeout_s=0.01)
            self.assertTrue(ladder._probes_waiting(results))
            for result in results.values():
                self.assertEqual(result["state"], "PENDING")
                self.assertEqual(result["state_resolution"], "live")
                self.assertTrue(result["observer_deadline_reached"])
                self.assertNotIn(result["state"], ladder.PROBE_TERMINAL_STATES)

    def test_expired_observer_deadline_runs_no_scheduler_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root)
            with patch.object(ladder.subprocess, "run") as runner:
                results = ladder._wait_for_probes({
                    "squeue": {"path": "/approved/squeue"},
                    "sacct": {"path": "/approved/sacct"},
                    "runtime_environment": {"USER": "nc437"},
                    "campaign_root": str(root),
                }, "p" * 64, specs, timeout_s=0)
            runner.assert_not_called()
            self.assertTrue(ladder._probes_waiting(results))
            for result in results.values():
                self.assertEqual(result["state"], "OBSERVER_DEADLINE")
                self.assertEqual(
                    result["state_resolution"], "observer_deadline"
                )
                self.assertTrue(result["observer_deadline_reached"])

    def test_completed_probe_awaits_atomic_artifact_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root, artifacts=True)
            Path(
                specs["scaglione"]["output"] + ".sha256"
            ).unlink()
            accounting = "".join(
                f"{spec['job_id']}|{spec['job_name']}|COMPLETED|"
                f"{partition}|{spec['comment']}|0:0\n"
                for partition, spec in specs.items()
            )

            def completed(command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout="" if command[0] == "/approved/squeue"
                    else accounting,
                    stderr="",
                )

            with patch.object(
                ladder.subprocess, "run", side_effect=completed
            ):
                results = ladder._wait_for_probes({
                    "squeue": {"path": "/approved/squeue"},
                    "sacct": {"path": "/approved/sacct"},
                    "runtime_environment": {"USER": "nc437"},
                    "python_identity": {
                        "portable_identity_sha256": "a" * 64,
                    },
                    "campaign_root": str(root),
                }, "p" * 64, specs, timeout_s=0.01)
            self.assertTrue(results["default_partition"]["compatible"])
            waiting = results["scaglione"]
            self.assertEqual(waiting["artifact_status"], "awaiting_sidecar")
            self.assertEqual(
                waiting["state_resolution"], "awaiting_artifact"
            )
            self.assertTrue(waiting["observer_deadline_reached"])
            self.assertTrue(ladder._probes_waiting(results))

    def test_artifact_portable_identity_mismatch_is_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root, artifacts=True)
            output = Path(specs["default_partition"]["output"])
            payload = json.loads(output.read_text())
            payload["observed_portable_identity_sha256"] = "x" * 64
            output.write_text(json.dumps(payload))
            Path(str(output) + ".sha256").write_text(
                f"{sha256_file(output)}  {output.name}\n"
            )
            artifact = ladder._probe_artifact_observation(
                {
                    "campaign_root": str(root),
                    "python_identity": {
                        "portable_identity_sha256": "a" * 64,
                    },
                },
                "p" * 64,
                "default_partition",
                specs["default_partition"],
            )
            self.assertEqual(artifact["status"], "invalid")
            self.assertIn(
                "observed_portable_identity_sha256",
                {
                    error["field"]
                    for error in artifact["artifact_identity_errors"]
                },
            )

    def test_terminal_environment_mismatch_is_not_retryable_infrastructure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs = self._probe_specs(root, artifacts=True)
            output = Path(specs["scaglione"]["output"])
            payload = json.loads(output.read_text())
            payload["compatible"] = False
            payload["differences"] = [{
                "field": "portable.numpy", "reason": "value_mismatch",
            }]
            output.write_text(json.dumps(payload))
            Path(str(output) + ".sha256").write_text(
                f"{sha256_file(output)}  {output.name}\n"
            )
            accounting = "".join(
                f"{spec['job_id']}|{spec['job_name']}|"
                f"{'FAILED' if partition == 'scaglione' else 'COMPLETED'}|"
                f"{partition}|{spec['comment']}|"
                f"{'3:0' if partition == 'scaglione' else '0:0'}\n"
                for partition, spec in specs.items()
            )

            def completed(command, **_kwargs):
                return SimpleNamespace(
                    returncode=0,
                    stdout="" if command[0] == "/approved/squeue"
                    else accounting,
                    stderr="",
                )

            with patch.object(
                ladder.subprocess, "run", side_effect=completed
            ):
                results = ladder._wait_for_probes({
                    "squeue": {"path": "/approved/squeue"},
                    "sacct": {"path": "/approved/sacct"},
                    "runtime_environment": {"USER": "nc437"},
                    "python_identity": {
                        "portable_identity_sha256": "a" * 64,
                    },
                    "campaign_root": str(root),
                }, "p" * 64, specs, timeout_s=0.01)
            mismatch = results["scaglione"]
            self.assertEqual(
                mismatch["state_resolution"], "environment_mismatch"
            )
            self.assertFalse(mismatch["compatible"])
            self.assertFalse(ladder._probe_result_waiting(mismatch))

    def test_probe_attempts_have_distinct_bound_outputs_and_comments(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = ladder._probe_spec(
                "a" * 64, "default_partition", Path(tmp), 1
            )
            second = ladder._probe_spec(
                "a" * 64, "default_partition", Path(tmp), 2
            )
            self.assertNotEqual(first["output"], second["output"])
            self.assertNotEqual(first["comment"], second["comment"])
            self.assertEqual(first["attempt"], 1)
            self.assertEqual(second["attempt"], 2)

    def test_probe_submission_binds_attempt_comment_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = ladder._probe_spec(
                "d" * 64, "scaglione", root, 2
            )
            plan = {
                "campaign_root": str(root),
                "python": {"path": "/approved/python", "sha256": "p"},
                "runtime_environment": {"HOME": "/approved/home"},
                "probe_worker_sha256": "w",
            }
            with patch.object(
                ladder, "_sbatch", return_value="9876"
            ) as submitted:
                job_id = ladder._submit_probe(
                    plan, root / "approved-plan.json", "d" * 64,
                    spec, root / "logs",
                )
            self.assertEqual(job_id, "9876")
            arguments = submitted.call_args.args[1]
            self.assertIn(f"--comment={spec['comment']}", arguments)
            self.assertIn(spec["output"], arguments)
            self.assertIn("2", arguments)

    def test_unrecorded_probe_is_recovered_from_bound_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = ladder._probe_spec(
                "b" * 64, "scaglione", Path(tmp), 1
            )
            output = Path(spec["output"])
            output.parent.mkdir()
            output.write_text(json.dumps({
                "plan_sha256": "b" * 64,
                "probe_id": "scaglione",
                "probe_attempt": 1,
                "slurm_job_id": "4321",
                "slurm_partition": "scaglione",
            }))
            recovered = _discover_probe_job(
                {}, "b" * 64, "scaglione", spec
            )
            self.assertEqual(recovered, "4321")

    def test_unrecorded_probe_is_recovered_from_unique_slurm_comment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            squeue = root / "squeue"
            squeue.write_text("approved bytes")
            spec = ladder._probe_spec(
                "c" * 64, "default_partition", root, 1
            )
            plan = {
                "squeue": {
                    "available": True,
                    "path": str(squeue),
                    "sha256": sha256_file(squeue),
                },
            }
            with patch(
                "reconcile_scale_ladder_gate.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout=f"7654|{spec['comment']}\n",
                    stderr="",
                ),
            ) as queried:
                recovered = _discover_probe_job(
                    plan, "c" * 64, "default_partition", spec
                )
            self.assertEqual(recovered, "7654")
            self.assertEqual(
                queried.call_args.kwargs["timeout"],
                10.0,
            )

    def test_real_environment_mismatch_is_not_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "probe.json"
            output.write_text(json.dumps({
                "compatible": False,
                "differences": [{"field": "portable.numpy"}],
            }))
            self.assertTrue(_hard_probe_mismatch({"output": str(output)}))
            output.write_text(json.dumps({
                "compatible": False,
                "differences": [],
            }))
            self.assertFalse(_hard_probe_mismatch({"output": str(output)}))

    def test_legacy_probe_spec_is_rejected_with_clear_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = {}
            for name in ("sacct", "squeue", "scontrol"):
                path = root / name
                path.write_text(name)
                tools[name] = {
                    "available": True,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            plan_raw = json.dumps({
                "campaign_root": str(root), **tools,
            }, sort_keys=True, separators=(",", ":")).encode()
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            (root / "approved-plan.json").write_bytes(plan_raw)
            groups = (
                "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
                "MIP_RAW", "MIP_KNOWN",
            )
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "gate_state": "release_attempting",
                "gate_job_id": "100",
                "submitted_arrays": {
                    group: str(400 + index)
                    for index, group in enumerate(groups)
                },
                "infrastructure_probes": {
                    "default_partition": {
                        "job_id": "201",
                        "output": str(root / "probes/default_partition.json"),
                        "probe_id": "default",
                        "partition": "default_partition",
                    },
                    "scaglione": {
                        "job_id": "202",
                        "output": str(root / "probes/scaglione.json"),
                        "probe_id": "scaglione",
                        "partition": "scaglione",
                    },
                },
            }))
            with patch(
                "reconcile_scale_ladder_gate.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0, stdout="100|PENDING\n", stderr=""
                ),
            ):
                with self.assertRaisesRegex(
                    ValueError, "predates attempt-safe recovery"
                ):
                    reconcile_gate(root, plan_sha)

    def test_reconcile_retries_preempted_probe_and_keeps_gate_held(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            probes = root / "probes"
            probes.mkdir()
            tools = {}
            for name in ("sacct", "squeue", "scontrol"):
                path = root / name
                path.write_text(name)
                tools[name] = {
                    "available": True,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            groups = (
                "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
                "MIP_RAW", "MIP_KNOWN",
            )
            plan = {
                "campaign_root": str(root),
                "runtime_environment": {"USER": "nc437"},
                "task_groups": {
                    group: [f"job_{group.lower()}"] for group in groups
                },
                **tools,
            }
            plan_raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            (root / "approved-plan.json").write_bytes(plan_raw)
            specs = {
                partition: ladder._probe_spec(
                    plan_sha, partition, root, 1
                )
                for partition in ladder.PROBE_PARTITIONS
            }
            specs["default_partition"]["job_id"] = "201"
            specs["scaglione"]["job_id"] = "202"
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "gate_state": "held_probe_failure",
                "gate_job_id": "100",
                "submitted_arrays": {
                    group: str(300 + index)
                    for index, group in enumerate(groups)
                },
                "infrastructure_probes": specs,
            }))

            def result(spec, compatible, state):
                return {
                    "job_id": spec["job_id"],
                    "state": state,
                    "output": spec["output"],
                    "compatible": compatible,
                    "probe_id": spec["probe_id"],
                    "partition": spec["partition"],
                    "attempt": spec["attempt"],
                    "path_bound": True,
                }

            failed = {
                "default_partition": result(
                    specs["default_partition"], True, "COMPLETED"
                ),
                "scaglione": result(
                    specs["scaglione"], False, "PREEMPTED"
                ),
            }
            retried_spec = ladder._probe_spec(
                plan_sha, "scaglione", root, 2
            )
            retried_spec["job_id"] = "303"
            passed = {
                "default_partition": failed["default_partition"],
                "scaglione": result(
                    retried_spec, True, "COMPLETED"
                ),
            }
            array_ids = {
                group: str(300 + index)
                for index, group in enumerate(groups)
            }

            def dependency_for(group):
                values = ["afterok:100"]
                if group in {"CG", "CG_SENSITIVITY"}:
                    values.append(
                        f"afterok:{array_ids['PREFLIGHT']}_*"
                    )
                elif group == "MIP_RAW":
                    values.append(f"aftercorr:{array_ids['CG']}_*")
                elif group == "MIP_KNOWN":
                    values.append(
                        f"aftercorr:{array_ids['CG']}_*:"
                        f"{array_ids['SEED']}_*"
                    )
                return ",".join(values)

            def fake_run(command, **_kwargs):
                if str(command[0]) == tools["squeue"]["path"]:
                    array_rows = "".join(
                        f"{array_ids[group]}|"
                        f"{ladder._array_name(group, plan_sha)}|PENDING|"
                        f"{'scaglione' if group.startswith('MIP') else 'default_partition'}|"
                        f"Dependency|SLAD:{plan_sha[:20]}:{group}\n"
                        for group in groups
                    )
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"100|LDG{plan_sha[:5]}|PENDING|"
                            "default_partition|JobHeldUser|"
                            f"SLADG:{plan_sha[:20]}\n"
                            f"{array_rows}"
                        ),
                        stderr="",
                    )
                if (
                    str(command[0]) == tools["scontrol"]["path"]
                    and command[1:3] == ["show", "job"]
                ):
                    job_id = command[3]
                    group = next(
                        key for key, value in array_ids.items()
                        if value == job_id
                    )
                    partition = (
                        "scaglione" if group.startswith("MIP")
                        else "default_partition"
                    )
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"JobId={job_id} ArrayJobId={job_id} "
                            "UserId=nc437(1646707) "
                            f"JobName={ladder._array_name(group, plan_sha)} "
                            f"JobState=PENDING Partition={partition} "
                            "Reason=Dependency RunTime=00:00:00 "
                            f"Comment=SLAD:{plan_sha[:20]}:{group} "
                            "ArrayTaskId=0-0 "
                            f"Dependency={dependency_for(group)}\n"
                        ),
                        stderr="",
                    )
                raise AssertionError(f"unexpected command: {command}")

            with patch(
                "reconcile_scale_ladder_gate.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "reconcile_scale_ladder_gate._wait_for_probes",
                side_effect=[failed, passed],
            ), patch(
                "reconcile_scale_ladder_gate._submit_probe",
                return_value="303",
            ) as submitted:
                reconciled = reconcile_gate(
                    root, plan_sha, retry_failed_probes=True
                )
            self.assertEqual(
                reconciled["gate_state"], "held_probe_passed"
            )
            self.assertEqual(
                reconciled["infrastructure_probes"]["scaglione"]
                ["attempt"],
                2,
            )
            self.assertEqual(
                reconciled["infrastructure_probes"]["scaglione"]
                ["job_id"],
                "303",
            )
            self.assertEqual(
                reconciled["probe_attempt_history"]["scaglione"]
                [0]["result"]["state"],
                "PREEMPTED",
            )
            submitted.assert_called_once()

    def test_reconcile_live_held_gate_precedes_stale_completed_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            tools = {}
            for name in ("sacct", "squeue", "scontrol"):
                path = root / name
                path.write_text(name)
                tools[name] = {
                    "available": True,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            plan = {
                "campaign_root": str(root),
                "runtime_environment": {"USER": "nc437"},
                **tools,
            }
            plan_raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            (root / "approved-plan.json").write_bytes(plan_raw)
            specs = self._probe_specs(root, plan_sha=plan_sha)
            groups = (
                "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
                "MIP_RAW", "MIP_KNOWN",
            )
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "gate_state": "release_attempting",
                "gate_job_id": "100",
                "submitted_arrays": {
                    group: str(500 + index)
                    for index, group in enumerate(groups)
                },
                "infrastructure_probes": specs,
            }))
            sacct_calls = []
            release_calls = []

            def fake_run(command, **_kwargs):
                executable = str(command[0])
                if executable == tools["squeue"]["path"]:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"100|LDG{plan_sha[:5]}|PENDING|"
                            "default_partition|JobHeldUser|"
                            f"SLADG:{plan_sha[:20]}\n"
                        ),
                        stderr="",
                    )
                if executable == tools["sacct"]["path"]:
                    sacct_calls.append(command)
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"100|LDG{plan_sha[:5]}|COMPLETED|"
                            "default_partition|"
                            f"SLADG:{plan_sha[:20]}|0:0\n"
                        ),
                        stderr="",
                    )
                if (
                    executable == tools["scontrol"]["path"]
                    and command[1:] == ["release", "100"]
                ):
                    release_calls.append((command, _kwargs))
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr=""
                    )
                raise AssertionError(f"unexpected command: {command}")

            with patch(
                "reconcile_scale_ladder_gate.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "reconcile_scale_ladder_gate._wait_for_probes",
                return_value={"probe": "passed"},
            ), patch(
                "reconcile_scale_ladder_gate._probes_compatible",
                return_value=True,
            ):
                reconciled = reconcile_gate(
                    root, plan_sha, release_held_gate=True
                )
            self.assertEqual(sacct_calls, [])
            self.assertEqual(len(release_calls), 1)
            self.assertEqual(
                release_calls[0][1]["timeout"], 10.0
            )
            self.assertEqual(
                reconciled["gate_state"], "release_retry_requested"
            )
            self.assertIsNot(reconciled.get("submitted"), True)

    def test_reconcile_recovers_partial_gate_and_array_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "logs").mkdir()
            tools = {}
            for name in ("sacct", "squeue", "scontrol"):
                path = root / name
                path.write_text(name)
                tools[name] = {
                    "available": True,
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            plan = {
                "campaign_root": str(root),
                "submission_protocol": "probe_first_activation_v1",
                "runtime_environment": {"USER": "nc437"},
                "task_groups": {
                    group: [f"job_{group.lower()}"] for group in (
                        "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
                        "MIP_RAW", "MIP_KNOWN",
                    )
                },
                **tools,
            }
            plan_raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            (root / "approved-plan.json").write_bytes(plan_raw)
            specs = self._probe_specs(root, plan_sha=plan_sha)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "submitted": False,
                "gate_state": "creating",
                "gate_job_id": None,
                "submitted_arrays": {},
                "infrastructure_probes": specs,
            }))

            def passed_result(partition):
                spec = specs[partition]
                return {
                    "job_id": spec["job_id"],
                    "state": "COMPLETED",
                    "output": spec["output"],
                    "compatible": True,
                    "probe_id": spec["probe_id"],
                    "partition": partition,
                    "attempt": spec["attempt"],
                    "path_bound": True,
                }

            passed = {
                partition: passed_result(partition)
                for partition in ladder.PROBE_PARTITIONS
            }

            def fake_run(command, **_kwargs):
                executable = str(command[0])
                if executable == tools["squeue"]["path"]:
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"900|LDG{plan_sha[:5]}|PENDING|"
                            "default_partition|JobHeldUser|"
                            f"SLADG:{plan_sha[:20]}\n"
                            f"401|LDPF{plan_sha[:4]}|PENDING|"
                            "default_partition|Dependency|"
                            f"SLAD:{plan_sha[:20]}:PREFLIGHT\n"
                        ),
                        stderr="",
                    )
                if executable == tools["sacct"]["path"]:
                    return SimpleNamespace(
                        returncode=0, stdout="", stderr=""
                    )
                if (
                    executable == tools["scontrol"]["path"]
                    and command[1:] == ["show", "job", "401", "-o"]
                ):
                    return SimpleNamespace(
                        returncode=0,
                        stdout=(
                            f"JobId=401 ArrayJobId=401 "
                            f"JobName=LDPF{plan_sha[:4]} "
                            "UserId=nc437(1646707) "
                            "JobState=PENDING Partition=default_partition "
                            "Reason=Dependency RunTime=00:00:00 "
                            f"Comment=SLAD:{plan_sha[:20]}:PREFLIGHT "
                            "ArrayTaskId=0-0 "
                            "Dependency=afterok:900(unfulfilled)\n"
                        ),
                        stderr="",
                    )
                raise AssertionError(f"unexpected command: {command}")

            with patch(
                "reconcile_scale_ladder_gate.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "reconcile_scale_ladder_gate._wait_for_probes",
                return_value=passed,
            ), patch(
                "reconcile_scale_ladder_gate._submit_array",
                side_effect=["402", "403", "404", "405", "406"],
            ) as submit_array:
                observed = reconcile_gate(
                    root, plan_sha, resume_missing_arrays=True
                )
            self.assertEqual(observed["gate_job_id"], "900")
            self.assertEqual(observed["gate_state"], "held_probe_passed")
            self.assertEqual(observed["submitted_arrays"], {
                "PREFLIGHT": "401",
                "SEED": "402",
                "CG": "403",
                "CG_SENSITIVITY": "404",
                "MIP_RAW": "405",
                "MIP_KNOWN": "406",
            })
            self.assertEqual(
                [call.args[3] for call in submit_array.call_args_list],
                ["SEED", "CG", "CG_SENSITIVITY", "MIP_RAW", "MIP_KNOWN"],
            )
            self.assertEqual(
                submit_array.call_args_list[1].kwargs["dependency"],
                "afterok:401",
            )
            self.assertEqual(
                submit_array.call_args_list[3].kwargs["dependency"],
                "aftercorr:403",
            )
            self.assertEqual(
                submit_array.call_args_list[4].kwargs["dependency"],
                "aftercorr:403:402",
            )

    def test_atomic_summary_publication_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            (source / "value.txt").write_text("new")
            _rename_noreplace(source, target)
            self.assertFalse(source.exists())
            self.assertEqual((target / "value.txt").read_text(), "new")

            second = root / "second"
            second.mkdir()
            with self.assertRaises(FileExistsError):
                _rename_noreplace(second, target)
            self.assertTrue(second.exists())
            self.assertEqual((target / "value.txt").read_text(), "new")

    def test_darwin_renamex_dispatch_and_unsupported_platform(self):
        class FakeFunction:
            argtypes = None
            restype = None

            def __call__(self, source, target, flags):
                self.flags = flags
                source_path = Path(os.fsdecode(source))
                target_path = Path(os.fsdecode(target))
                try:
                    os.link(source_path, target_path)
                except FileExistsError:
                    ctypes.set_errno(errno.EEXIST)
                    return -1
                source_path.unlink()
                return 0

        class FakeLib:
            renamex_np = FakeFunction()

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            target = Path(tmp) / "target"
            source.write_text("value")
            fake = FakeLib()
            _rename_noreplace(
                source, target, platform="darwin", libc=fake
            )
            self.assertEqual(fake.renamex_np.flags, 0x00000004)
            self.assertEqual(target.read_text(), "value")
            another = Path(tmp) / "another"
            another.write_text("other")
            with self.assertRaises(FileExistsError):
                _rename_noreplace(
                    another, target, platform="darwin", libc=fake
                )
            with self.assertRaisesRegex(OSError, "unsupported"):
                _rename_noreplace(
                    another, Path(tmp) / "x",
                    platform="freebsd", libc=fake,
                )

    def test_manifest_has_exact_cells_and_identity_domains(self):
        with INSTANCE_MANIFEST.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 22)
        counts = {}
        for row in rows:
            counts[int(row["scale"])] = counts.get(
                int(row["scale"]), 0
            ) + 1
            self.assertEqual(
                row["trip_identity_schema"], "evsp-dr-trip-identity-v1"
            )
            self.assertNotIn("trip_set_sha256", row)
            observed = identity(REPO_ROOT / row["relative_path"])
            for field in (
                "ordered_trip_id_set_sha256",
                "solver_local_trip_index_sha256",
                "ordered_trip_sequence_sha256",
                "instance_file_sha256",
            ):
                self.assertEqual(row[field], observed[field])
            duties = json.loads(row["duties_json"])
            bases = [
                "13316" if duty in {"13316m", "13316uwt"}
                else "13324" if duty in {"13324muw", "13324t"}
                else duty
                for duty in duties
            ]
            self.assertEqual(len(bases), len(set(bases)))
        self.assertEqual(
            counts, {2: 3, 3: 3, 5: 3, 8: 3, 13: 3, 20: 3, 30: 3, 40: 1}
        )

    def test_k40_hash_domains_are_explicit_and_not_cross_compared(self):
        k40 = (
            REPO_ROOT
            / "data/tariff_response/frozen_instances/"
            "Practice_Custom_DutyUnion_k40_r2.csv"
        )
        observed = identity(k40)
        self.assertEqual(
            observed["instance_file_sha256"],
            "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd",
        )
        self.assertEqual(
            observed["ordered_trip_id_set_sha256"],
            "2756baf18e81509df74a2e92e3925aefd17fdf81d95ded9161ad113fd6830e50",
        )
        self.assertEqual(
            observed["solver_local_trip_index_sha256"],
            "35604b22facf1646963e85eb98a858906f0dd7dbebd86ea0d3ac7b797de62ed0",
        )
        for field in (
            "ordered_trip_id_set_sha256",
            "solver_local_trip_index_sha256",
        ):
            classified = classify_legacy_trip_hash(
                observed[field], observed
            )
            self.assertEqual(classified["legacy_trip_hash_field"], field)
        changed = dict(observed)
        changed["ordered_trip_id_set_sha256"] = observed[
            "solver_local_trip_index_sha256"
        ]
        with self.assertRaisesRegex(ValueError, "domain"):
            require_compatible(
                observed, changed, "ordered_trip_id_set_sha256"
            )

    def test_deterministic_regeneration_matches_all_instance_hashes(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            manifest, _campaign, generated = build_inputs(Path(tmp) / "out")
            self.assertTrue(manifest.is_file())
            with INSTANCE_MANIFEST.open(newline="") as handle:
                reviewed = {
                    (int(row["scale"]), int(row["selection_replicate"])):
                        row["instance_file_sha256"]
                    for row in csv.DictReader(handle)
                }
            self.assertEqual(
                {
                    (int(row["scale"]), int(row["selection_replicate"])):
                        row["instance_file_sha256"]
                    for row in generated
                },
                reviewed,
            )

    def test_plan_has_exact_task_mapping_and_no_k40_mips(self):
        environment = {
            "schema": "evsp-dr-portable-environment-v1",
            "portable": {
                "python": "3.12.3",
                "executable": str(Path(sys.executable).resolve()),
                "executable_sha256":
                    sha256_file(Path(sys.executable).resolve()),
            },
            "portable_identity_sha256": "e" * 64,
            "node_metadata": {},
        }
        checkout = {
            "commit": "a" * 40,
            "reviewed_base": ladder.REVIEWED_BASE,
            "detached": True,
            "tracked_clean": True,
        }
        with (
            patch.object(ladder, "_environment", return_value=environment),
            patch.object(ladder, "checkout_identity", return_value=checkout),
            patch.object(ladder.shutil, "which", return_value="/usr/bin/true"),
        ):
            plan = ladder.build_plan(
                "ladder-test", Path(sys.executable), Path("/tmp/reservations")
            )
        self.assertEqual(plan["task_count"], 138)
        self.assertEqual(plan["preflight_task_count"], 22)
        self.assertEqual(plan["cg_task_count"], 23)
        self.assertEqual(plan["sensitivity_cg_task_count"], 30)
        self.assertEqual(plan["mip_task_count"], 42)
        self.assertEqual(plan["k40_mip_submission_count"], 0)
        self.assertEqual(plan["infrastructure_probe_task_count"], 2)
        self.assertEqual(plan["infrastructure_activation_task_count"], 1)
        self.assertEqual(plan["infrastructure_task_count"], 3)
        self.assertEqual(
            {key: len(value) for key, value in plan["task_groups"].items()},
            {
                "PREFLIGHT": 22, "SEED": 21, "CG": 23,
                "CG_SENSITIVITY": 30,
                "MIP_RAW": 21, "MIP_KNOWN": 21,
            },
        )
        self.assertFalse(any(
            job["phase"] == "MIP" and job["scale"] == 40
            for job in plan["jobs"]
        ))
        self.assertTrue(all(
            len(job["job_name"]) <= 15 for job in plan["jobs"]
        ))
        self.assertEqual(
            len({job["job_name"] for job in plan["jobs"]}),
            len(plan["jobs"]),
        )
        self.assertEqual(
            plan["tariff"]["primary_tariff_sha256"],
            ladder.HISTORICAL_FLAT_SHA256,
        )
        self.assertNotIn("alpha", json.dumps(plan).lower())
        self.assertEqual(len(plan["k40_reuse_slots"]), 4)
        fallback = [
            job for job in plan["jobs"]
            if job["phase"] == "CG_SENSITIVITY"
            and job["soc_step"] == 1.0 and job["block_min"] == 5
        ]
        self.assertEqual(len(fallback), 3)
        self.assertTrue(all(
            job["scale"] == 2 and job["diagnostic_only"] is True
            for job in fallback
        ))
        self.assertTrue(all(
            job["telemetry"] is None
            for job in plan["jobs"]
            if job["phase"] == "CG" and job["scale"] >= 30
        ))
        self.assertTrue(all(
            job["telemetry"] is not None
            for job in plan["jobs"]
            if job["phase"] == "CG" and job["scale"] <= 20
        ))
        self.assertIn(1440, next(
            job["snapshot_minutes"] for job in plan["jobs"]
            if job["phase"] == "CG" and job["scale"] == 40
        ))

    def test_operator_wrapper_matches_probe_first_contract(self):
        wrapper = (
            REPO_ROOT / "scripts/launch_scale_ladder_probe_first.sh"
        ).read_text()
        retired = (
            REPO_ROOT / "scripts/launch_scale_ladder_ba09d46.sh"
        ).read_text()
        self.assertIn('CAMPAIGN=${LADDER_CAMPAIGN:-}', wrapper)
        self.assertNotIn('CAMPAIGN="slad_${STAMP}', wrapper)
        self.assertIn('.infrastructure_task_count == 3', wrapper)
        self.assertIn('--retry-failed-probes', wrapper)
        self.assertIn('--retry-failed-activation', wrapper)
        self.assertIn('--approved-plan-sha256 "$FILE_SHA" --submit', wrapper)
        self.assertNotIn('ALREADY_ARMED', wrapper)
        self.assertNotIn(
            '.submitted == true and .gate_state == "released"', wrapper
        )
        self.assertIn('INFRASTRUCTURE_ARMED=true', wrapper)
        self.assertIn('SCIENCE_ONLY_AFTER_BOTH_PROBES_VALIDATE=true', wrapper)
        self.assertIn('This launcher is retired', retired)

    def test_stalled_campaign_replacement_is_exact_and_fail_closed(self):
        script = (
            REPO_ROOT / "scripts/replace_stalled_scale_ladder_20260819.sh"
        )
        text = script.read_text()
        parsed = subprocess.run(
            ["bash", "-n", str(script)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        self.assertIn(
            "bcea6b9391cf49515de2dc8ae06ee6bdc70186fde2c80243a2eeaec62b2c083b",
            text,
        )
        self.assertIn(
            "ba09d4602ded98f9c9157f52af169ef511b5abf7", text
        )
        for job_id in range(218102, 218109):
            self.assertIn(str(job_id), text)
        self.assertIn(
            "for jid in 218103 218104 218105 218106 218107 218108 218102",
            text,
        )
        self.assertIn("verify_terminal_probe 218196", text)
        self.assertIn("verify_terminal_probe 218197", text)
        self.assertIn("verify_cancelled_accounting", text)
        self.assertIn("wait_for_cancelled_accounting", text)
        self.assertIn("attempt<=60", text)
        self.assertIn("sleep 5", text)
        self.assertIn("cancellation_receipt_v1.bundle", text)
        self.assertIn("sacct -X --array", text)
        self.assertIn("--format=JobID,JobName,State,Elapsed,ExitCode", text)
        self.assertNotIn("JobIDRaw", text)
        self.assertIn("expected_task_count", text)
        self.assertIn("cancelled_array_entry_from_rows", text)
        self.assertIn('EXPECTED_DEPENDENCY[218108]', text)
        self.assertIn('canonical_dependency "$dependency"', text)
        self.assertIn("identity mismatch: field=%s", text)
        self.assertLess(
            text.index('tar -C "$OLD_RUN_ROOT"'),
            text.index('scancel "$jid"'),
        )
        self.assertLess(
            text.index('scancel "$jid"'),
            text.rindex("verify_receipt || return 1"),
        )
        self.assertIn("cancelled_accounting_entry", text)
        self.assertIn('for jid in "${ABSENT_OLD[@]}"', text)
        self.assertIn('for jid in "${ACTIVE_OLD[@]}"', text)
        self.assertIn("pre_cancel_closeout_v1.bundle", text)
        self.assertIn("pre_cancel_v1.bundle", text)
        self.assertIn('mv "$temporary_dir" "$RECEIPT_DIR"', text)
        self.assertIn('mv "$archive_tmp" "$archive_bundle"', text)
        self.assertIn(
            'EVSP_LADDER_RESERVATIONS="$NEW_RESERVATIONS"', text
        )
        self.assertIn('EVSP_LADDER_RUN_ROOT="$NEW_RUN_ROOT"', text)
        self.assertIn('EVSP_LADDER_PLAN_ROOT="$NEW_PLAN_ROOT"', text)
        self.assertIn('EVSP_LADDER_PYTHON="$NEW_PYTHON"', text)
        self.assertIn('EVSP_LADDER_RETRY=', text)
        self.assertNotIn("scancel --user", text)
        self.assertNotIn("scancel -u", text)
        self.assertNotIn("set -e", text)
        self.assertNotIn("set -u", text)
        self.assertNotIn("set -o pipefail", text)

        def verify_rows(rows):
            return subprocess.run(
                [
                    "bash", "-c",
                    'source "$1" && cancelled_array_entry_from_rows '
                    '218103 LDPFbcea 0-2 "$2"',
                    "bash", str(script), rows,
                ],
                text=True, capture_output=True, check=False,
            )

        valid = "\n".join((
            "218103_2|LDPFbcea|CANCELLED by 1646707|00:00:00|0:0",
            "218103_0|LDPFbcea|CANCELLED|00:00:00|0:0",
            "218103_1|LDPFbcea|CANCELLED+|00:00:00|0:0",
        ))
        accepted = verify_rows(valid)
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(accepted.stderr, "")
        payload = json.loads(accepted.stdout)
        self.assertEqual(payload["task_count"], 3)
        self.assertEqual(
            [item["task_id"] for item in payload["tasks"]], [0, 1, 2]
        )
        invalid_rows = (
            # Missing one expected task.
            "\n".join(valid.splitlines()[:2]),
            # Duplicate task 1.
            valid + "\n218103_1|LDPFbcea|CANCELLED|00:00:00|0:0",
            # One task actually ran.
            valid.replace(
                "218103_1|LDPFbcea|CANCELLED+|00:00:00|0:0",
                "218103_1|LDPFbcea|COMPLETED|00:00:01|0:0",
            ),
            # Out-of-range task.
            valid + "\n218103_3|LDPFbcea|CANCELLED|00:00:00|0:0",
            # A raw allocation ID cannot stand in for ArrayJobID_TaskID.
            valid.replace("218103_0", "218999"),
        )
        for rows in invalid_rows:
            rejected = verify_rows(rows)
            self.assertNotEqual(rejected.returncode, 0, rows)

        def verify_live_array(rows):
            return subprocess.run(
                [
                    "bash", "-c",
                    'source "$1" && validate_pending_array_records '
                    '218107 LDMRbcea scaglione 0-20 '
                    'SLAD:bcea6b9391cf49515de2:MIP_RAW '
                    '"afterok:218102,aftercorr:218105_*" nc437 "$2"',
                    "bash", str(script), rows,
                ],
                text=True, capture_output=True, check=False,
            )

        live_dependency = (
            "afterok:218102(unfulfilled),"
            "aftercorr:218105_*(unfulfilled)"
        )

        def live_record(raw_job, tasks, **changes):
            fields = {
                "JobId": str(raw_job),
                "ArrayJobId": "218107",
                "ArrayTaskId": str(tasks),
                "JobName": "LDMRbcea",
                "UserId": "nc437(1646707)",
                "JobState": "PENDING",
                "Reason": "Dependency",
                "RunTime": "00:00:00",
                "Partition": "scaglione",
                "Comment": "SLAD:bcea6b9391cf49515de2:MIP_RAW",
                "Dependency": live_dependency,
            }
            fields.update(changes)
            return " ".join(
                f"{key}={value}" for key, value in fields.items()
            )

        live_rows = [live_record(218107, "10-20")] + [
            live_record(218120 - task, task) for task in range(9, -1, -1)
        ]
        accepted_live = verify_live_array("\n".join(live_rows))
        self.assertEqual(
            accepted_live.returncode, 0,
            accepted_live.stdout + accepted_live.stderr,
        )
        invalid_live_arrays = (
            live_rows[:-1],
            [live_record(218107, "9-20"), *live_rows[1:]],
            [
                live_rows[0],
                live_record(218111, 9, ArrayJobId="999"),
                *live_rows[2:],
            ],
            [live_rows[0], live_record(218107, 9), *live_rows[2:]],
            [
                live_rows[0],
                live_record(218111, 9, Dependency="afterok:218102"),
                *live_rows[2:],
            ],
        )
        for rows in invalid_live_arrays:
            rejected_live = verify_live_array("\n".join(rows))
            self.assertNotEqual(rejected_live.returncode, 0, rows)

        def canonical(value):
            return subprocess.run(
                [
                    "bash", "-c",
                    'source "$1" && canonical_dependency "$2"',
                    "bash", str(script), value,
                ],
                text=True, capture_output=True, check=False,
            )

        expected = canonical(
            "afterok:218102,afterok:218103_*"
        )
        self.assertEqual(expected.returncode, 0, expected.stderr)
        equivalent_displays = (
            "afterok:218102(unfulfilled),afterok:218103_*(unfulfilled)",
            "afterok:218103_*(unfulfilled),afterok:218102(unfulfilled)",
            "afterok:218102:218103_*(failed)",
        )
        for displayed in equivalent_displays:
            observed = canonical(displayed)
            self.assertEqual(observed.returncode, 0, displayed)
            self.assertEqual(observed.stdout, expected.stdout, displayed)

        all_group_displays = (
            (
                "afterok:218102",
                "afterok:218102(unfulfilled)",
            ),
            (
                "afterok:218102,afterok:218103_*",
                "afterok:218103_*(unfulfilled),"
                "afterok:218102(unfulfilled)",
            ),
            (
                "afterok:218102,aftercorr:218105_*",
                "aftercorr:218105_*(unfulfilled),"
                "afterok:218102(unfulfilled)",
            ),
            (
                "afterok:218102,aftercorr:218105_*:218104_*",
                "aftercorr:218105_*(unfulfilled),"
                "aftercorr:218104_*(unfulfilled),"
                "afterok:218102(unfulfilled)",
            ),
        )
        for expected_value, displayed in all_group_displays:
            expected_group = canonical(expected_value)
            observed_group = canonical(displayed)
            self.assertEqual(expected_group.returncode, 0, expected_value)
            self.assertEqual(observed_group.returncode, 0, displayed)
            self.assertEqual(
                observed_group.stdout, expected_group.stdout, displayed
            )

        changed_dependencies = (
            "afterok:218102,afterok:218103",
            "afterok:218102_*,afterok:218103_*",
            "afterok:218102,afterok:218103_5",
            "afterok:218102,afterany:218103_*",
            "afterok:218102,afterok:218103_*,afterok:999999",
            "afterok:218102?afterok:218103_*",
            "afterok:218102,afterok:",
            "afterok:218102,afterok:218103_*,",
            "afterok:218102,afterok:218103_*:",
            "afterok:218(unfulfilled)102,afterok:218103_*",
            "after(unfulfilled)ok:218102,afterok:218103_*",
            "afterok:218102(unfulfilled)(failed),afterok:218103_*",
        )
        for displayed in changed_dependencies:
            observed = canonical(displayed)
            if (
                "_5" not in displayed
                and "?" not in displayed
                and not displayed.endswith(("afterok:", ",", ":"))
                and "(unfulfilled)102" not in displayed
                and "after(unfulfilled)ok" not in displayed
                and "(unfulfilled)(failed)" not in displayed
            ):
                self.assertEqual(observed.returncode, 0, displayed)
                self.assertNotEqual(observed.stdout, expected.stdout, displayed)
            else:
                self.assertNotEqual(observed.returncode, 0, displayed)

        wait_script = r'''
source "$1" || exit 90
checks=0
sleeps=0
verify_cancelled_accounting() {
  checks=$((checks + 1))
  [[ "$checks" -ge 61 ]]
}
sleep() {
  [[ "$1" == "5" ]] || return 91
  sleeps=$((sleeps + 1))
}
wait_for_cancelled_accounting || exit 92
printf '%s %s\n' "$checks" "$sleeps"
'''
        waited = subprocess.run(
            ["bash", "-c", wait_script, "bash", str(script)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(waited.returncode, 0, waited.stderr)
        self.assertEqual(waited.stdout.splitlines()[-1], "61 60")

        interrupted_wait_script = r'''
source "$1" || exit 90
checks=0
verify_cancelled_accounting() {
  checks=$((checks + 1))
  return 1
}
sleep() { return 1; }
wait_for_cancelled_accounting
status=$?
printf '%s %s\n' "$status" "$checks"
[[ "$status" -ne 0 && "$checks" -eq 1 ]]
'''
        interrupted = subprocess.run(
            [
                "bash", "-c", interrupted_wait_script,
                "bash", str(script),
            ],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(interrupted.returncode, 0, interrupted.stderr)
        self.assertEqual(interrupted.stdout, "1 1\n")


    def test_worker_maps_dependencies_and_resume(self):
        worker = (REPO_ROOT / "src/submit_scale_ladder.sub").read_text()
        launcher = (REPO_ROOT / "src/launch_scale_ladder.py").read_text()
        self.assertIn("--resume", worker)
        self.assertIn("--snapshot-at-minutes", worker)
        self.assertIn("KNOWN-PARTITION", worker)
        reconciler = (
            REPO_ROOT / "src/reconcile_scale_ladder_gate.py"
        ).read_text()
        self.assertIn("aftercorr:", reconciler)
        self.assertIn("JobName=\"$JOB_NAME\"", worker)
        self.assertIn("EVSP_MIP_EXPECTED_RESULT_SHA256", worker)
        local = (
            REPO_ROOT / "src/run_scale_ladder_local_diagnostics.py"
        ).read_text()
        self.assertNotIn("sbatch", local)
        self.assertNotIn('"phase": "PREFLIGHT"', local)
        self.assertIn("default=3", local)
        self.assertIn("diagnostic_only", local)
        self.assertFalse(set(LOCAL_CODE_PATHS) - set(ladder.CODE_PATHS))

    def test_array_requeue_policy_matches_resumable_cg_phases(self):
        groups = (
            "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
            "MIP_RAW", "MIP_KNOWN",
        )
        plan = {
            "task_groups": {
                group: [f"job_{group.lower()}"] for group in groups
            },
            "jobs": [
                {"job_key": f"job_{group.lower()}", "budget_s": 60}
                for group in groups
            ],
            "python": {"path": "/approved/python", "sha256": "p"},
            "scontrol": {"path": "/approved/scontrol", "sha256": "s"},
            "worker_sha256": "w",
            "runtime_environment": {
                "HOME": "/home/test", "USER": "test",
            },
        }
        for group in groups:
            with self.subTest(group=group), patch.object(
                ladder, "_sbatch", return_value="100"
            ) as submitted:
                ladder._submit_array(
                    plan,
                    Path("/campaign/approved-plan.json"),
                    "a" * 64,
                    group,
                    "99",
                    Path("/campaign/logs"),
                )
                arguments = submitted.call_args.args[1]
                if group in {"CG", "CG_SENSITIVITY"}:
                    self.assertIn("--requeue", arguments)
                    self.assertNotIn("--no-requeue", arguments)
                else:
                    self.assertIn("--no-requeue", arguments)
                    self.assertNotIn("--requeue", arguments)

    def test_k2_r1_certified_15_vs_5_route_space_distinction(self):
        with INSTANCE_MANIFEST.open(newline="") as handle:
            row = next(
                item for item in csv.DictReader(handle)
                if item["scale"] == "2"
                and item["selection_replicate"] == "1"
            )
        observed = {}
        for soc_step in (15.0, 5.0):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "cg.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(REPO_ROOT / "src/exact_pricer_expanded.py"),
                        "--csv", row["relative_path"].removeprefix("data/"),
                        "--prices_csv", "hourly_prices_flat.csv",
                        "--soc-step", str(soc_step),
                        "--block-min", "10",
                        "--max-iters", "2000",
                        "--master-sense", "partition",
                        "--initial-pool", "singletons",
                        "--wall-limit-s", "300",
                        "--checkpoint-every", "25",
                        "--resume", "--out", str(output),
                    ],
                    cwd=REPO_ROOT,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=180,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                payload = json.loads(output.read_text())
                self.assertTrue(payload["certified_rc_optimal"])
                self.assertEqual(payload["final"]["artificials"], 0.0)
                observed[soc_step] = payload["final_lp"]["route_weight"]
        self.assertGreater(observed[15.0], 2.0)
        self.assertAlmostEqual(observed[5.0], 2.0, places=8)

    def test_k2_r2_positive_gap_is_not_labeled_runtime_failure(self):
        with INSTANCE_MANIFEST.open(newline="") as handle:
            row = next(
                item for item in csv.DictReader(handle)
                if item["scale"] == "2"
                and item["selection_replicate"] == "2"
            )
        membership = audit(
            REPO_ROOT / row["relative_path"],
            row["instance_file_sha256"],
            2,
            2,
        )
        self.assertFalse(
            membership["known_partition_in_primary_expanded_space"]
        )
        self.assertEqual(
            target_gap_interpretation(
                2.153846, 2,
                membership["known_partition_in_primary_expanded_space"],
            ),
            "known_partition_outside_primary_space_not_scaling_failure",
        )
        self.assertEqual(
            target_gap_interpretation(2.0, 2, True, 1.0),
            "target_not_comparable_positive_or_missing_artificial_mass",
        )

    def test_summary_schema_and_censoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            instance = root / "instance.csv"
            instance.write_text(
                "Ordered_Trip_ID\n10\n20\n"
            )
            identities = identity(instance)
            cg = root / "cg.json"
            journal = Path(str(cg) + ".columns.jsonl")
            journal.write_text(
                json.dumps({"trips": [0, 1], "cost": 200000.0}) + "\n"
            )
            cg.write_text(json.dumps({
                "columns_journal": str(journal),
                "wall_s": 100.0,
                "iterations": 2,
                "certified_rc_optimal": False,
                "stop_reason": "wall_limit",
                "columns": 2,
                "trip_set_sha256":
                    identities["solver_local_trip_index_sha256"],
                "soc_step": 15.0, "block_min": 10,
                "g_kwh": 300.0, "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "provenance": {
                    "instance_sha256":
                        identities["instance_file_sha256"],
                    "prices_sha256": "c" * 64,
                },
            }))
            Path(str(cg) + ".iters.csv").write_text(
                "elapsed_s,iteration,lp_obj,route_weight,artificials,min_rc,pool_columns\n"
                "10,1,3,2.5,1,-1,1\n"
                "100,2,2,2,0,-0.1,2\n"
            )
            telemetry = root / "telemetry.jsonl"
            telemetry.write_text(
                json.dumps({
                    "record_type": "phase", "phase": "master_attempt",
                    "duration_s": 1.0, "iteration": 1,
                }) + "\n" + json.dumps({
                    "record_type": "phase",
                    "phase": "pricing_shortest_path",
                    "duration_s": 2.0, "iteration": 1,
                }) + "\n" + json.dumps({
                    "record_type": "phase", "phase": "master_attempt",
                    "duration_s": 1.0, "iteration": 2,
                }) + "\n" + json.dumps({
                    "record_type": "phase",
                    "phase": "pricing_shortest_path",
                    "duration_s": 2.0, "iteration": 2,
                }) + "\n"
            )
            mip = root / "mip.json"
            mip.write_text(json.dumps({
                "status_name": "TIME_LIMIT", "incumbent_found": True,
                "buses": 2, "fleet_bound": 1, "mip_gap": 0.5,
                "fleet_proven": False, "runtime_s": 50,
                "optimal_scope": "none",
                "source_result_sha256": sha256_file(cg),
                "source_journal_sha256": sha256_file(journal),
                "physics": {
                    "g_kwh": 300.0, "charge_kw": 300.0,
                    "min_soc_frac": 0.0,
                },
                "experiment_arm": "B",
                "mip_start": {
                    "kind": "none", "source": None,
                },
                "progress": {"checkpoint_schedule_s": [60.0]},
            }))
            progress = root / "progress"
            progress.mkdir()
            (progress / "checkpoint_0001m.json").write_text(json.dumps({
                "checkpoint_elapsed_s": 60,
                "incumbent": {
                    "fleet": 2, "route_vector_sha256": "a" * 64,
                },
                "latest_statistics": {
                    "statistics_incumbent_fleet": 2,
                    "fleet_bound": 1, "fleet_gap": 0.5,
                    "node_count": 3, "solution_count": 1,
                },
                "solver_ended_before_checkpoint": False,
            }))
            (progress / "final.json").write_text(json.dumps({
                "kind": "final", "final": {"status_name": "TIME_LIMIT"},
            }))
            base = {
                "cell_id": "k02_s1_c1", "scale": 2,
                "selection_replicate": 1, "cg_replicate": 1,
                "campaign_replicate": 1, "target_fleet": 2,
                "soc_step": 15.0, "block_min": 10,
                "instance": {
                    "path": str(instance), **identities,
                },
            }
            preflight_output = root / "preflight.json"
            preflight_payload = {
                "schema": "evsp-dr-scale-ladder-known-membership-v1",
                "cell_id": "k02_s1", "scale": 2,
                "selection_replicate": 1,
                "known_partition_continuously_feasible": True,
                "known_partition_in_primary_expanded_space": False,
                "fixed_sequence_pricing_certified": True,
                "first_feasible_soc_step": 5.0,
                "first_feasible_block_min": 10,
                "nonrepresentability_reason": "primary_grid_blocked",
                "duties": [],
            }
            preflight_output.write_text(json.dumps(preflight_payload))
            preflight_output.with_suffix(".csv").write_text(
                ",".join([
                    "cell_id", "scale", "selection_replicate", "duty_id"
                ]) + "\n"
            )
            preflight_job = {
                **base, "job_key": "preflight", "phase": "PREFLIGHT",
                "arm": None, "budget_s": 10,
                "output": str(preflight_output),
                "telemetry": None, "progress_dir": None,
                "snapshot_minutes": [],
            }
            cg_job = {
                **base, "job_key": "cg", "phase": "CG", "arm": None,
                "budget_s": 100, "output": str(cg),
                "telemetry": str(telemetry), "progress_dir": None,
                "snapshot_minutes": [],
            }
            cg5 = root / "cg5.json"
            cg5_journal = Path(str(cg5) + ".columns.jsonl")
            cg5_journal.write_text(journal.read_text())
            cg5_status = json.loads(cg.read_text())
            cg5_status["columns_journal"] = str(cg5_journal)
            cg5_status["soc_step"] = 5.0
            cg5.write_text(json.dumps(cg5_status))
            Path(str(cg5) + ".iters.csv").write_text(
                Path(str(cg) + ".iters.csv").read_text()
            )
            telemetry5 = root / "telemetry5.jsonl"
            telemetry5.write_text(telemetry.read_text())
            cg5_job = {
                **base, "job_key": "cg5",
                "phase": "CG_SENSITIVITY", "arm": None,
                "budget_s": 100, "output": str(cg5),
                "telemetry": str(telemetry5), "progress_dir": None,
                "snapshot_minutes": [], "soc_step": 5.0,
                "block_min": 10,
            }
            mip_job = {
                **base, "job_key": "mip", "phase": "MIP", "arm": "RAW",
                "budget_s": 60, "output": str(mip),
                "progress_dir": str(progress), "telemetry": None,
                "scientific_role": None,
                "dependency_cg": "cg",
                "snapshot_minutes": [],
            }
            plan = {
                "execution_mode": "local_diagnostic",
                "checkout_identity": {"commit": "b" * 40},
                "tariff": {"primary_tariff_sha256": "c" * 64},
                "physics": {}, "trip_identity_schema":
                    "evsp-dr-trip-identity-v1",
                "code_hashes": {}, "python_identity": {},
                "task_groups": {
                    "PREFLIGHT": ["preflight"],
                    "CG": ["cg"], "CG_SENSITIVITY": ["cg5"],
                    "MIP_RAW": ["mip"],
                },
                "jobs": [preflight_job, cg_job, cg5_job, mip_job],
                "k40_reuse_slots": [],
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "execution_mode": "local_diagnostic",
                "diagnostic_only": True,
                "submitted": False,
                "gate_state": "not_applicable_local",
                "submitted_arrays": {},
            }))
            for job, artifacts in (
                (
                    preflight_job,
                    [preflight_output, preflight_output.with_suffix(".csv")],
                ),
                (
                    cg_job,
                    [cg, journal, Path(str(cg) + ".iters.csv"), telemetry],
                ),
                (
                    cg5_job,
                    [
                        cg5, cg5_journal,
                        Path(str(cg5) + ".iters.csv"), telemetry5,
                    ],
                ),
                (
                    mip_job,
                    [
                        mip, progress / "checkpoint_0001m.json",
                        progress / "final.json",
                    ],
                ),
            ):
                completion = {
                    "schema":
                        "evsp-dr-scale-ladder-worker-completion-v1",
                    "phase": job["phase"],
                    "plan_sha256": plan_sha,
                    "instance_file_sha256":
                        identities["instance_file_sha256"],
                    "job_key": job["job_key"],
                    "arm": job["arm"],
                    "artifact_sha256": {
                        str(path.resolve()): sha256_file(path)
                        for path in artifacts
                    },
                }
                Path(
                    str(job["output"]) + ".worker-completion.json"
                ).write_text(json.dumps(completion))
            out = root / "summary"
            summarize(root, out)
            required = {
                "cg_iteration_long.csv", "cg_run_summary.csv",
                "mip_checkpoint_long.csv", "mip_run_summary.csv",
                "artifact_inventory.csv", "scale_progress_summary.csv",
                "known_route_membership_long.csv",
                "cg_route_weight.png", "mip_incumbent_bound.png",
                "provenance.json",
            }
            self.assertTrue(required <= {
                path.name for path in out.iterdir()
            })
            with (out / "cg_run_summary.csv").open(newline="") as handle:
                summary_rows = list(csv.DictReader(handle))
            row = next(
                item for item in summary_rows
                if item["campaign_role"] == "primary"
            )
            sensitivity = next(
                item for item in summary_rows
                if item["campaign_role"] == "small_grid_sensitivity"
            )
            self.assertEqual(row["censored"], "True")
            self.assertEqual(sensitivity["soc_step"], "5.0")
            self.assertEqual(
                sensitivity["grid_interpretation"],
                "route_space_sensitivity_diagnostic",
            )
            self.assertEqual(row["trip_identity_schema"],
                             "evsp-dr-trip-identity-v1")
            self.assertNotIn("trip_set_sha256", CG_FIELDS)
            self.assertNotIn("trip_set_sha256", PROGRESS_FIELDS)

    def test_early_certified_cg_censors_future_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "cg.json"
            journal = Path(str(output) + ".columns.jsonl")
            iterations = Path(str(output) + ".iters.csv")
            output.write_text(json.dumps({
                "wall_s": 100.0, "stop_reason": "certified",
                "snapshot_availability": {
                    "5": "censored_solver_terminated_before_mark"
                },
            }))
            journal.write_text("{}\n")
            iterations.write_text("header\n")
            job = {
                "job_key": "cg", "phase": "CG", "arm": None,
                "output": str(output), "telemetry": None,
                "snapshot_minutes": [5],
                "instance": {"instance_file_sha256": "a" * 64},
            }
            completion = {
                "schema":
                    "evsp-dr-scale-ladder-worker-completion-v1",
                "phase": "CG", "plan_sha256": "p" * 64,
                "instance_file_sha256": "a" * 64,
                "job_key": "cg", "arm": None,
                "snapshot_availability": {
                    "5": "censored_solver_terminated_before_mark"
                },
                "artifact_sha256": {
                    str(path.resolve()): sha256_file(path)
                    for path in (output, journal, iterations)
                },
            }
            Path(str(output) + ".worker-completion.json").write_text(
                json.dumps(completion)
            )
            _validate_completion(job, "p" * 64)

    def test_pending_mip_result_recovers_only_censored_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = Path(tmp)
            (progress / "result_pending.json").write_text(json.dumps({
                "status": 9, "status_name": "TIME_LIMIT",
                "incumbent_found": True, "buses": 3,
                "mip_obj": 300000.0, "mip_bound": 200000.0,
                "mip_gap": 0.5, "fleet_proven": False,
                "optimal_scope": "none", "runtime_s": 100.0,
                "progress": {
                    "checkpoint_schedule_s": [0.0, 60.0, 300.0],
                    "termination_signal": "SIGUSR1",
                },
            }))
            (progress / "latest.json").write_text(json.dumps({
                "schema": "evsp-dr-mip-convergence-v1",
                "kind": "latest",
                "incumbent": {
                    "fleet": 3, "route_vector_sha256": "a" * 64,
                },
                "latest_statistics": {"fleet_bound": 2.0},
            }))
            (progress / "checkpoint_0000m.json").write_text("{}")
            recover(progress)
            self.assertTrue((progress / "final.json").is_file())
            crossed = json.loads(
                (progress / "checkpoint_0001m.json").read_text()
            )
            self.assertFalse(crossed["solver_ended_before_checkpoint"])
            self.assertIsNone(crossed["incumbent"])
            self.assertEqual(
                crossed["recovery"]["observation_availability"],
                "unavailable_interrupted_before_checkpoint_publication",
            )
            recovered = json.loads(
                (progress / "checkpoint_0005m.json").read_text()
            )
            self.assertTrue(recovered["solver_ended_before_checkpoint"])
            self.assertTrue(recovered["recovery"]["observational_only"])


if __name__ == "__main__":
    unittest.main()
