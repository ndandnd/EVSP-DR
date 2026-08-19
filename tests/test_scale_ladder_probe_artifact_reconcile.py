import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import reconcile_scale_ladder_probe_artifacts as reconcile  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ProbeArtifactReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        (self.campaign / "probes").mkdir()
        self.scontrol = self.root / "scontrol"
        self.squeue = self.root / "squeue"
        self.sacct = self.root / "sacct-must-not-run"
        for path in (self.scontrol, self.squeue, self.sacct):
            path.write_text("fixture executable\n")
            path.chmod(0o755)
        self.plan = {
            "campaign_root": str(self.campaign),
            "probe_worker_sha256": "a" * 64,
            "python_identity": {
                "portable_identity_sha256": "portable-identity",
            },
            "scontrol": {
                "available": True,
                "path": str(self.scontrol),
                "sha256": sha(self.scontrol),
            },
            "squeue": {
                "available": True,
                "path": str(self.squeue),
                "sha256": sha(self.squeue),
            },
            "sacct": {
                "available": True,
                "path": str(self.sacct),
                "sha256": sha(self.sacct),
            },
            "jobs": [],
        }
        self.plan_path = self.campaign / "approved-plan.json"
        self.plan_path.write_text(json.dumps(self.plan, sort_keys=True))
        self.plan_sha = sha(self.plan_path)
        self.arrays = {
            "PREFLIGHT": "301",
            "SEED": "302",
            "CG": "303",
            "CG_SENSITIVITY": "304",
            "MIP_RAW": "305",
            "MIP_KNOWN": "306",
        }
        self.manifest = {
            "approval_sha256": self.plan_sha,
            "submitted": False,
            "gate_state": "held_probe_failure",
            "gate_job_id": "300",
            "submitted_arrays": self.arrays,
            "probe_state": "failed_gate_retained",
            "infrastructure_probes": {
                partition: {
                    "job_id": str(100 + index),
                    "attempt": 1,
                    "partition": partition,
                }
                for index, partition in enumerate(reconcile.PARTITIONS)
            },
            "probe_results": {
                partition: {
                    "job_id": str(100 + index),
                    "state": "TIMEOUT",
                    "compatible": False,
                }
                for index, partition in enumerate(reconcile.PARTITIONS)
            },
        }
        self.manifest_path = self.campaign / "campaign.json"
        self._write_manifest()
        self.probe_jobs = {
            "default_partition": "401",
            "scaglione": "402",
        }
        for partition in reconcile.PARTITIONS:
            self._write_probe(partition)
        self.audit = self.campaign / "operator_probe_recovery.txt"
        self._write_audit()
        self.calls = []
        self.reconciler = {
            "git_commit": "c" * 40,
            "relative_path": "src/reconcile_scale_ladder_probe_artifacts.py",
            "sha256": "d" * 64,
            "detached": True,
            "tracked_clean": True,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _write_manifest(self):
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n"
        )

    def _sidecar(self, path, *, absolute=False):
        target = str(path.resolve()) if absolute else path.name
        Path(str(path) + ".sha256").write_text(f"{sha(path)}  {target}\n")

    def _probe_path(self, partition):
        return self.campaign / "probes" / f"{partition}.attempt2.json"

    def _write_probe(self, partition, **changes):
        probe_id = reconcile.PROBE_IDS[partition]
        payload = {
            "schema": reconcile.PROBE_SCHEMA,
            "probe_id": probe_id,
            "probe_attempt": 2,
            "slurm_job_id": self.probe_jobs[partition],
            "slurm_partition": partition,
            "plan_sha256": self.plan_sha,
            "compatible": True,
            "differences": [],
            "planned_portable_identity_sha256": "portable-identity",
            "observed_portable_identity_sha256": "portable-identity",
            "observed_node_metadata": {"hostname": f"{probe_id}-node"},
        }
        payload.update(changes)
        path = self._probe_path(partition)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self._sidecar(path)

    def _write_audit(self, dependency=None):
        dependency = dependency or (
            "afterok:401(unfulfilled),afterok:402(unfulfilled)"
        )
        self.audit.write_text(
            "\n".join([
                f"schema={reconcile.AUDIT_SCHEMA}",
                f"plan_sha256={self.plan_sha}",
                "gate_job_id=300",
                "default_probe_job_id=401",
                "scaglione_probe_job_id=402",
                f"default_output={self._probe_path('default_partition')}",
                f"scaglione_output={self._probe_path('scaglione')}",
                f"worker_sha256={self.plan['probe_worker_sha256']}",
                f"JobId=300 JobName=LDG{self.plan_sha[:5]} "
                "Partition=default_partition JobState=PENDING "
                "Reason=JobHeldUser "
                f"Dependency={dependency} Comment=SLADG:{self.plan_sha[:20]}",
            ]) + "\n"
        )
        self._sidecar(self.audit, absolute=True)

    def _runner(self, command, **kwargs):
        self.calls.append(command)
        self.assertNotEqual(Path(command[0]), self.sacct)
        if Path(command[0]) == self.scontrol:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"JobId=300 JobName=LDG{self.plan_sha[:5]} "
                    "Partition=default_partition JobState=COMPLETED "
                    "ExitCode=0:0 Reason=None "
                    f"Comment=SLADG:{self.plan_sha[:20]}\n"
                ),
                stderr="",
            )
        if Path(command[0]) == self.squeue:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    def _preview(self, runner=None):
        return reconcile.build_preview(
            self.campaign,
            self.plan_sha,
            self.audit,
            runner=runner or self._runner,
            reconciler=self.reconciler,
        )

    def test_preview_uses_artifacts_and_controller_without_sacct(self):
        preview = self._preview()
        self.assertFalse(preview["evidence"]["sacct_used"])
        self.assertEqual(
            preview["evidence"]["controller_proof"]["type"],
            "live_gate_state",
        )
        self.assertTrue(self.calls)
        self.assertTrue(all(Path(call[0]) != self.sacct for call in self.calls))
        proposed = preview["proposed_manifest"]
        self.assertTrue(proposed["submitted"])
        self.assertEqual(proposed["probe_state"], "passed")
        self.assertEqual(proposed["gate_state"], "released_reconciled")
        self.assertEqual(
            proposed["infrastructure_probes"]["default_partition"]["attempt"],
            2,
        )
        self.assertEqual(
            proposed["probe_attempt_history"]["default_partition"][0]
            ["result"]["state"],
            "TIMEOUT",
        )

    def test_preview_and_hash_approved_apply_are_atomic_and_idempotent(self):
        preview = self._preview()
        report = self.campaign / "reconciliation-preview.json"
        report_sha = reconcile.publish_preview(preview, report)
        before = sha(self.manifest_path)
        def controller_must_not_be_requeried(command, **kwargs):
            raise AssertionError(f"dynamic controller was requeried: {command}")
        result = reconcile.apply_report(
            report, report_sha, self.audit,
            runner=controller_must_not_be_requeried,
            reconciler=self.reconciler,
        )
        self.assertEqual(result["status"], "applied")
        self.assertEqual(sha(self.manifest_path), preview["manifest_after_sha256"])
        backup = self.campaign / (
            f"campaign.json.before-probe-reconciliation.{before}.json"
        )
        self.assertTrue(backup.is_file())
        self.assertEqual(sha(backup), before)
        second = reconcile.apply_report(
            report, report_sha, self.audit, runner=self._runner,
            reconciler=self.reconciler,
        )
        self.assertEqual(second["status"], "already_applied")

    def test_apply_rejects_report_hash_mismatch_without_mutation(self):
        preview = self._preview()
        report = self.campaign / "preview.json"
        reconcile.publish_preview(preview, report)
        before = self.manifest_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "report hash mismatch"):
            reconcile.apply_report(
                report, "0" * 64, self.audit, runner=self._runner
            )
        self.assertEqual(self.manifest_path.read_bytes(), before)

    def test_apply_rejects_manifest_change_after_preview(self):
        preview = self._preview()
        report = self.campaign / "preview.json"
        report_sha = reconcile.publish_preview(preview, report)
        self.manifest["unexpected_change"] = True
        self._write_manifest()
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            reconcile.apply_report(
                report, report_sha, self.audit, runner=self._runner
            )

    def test_afterany_controller_audit_is_rejected(self):
        self._write_audit("afterany:401:402")
        with self.assertRaisesRegex(ValueError, "only afterok"):
            self._preview()

    def test_missing_or_extra_afterok_job_is_rejected(self):
        for dependency in ("afterok:401", "afterok:401:402:403"):
            with self.subTest(dependency=dependency):
                self._write_audit(dependency)
                with self.assertRaisesRegex(ValueError, "exact held afterok gate"):
                    self._preview()

    def test_probe_tamper_and_compatibility_failure_are_rejected(self):
        path = self._probe_path("default_partition")
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self._preview()
        self._write_probe("default_partition", compatible=False)
        with self.assertRaisesRegex(ValueError, "compatibility mismatch"):
            self._preview()

    def test_probe_identity_swap_is_rejected(self):
        self._write_probe("default_partition", slurm_job_id="402")
        with self.assertRaisesRegex(ValueError, "compatibility mismatch"):
            self._preview()

    def test_probe_symlink_is_rejected(self):
        path = self._probe_path("default_partition")
        original = path.with_name("original.json")
        path.replace(original)
        path.symlink_to(original)
        with self.assertRaisesRegex(ValueError, "non-symlink"):
            self._preview()

    def test_gate_still_blocked_without_downstream_evidence_is_rejected(self):
        def blocked_runner(command, **kwargs):
            self.assertNotEqual(Path(command[0]), self.sacct)
            if Path(command[0]) == self.scontrol:
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        f"JobId=300 JobName=LDG{self.plan_sha[:5]} "
                        "Partition=default_partition JobState=PENDING "
                        "ExitCode=0:0 Reason=Dependency "
                        f"Comment=SLADG:{self.plan_sha[:20]}\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with self.assertRaisesRegex(ValueError, "no controller-independent proof"):
            self._preview(runner=blocked_runner)

    def test_running_gate_is_not_monotonic_afterok_success_proof(self):
        def running_runner(command, **kwargs):
            self.assertNotEqual(Path(command[0]), self.sacct)
            self.assertGreater(kwargs.get("timeout", 0), 0)
            self.assertLessEqual(
                kwargs["timeout"], reconcile.CONTROLLER_QUERY_TIMEOUT_S
            )
            if Path(command[0]) == self.scontrol:
                return SimpleNamespace(
                    returncode=0,
                    stdout=(
                        f"JobId=300 JobName=LDG{self.plan_sha[:5]} "
                        "Partition=default_partition JobState=RUNNING "
                        "ExitCode=0:0 Reason=None "
                        f"Comment=SLADG:{self.plan_sha[:20]}\n"
                    ),
                    stderr="",
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with self.assertRaisesRegex(
            ValueError, "no controller-independent proof"
        ):
            self._preview(runner=running_runner)

    def test_controller_queries_are_bounded_and_timeout_fails_closed(self):
        observed_timeouts = []

        def timeout_runner(command, **kwargs):
            observed_timeouts.append(kwargs.get("timeout"))
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

        with self.assertRaisesRegex(RuntimeError, "controller query timed out"):
            self._preview(runner=timeout_runner)
        self.assertEqual(len(observed_timeouts), 1)
        self.assertGreater(observed_timeouts[0], 0)
        self.assertLessEqual(
            observed_timeouts[0], reconcile.CONTROLLER_QUERY_TIMEOUT_S
        )

        observed_timeouts.clear()

        def squeue_timeout_runner(command, **kwargs):
            observed_timeouts.append(kwargs.get("timeout"))
            if Path(command[0]) == self.scontrol:
                return SimpleNamespace(
                    returncode=1, stdout="", stderr="invalid job id"
                )
            raise subprocess.TimeoutExpired(command, kwargs.get("timeout"))

        with self.assertRaisesRegex(RuntimeError, "controller query timed out"):
            self._preview(runner=squeue_timeout_runner)
        self.assertEqual(len(observed_timeouts), 2)
        self.assertTrue(all(
            timeout is not None
            and 0 < timeout <= reconcile.CONTROLLER_QUERY_TIMEOUT_S
            for timeout in observed_timeouts
        ))

    def test_static_running_gate_proof_is_rejected_during_apply_rebuild(self):
        proof = {
            "type": "live_gate_state",
            "gate_job_id": "300",
            "job_name": f"LDG{self.plan_sha[:5]}",
            "partition": "default_partition",
            "job_state": "RUNNING",
            "exit_code": "0:0",
            "comment": f"SLADG:{self.plan_sha[:20]}",
            "reason": "None",
        }
        with self.assertRaisesRegex(
            ValueError, "live gate controller proof is invalid"
        ):
            reconcile.build_preview(
                self.campaign,
                self.plan_sha,
                self.audit,
                controller_proof=proof,
                reconciler=self.reconciler,
            )

    def test_controller_audit_binds_gate_name_and_partition(self):
        original = self.audit.read_text()
        replacements = (
            (f"JobName=LDG{self.plan_sha[:5]}", "JobName=wrong"),
            ("Partition=default_partition", "Partition=scaglione"),
        )
        for expected, replacement in replacements:
            with self.subTest(replacement=replacement):
                self.audit.write_text(original.replace(expected, replacement))
                self._sidecar(self.audit, absolute=True)
                with self.assertRaisesRegex(
                    ValueError, "exact held afterok gate"
                ):
                    self._preview()
                self.audit.write_text(original)
                self._sidecar(self.audit, absolute=True)

    def test_live_downstream_array_is_accepted(self):
        def array_runner(command, **kwargs):
            self.assertNotEqual(Path(command[0]), self.sacct)
            if Path(command[0]) == self.scontrol:
                return SimpleNamespace(returncode=1, stdout="", stderr="gone")
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"301|0|SLAD:{self.plan_sha[:20]}:PREFLIGHT|"
                    "PENDING|Priority\n"
                ),
                stderr="",
            )
        preview = self._preview(runner=array_runner)
        self.assertEqual(
            preview["evidence"]["controller_proof"]["type"],
            "live_downstream_array",
        )

    def test_validated_worker_completion_is_fallback_controller_proof(self):
        output = self.campaign / "preflight.json"
        output.write_text("{}\n")
        completion = Path(str(output) + ".worker-completion.json")
        instance_sha = "b" * 64
        self.plan["jobs"] = [{
            "job_key": "preflight_k2",
            "phase": "PREFLIGHT",
            "arm": None,
            "output": str(output),
            "instance": {"instance_file_sha256": instance_sha},
        }]
        self.plan_path.write_text(json.dumps(self.plan, sort_keys=True))
        old_plan_sha = self.plan_sha
        self.plan_sha = sha(self.plan_path)
        self.manifest["approval_sha256"] = self.plan_sha
        self._write_manifest()
        for partition in reconcile.PARTITIONS:
            self._write_probe(partition)
        self._write_audit()
        completion.write_text(json.dumps({
            "schema": reconcile.COMPLETION_SCHEMA,
            "phase": "PREFLIGHT",
            "plan_sha256": self.plan_sha,
            "instance_file_sha256": instance_sha,
            "job_key": "preflight_k2",
            "arm": None,
            "artifact_sha256": {str(output.resolve()): sha(output)},
        }, sort_keys=True) + "\n")

        controller_calls = []

        def gone_runner(command, **kwargs):
            controller_calls.append(command)
            self.assertNotEqual(Path(command[0]), self.sacct)
            return SimpleNamespace(returncode=1, stdout="", stderr="gone")

        self.assertNotEqual(old_plan_sha, self.plan_sha)
        preview = self._preview(runner=gone_runner)
        self.assertEqual(
            preview["evidence"]["controller_proof"]["type"],
            "validated_worker_completion",
        )
        self.assertEqual(controller_calls, [])

    def test_audit_sidecar_and_duplicate_keys_are_rejected(self):
        Path(str(self.audit) + ".sha256").write_text(
            f"{'0' * 64}  {self.audit.resolve()}\n"
        )
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self._preview()
        self._write_audit()
        self.audit.write_text(self.audit.read_text() + f"gate_job_id=300\n")
        self._sidecar(self.audit, absolute=True)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            self._preview()


if __name__ == "__main__":
    unittest.main()
