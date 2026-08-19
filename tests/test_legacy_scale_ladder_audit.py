import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_legacy_scale_ladder_campaign import (  # noqa: E402
    CAPTURE_SCHEMA,
    audit_legacy_campaign,
    validate_legacy_sidecar,
)
from build_tariff_response_manifest import sha256_file  # noqa: E402


class LegacyScaleLadderAuditTests(unittest.TestCase):
    GROUPS = (
        "PREFLIGHT", "SEED", "CG", "CG_SENSITIVITY",
        "MIP_RAW", "MIP_KNOWN",
    )

    def _fixture(self, root):
        root = Path(root)
        campaign = root / "campaign"
        campaign.mkdir()
        commit = (
            __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip()
        )
        instance = root / "instance.csv"
        instance.write_text("Ordered_Trip_ID\n1\n")
        instance_sha = sha256_file(instance)
        input_manifest = root / "input.json"
        input_manifest.write_text("{}\n")
        membership = root / "membership.json"
        membership.write_text("{}\n")
        instance_manifest = root / "instances.csv"
        instance_manifest.write_text("scale\n2\n")
        jobs = []
        groups = {}
        for index, group in enumerate(self.GROUPS):
            phase = (
                "MIP" if group.startswith("MIP")
                else group
            )
            job_key = f"job-{group.lower()}"
            output = campaign / f"{job_key}.json"
            artifacts = [output]
            job = {
                "job_key": job_key,
                "phase": phase,
                "arm": (
                    "RAW" if group == "MIP_RAW"
                    else "KNOWN-PARTITION" if group == "MIP_KNOWN"
                    else None
                ),
                "output": str(output),
                "instance": {
                    "path": str(instance),
                    "instance_file_sha256": instance_sha,
                },
                "snapshot_minutes": [],
                "telemetry": None,
                "membership_output": None,
                "progress_dir": None,
            }
            if phase == "PREFLIGHT":
                output.write_text(json.dumps({
                    "schema":
                        "evsp-dr-scale-ladder-known-membership-v1",
                    "duties": [],
                }))
                csv_path = output.with_suffix(".csv")
                csv_path.write_text("cell_id\n")
                artifacts.append(csv_path)
            elif phase in {"CG", "CG_SENSITIVITY"}:
                output.write_text(json.dumps({
                    "trip_ids": [0],
                    "columns_journal": str(
                        Path(str(output) + ".columns.jsonl")
                    ),
                    "stop_reason": "certified",
                }))
                journal = Path(str(output) + ".columns.jsonl")
                journal.write_text('{"trips":[0],"cost":100000}\n')
                iterations = Path(str(output) + ".iters.csv")
                iterations.write_text("iteration\n1\n")
                artifacts.extend([journal, iterations])
            elif phase == "MIP":
                output.write_text(json.dumps({
                    "incumbent_found": False,
                    "progress": {"checkpoint_schedule_s": []},
                }))
                progress = campaign / f"progress-{index}"
                progress.mkdir()
                final = progress / "final.json"
                final.write_text('{"kind":"final"}\n')
                artifacts.append(final)
                job["progress_dir"] = str(progress)
                job["dependency_cg"] = (
                    "job-cg" if group == "MIP_RAW"
                    else "job-cg_sensitivity"
                )
            else:
                output.write_text('{"routes":[]}\n')
            completion = {
                "schema": "evsp-dr-scale-ladder-worker-completion-v1",
                "phase": phase,
                "plan_sha256": None,
                "instance_file_sha256": instance_sha,
                "job_key": job_key,
                "arm": job["arm"],
                "artifact_sha256": {},
            }
            job["_completion"] = completion
            job["_artifacts"] = artifacts
            jobs.append(job)
            groups[group] = [job_key]
        plan = {
            "campaign": "legacy-fixture",
            "campaign_root": str(campaign),
            "checkout_identity": {"commit": commit},
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": sha256_file(input_manifest),
            "membership_preflight": str(membership),
            "membership_preflight_sha256": sha256_file(membership),
            "instance_manifest": str(instance_manifest),
            "instance_manifest_sha256": sha256_file(instance_manifest),
            "tariff": {
                "primary_tariff_relative_path":
                    "data/hourly_prices_flat.csv",
                "primary_tariff_sha256": sha256_file(
                    REPO_ROOT / "data/hourly_prices_flat.csv"
                ),
                "extended_comparator_relative_path":
                    "data/tariff_response/flat_h26.csv",
                "extended_comparator_sha256": sha256_file(
                    REPO_ROOT / "data/tariff_response/flat_h26.csv"
                ),
            },
            "physics": {
                "g_kwh": 300.0, "charge_kw": 300.0,
                "reserve_kwh": 0.0, "soc_step_kwh": 15.0,
                "block_min": 10,
            },
            "runtime_environment": {"USER": "nathan"},
            "code_hashes": {},
            "jobs": [
                {
                    key: value for key, value in job.items()
                    if not key.startswith("_")
                }
                for job in jobs
            ],
            "task_groups": groups,
            "task_count": len(jobs),
        }
        plan_raw = json.dumps(
            plan, sort_keys=True, separators=(",", ":")
        ).encode()
        plan_sha = hashlib.sha256(plan_raw).hexdigest()
        (campaign / "approved-plan.json").write_bytes(plan_raw)
        for job in jobs:
            completion = job["_completion"]
            completion["plan_sha256"] = plan_sha
            completion["artifact_sha256"] = {
                str(path.resolve()): sha256_file(path)
                for path in job["_artifacts"]
            }
            Path(
                str(job["output"]) + ".worker-completion.json"
            ).write_text(json.dumps(completion))
        arrays = {
            group: str(200 + index)
            for index, group in enumerate(self.GROUPS)
        }
        manifest = {
            "approval_sha256": plan_sha,
            "submitted": True,
            "gate_state": "released",
            "gate_job_id": "100",
            "submitted_arrays": arrays,
        }
        (campaign / "campaign.json").write_text(json.dumps(manifest))
        return campaign, plan, manifest, plan_sha, commit

    def _capture(self, path, plan, manifest, plan_sha, commit):
        prefixes = {
            "PREFLIGHT": "LDPF", "SEED": "LDSD",
            "CG": "LDCG", "CG_SENSITIVITY": "LDCS",
            "MIP_RAW": "LDMR", "MIP_KNOWN": "LDMK",
        }
        arrays = {}
        for group in self.GROUPS:
            dependencies = {"afterok": ["100"]}
            if group in {"CG", "CG_SENSITIVITY"}:
                dependencies["afterok"].append(
                    f"{manifest['submitted_arrays']['PREFLIGHT']}_*"
                )
            elif group == "MIP_RAW":
                dependencies["aftercorr"] = [
                    f"{manifest['submitted_arrays']['CG']}_*"
                ]
            elif group == "MIP_KNOWN":
                dependencies["aftercorr"] = sorted([
                    f"{manifest['submitted_arrays']['CG']}_*",
                    f"{manifest['submitted_arrays']['SEED']}_*",
                ])
            arrays[group] = {
                "job_id": manifest["submitted_arrays"][group],
                "job_name": prefixes[group] + plan_sha[:4],
                "partition": (
                    "scaglione" if group.startswith("MIP")
                    else "default_partition"
                ),
                "comment": f"SLAD:{plan_sha[:20]}:{group}",
                "state": "COMPLETED",
                "exit_code": "0:0",
                "task_count": 1,
                "dependency_semantics": {
                    key: sorted(value)
                    for key, value in dependencies.items()
                },
            }
        path.write_text(json.dumps({
            "schema": CAPTURE_SCHEMA,
            "plan_sha256": plan_sha,
            "source_commit": commit,
            "user": "nathan",
            "gate": {
                "job_id": "100",
                "job_name": f"LDG{plan_sha[:5]}",
                "partition": "default_partition",
                "comment": f"SLADG:{plan_sha[:20]}",
                "state": "COMPLETED",
                "exit_code": "0:0",
            },
            "arrays": arrays,
        }))

    def test_old_schema_requires_then_validates_no_clobber_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _plan, _manifest, _sha, commit = self._fixture(tmp)
            sidecar = Path(tmp) / "legacy-audit.json"
            with self.assertRaisesRegex(ValueError, "missing"):
                validate_legacy_sidecar(campaign, sidecar)
            payload = audit_legacy_campaign(
                campaign, sidecar, expected_commit=commit
            )
            self.assertEqual(
                payload["legacy_evidence_status"],
                "legacy_scheduler_unverified",
            )
            validated = validate_legacy_sidecar(campaign, sidecar)
            self.assertEqual(
                validated["normalization_scope"],
                "artifact_provenance_only_scheduler_unverified",
            )
            with self.assertRaises(FileExistsError):
                audit_legacy_campaign(
                    campaign, sidecar, expected_commit=commit
                )

    def test_valid_scheduler_capture_upgrades_only_posthoc_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, plan, manifest, plan_sha, commit = self._fixture(tmp)
            capture = Path(tmp) / "scheduler.json"
            self._capture(
                capture, plan, manifest, plan_sha, commit
            )
            sidecar = Path(tmp) / "legacy-audit.json"
            payload = audit_legacy_campaign(
                campaign,
                sidecar,
                expected_commit=commit,
                scheduler_capture=capture,
            )
            self.assertEqual(
                payload["legacy_evidence_status"],
                "legacy_posthoc_audited",
            )
            self.assertNotIn(
                "gate_release_verification",
                json.loads((campaign / "campaign.json").read_text()),
            )

    def test_tampered_sidecar_and_mixed_evidence_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, _plan, manifest, _plan_sha, commit = self._fixture(tmp)
            sidecar = Path(tmp) / "legacy-audit.json"
            audit_legacy_campaign(
                campaign, sidecar, expected_commit=commit
            )
            sidecar.write_text(sidecar.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_legacy_sidecar(campaign, sidecar)

            second = Path(tmp) / "mixed"
            second.mkdir()
            mixed_campaign, _p, mixed, _s, mixed_commit = self._fixture(
                second
            )
            mixed["gate_release_verification"] = {"verified": True}
            (mixed_campaign / "campaign.json").write_text(
                json.dumps(mixed)
            )
            with self.assertRaisesRegex(ValueError, "mixed old/new"):
                audit_legacy_campaign(
                    mixed_campaign,
                    second / "sidecar.json",
                    expected_commit=mixed_commit,
                )

    def test_missing_completion_and_duplicate_tasks_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign, plan, manifest, _sha, commit = self._fixture(tmp)
            completion = next(campaign.glob("*.worker-completion.json"))
            completion.unlink()
            with self.assertRaises(FileNotFoundError):
                audit_legacy_campaign(
                    campaign,
                    Path(tmp) / "missing.json",
                    expected_commit=commit,
                )

            second = Path(tmp) / "duplicate"
            second.mkdir()
            (
                duplicate_campaign,
                duplicate_plan,
                duplicate_manifest,
                _duplicate_sha,
                duplicate_commit,
            ) = self._fixture(second)
            duplicate_plan["task_groups"]["SEED"] = list(
                duplicate_plan["task_groups"]["PREFLIGHT"]
            )
            raw = json.dumps(
                duplicate_plan, sort_keys=True, separators=(",", ":")
            ).encode()
            (duplicate_campaign / "approved-plan.json").write_bytes(raw)
            duplicate_manifest["approval_sha256"] = hashlib.sha256(
                raw
            ).hexdigest()
            (duplicate_campaign / "campaign.json").write_text(
                json.dumps(duplicate_manifest)
            )
            with self.assertRaisesRegex(ValueError, "duplicated"):
                audit_legacy_campaign(
                    duplicate_campaign,
                    second / "duplicate.json",
                    expected_commit=duplicate_commit,
                )


if __name__ == "__main__":
    unittest.main()
