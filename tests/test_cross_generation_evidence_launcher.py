import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from archive_cross_generation_evidence import archive_evidence  # noqa: E402
from launch_cross_generation_evidence import (  # noqa: E402
    _dependency_job_ids,
    build_sbatch_command,
)
from run_cross_generation_evidence_job import (  # noqa: E402
    _campaign_ready,
    _require_campaign_artifacts,
    parse_campaign_assignments,
    wait_for_campaigns,
)


class CrossGenerationEvidenceLauncherTests(unittest.TestCase):
    def test_repeatable_campaign_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            pilot = root / "pilot"
            raw.mkdir()
            pilot.mkdir()
            one = parse_campaign_assignments([
                f"raw_k40={raw}",
            ])
            self.assertEqual(one, [(raw.resolve(), "raw_k40")])
            multiple = parse_campaign_assignments([
                f"raw_k40={raw}",
                f"pilot={pilot}",
            ])
            self.assertEqual(
                multiple,
                [(raw.resolve(), "raw_k40"), (pilot.resolve(), "pilot")],
            )

    def test_invalid_campaign_and_dependency_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign"
            campaign.mkdir()
            with self.assertRaisesRegex(ValueError, "at least one"):
                parse_campaign_assignments([])
            with self.assertRaisesRegex(ValueError, "unsupported"):
                parse_campaign_assignments([f"other={campaign}"])
            with self.assertRaisesRegex(ValueError, "must be absolute"):
                parse_campaign_assignments(["pilot=relative/path"])
            with self.assertRaisesRegex(ValueError, "not a directory"):
                parse_campaign_assignments([
                    f"pilot={root / 'missing'}"
                ])
            with self.assertRaisesRegex(ValueError, "decimal integers"):
                _dependency_job_ids(Namespace(after_job_id=["123x"]))
            with self.assertRaisesRegex(ValueError, "must be unique"):
                _dependency_job_ids(
                    Namespace(after_job_id=["123", "123"])
                )

    def test_incomplete_live_campaign_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "running"
            campaign.mkdir()
            (campaign / "campaign.json").write_text(json.dumps({
                "approval_sha256": "0" * 64,
                "jobs": [{"cell_id": "cell"}],
            }))
            with self.assertRaisesRegex(
                RuntimeError, "MIP campaign incomplete"
            ):
                wait_for_campaigns(
                    [(campaign, "pilot")], timeout_s=0, poll_s=1
                )

    def test_raw_campaign_requires_four_cell_raw_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "relabeled"
            campaign.mkdir()
            plan = {
                "mode": "raw_k40",
                "checkout_identity": {"expected_commit": "a" * 40},
                "jobs": [],
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            (campaign / "approved-plan.json").write_bytes(raw)
            (campaign / "campaign.json").write_text(json.dumps({
                "approval_sha256": hashlib.sha256(raw).hexdigest(),
                "jobs": [],
            }))
            ready, reason = _campaign_ready(
                campaign, expected_mode="raw_k40"
            )
            self.assertFalse(ready)
            self.assertIn("raw_k40_validation_failed", reason)

    def test_collection_paths_precede_reviewed_treatment_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = Path(tmp) / "campaign"
            progress = campaign / "progress/cell"
            input_root = campaign / "input/cell"
            output = campaign / "outputs/cell.json"
            campaign.mkdir()
            progress.mkdir(parents=True)
            input_root.mkdir(parents=True)
            (progress / "checkpoint_0000m.json").write_text(json.dumps({
                "schema": "evsp-dr-mip-convergence-v1",
                "kind": "checkpoint",
                "checkpoint_elapsed_s": 0.0,
            }))
            (campaign / "campaign.json").write_text(json.dumps({
                "jobs": [{
                    "cell_id": "cell",
                    "arm": "RAW",
                    "time_limit_s": 0,
                    "output": str(output),
                    "progress_dir": str(progress),
                }],
            }))
            artifacts = [
                {
                    "artifact_type": "mip_final",
                    "path": str(output),
                    "metadata": {},
                },
                {
                    "artifact_type": "mip_checkpoint",
                    "path": str(progress / "checkpoint_0000m.json"),
                    "metadata": {},
                },
                {
                    "artifact_type": "mip_pool_status_json",
                    "path": str(input_root / "pool.snapshot.json"),
                    "metadata": {},
                },
                {
                    "artifact_type": "exact_cg_column_journal_jsonl",
                    "path": str(
                        input_root / "pool.snapshot.json.columns.jsonl"
                    ),
                    "metadata": {},
                },
            ]
            payload = {"artifacts": artifacts}
            _require_campaign_artifacts(
                payload, campaign, require_reviewed_metadata=False
            )
            with self.assertRaisesRegex(
                ValueError, "treatment differs"
            ):
                _require_campaign_artifacts(
                    payload, campaign, require_reviewed_metadata=True
                )
            for artifact in artifacts[:2]:
                artifact["metadata"] = {
                    "treatment": "RAW",
                    "augmentation_kind": "none",
                }
            _require_campaign_artifacts(
                payload, campaign, require_reviewed_metadata=True
            )
            (progress / "checkpoint_0000m.json").write_text(json.dumps({
                "schema": "evsp-dr-mip-convergence-v1",
                "kind": "checkpoint",
                "checkpoint_elapsed_s": 300.0,
            }))
            with self.assertRaisesRegex(
                ValueError, "filename/payload mismatch"
            ):
                _require_campaign_artifacts(
                    payload, campaign, require_reviewed_metadata=True
                )

    def test_archive_is_deterministic_checksummed_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build"
            build.mkdir()
            payload = b"a,b\n1,2\n"
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": "input"}))
            manifest_payload = manifest.read_bytes()
            (build / "artifact_inventory.csv").write_bytes(payload)
            (build / "input_manifest.json").write_bytes(manifest_payload)
            (build / "completion.json").write_text(json.dumps({
                "schema":
                    "evsp-dr-cross-generation-output-completion-v1",
                "members": {
                    "artifact_inventory.csv":
                        hashlib.sha256(payload).hexdigest(),
                    "input_manifest.json":
                        hashlib.sha256(manifest_payload).hexdigest(),
                },
            }))
            first = root / "archive-one"
            second = root / "archive-two"
            result_one = archive_evidence(build, manifest, first)
            result_two = archive_evidence(build, manifest, second)
            self.assertEqual(
                (first / "evidence.tar").read_bytes(),
                (second / "evidence.tar").read_bytes(),
            )
            self.assertEqual(
                result_one["archive_sha256"],
                result_two["archive_sha256"],
            )
            self.assertTrue((first / "completion.json").is_file())
            with self.assertRaises(FileExistsError):
                archive_evidence(build, manifest, first)
            manifest.write_text(json.dumps({"schema": "changed"}))
            with self.assertRaisesRegex(
                ValueError, "differs from manifest used"
            ):
                archive_evidence(build, manifest, root / "archive-three")

    def test_launcher_requires_all_explicit_roots_and_submits_no_solves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            common = {
                alias: root / alias
                for alias in (
                    "current_heuristic", "repool_small", "exact_big",
                    "k40_factorial", "mip_campaign", "releases",
                )
            }
            for path in common.values():
                path.mkdir()
            log_dir = root / "logs"
            log_dir.mkdir()
            current_campaign = root / "current-campaign"
            raw_campaign = root / "raw-campaign"
            current_campaign.mkdir()
            raw_campaign.mkdir()
            args = Namespace(
                repo_root=REPO_ROOT,
                root=[
                    f"{alias}={path}"
                    for alias, path in common.items()
                ],
                phase="collect",
                approved_manifest_sha256=None,
                build_out=None,
                archive_out=None,
                template=REPO_ROOT / (
                    "CROSS_GENERATION_EVIDENCE_INPUT_MANIFEST_20260816.json"
                ),
                manifest=root / "manifest.json",
                campaign=[],
                after_job_id=[],
                current_mip_campaign_root=current_campaign,
                raw_k40_campaign_root=raw_campaign,
                current_mip_mode="pilot",
                wait_timeout_s=0,
                poll_s=1,
                log_dir=log_dir,
                partition="scaglione",
                cpus=4,
                memory="32G",
                time_limit="24:00:00",
                expected_commit="a" * 40,
            )
            command, plan = build_sbatch_command(args)
            joined = " ".join(command)
            self.assertIn("run_cross_generation_evidence_job.py", joined)
            self.assertNotIn("run_exact_pool_mip.py", joined)
            self.assertNotIn("exact_pricer_expanded.py", joined)
            self.assertFalse(plan["submits_cg_or_mip_solves"])
            authoritative = Namespace(**vars(args))
            authoritative.current_mip_campaign_root = None
            authoritative.raw_k40_campaign_root = None
            authoritative.current_mip_mode = None
            authoritative.campaign = [f"raw_k40={raw_campaign}"]
            authoritative.after_job_id = ["101", "202"]
            authoritative.wait_timeout_s = None
            dependency_command, dependency_plan = build_sbatch_command(
                authoritative
            )
            self.assertIn(
                "--dependency=afterany:101:202", dependency_command
            )
            self.assertEqual(
                dependency_plan["worker_wait_timeout_s"], 0.0
            )
            self.assertEqual(
                dependency_plan["campaigns"],
                [{"mode": "raw_k40", "path": str(raw_campaign.resolve())}],
            )
            args.root = args.root[:-1]
            with self.assertRaisesRegex(ValueError, "roots missing"):
                build_sbatch_command(args)


if __name__ == "__main__":
    unittest.main()
