import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from campaign_artifact_status import output_complete, source_identity  # noqa: E402


class CampaignArtifactStatusTests(unittest.TestCase):
    def make_source(self, folder: Path, *, snapshot=True):
        suffix = ".snapshot.json" if snapshot else ".json"
        source = folder / f"pool{suffix}"
        journal = Path(str(source) + ".columns.jsonl")
        journal.write_text(json.dumps({"trips": [1], "cost": 100000.0}) + "\n")
        source.write_text(json.dumps({
            "stop_reason": "snapshot_m60" if snapshot else "running",
            "columns_journal": str(journal),
            "trip_ids": [1],
        }))
        return source

    def test_nonterminal_source_allowed_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(Path(tmp), snapshot=False)
            identity = source_identity(source, require_terminal=False)
            self.assertEqual(identity["stop_reason"], "running")
            with self.assertRaisesRegex(ValueError, "not immutable/terminal"):
                source_identity(source, require_terminal=True)

    def test_canonical_snapshot_partial_status_is_not_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = self.make_source(folder, snapshot=False)
            payload = json.loads(source.read_text())
            payload["stop_reason"] = "snapshot_m360"
            source.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "not immutable/terminal"):
                source_identity(source, require_terminal=True)

    def test_mip_output_must_match_source_and_runtime_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = self.make_source(folder)
            identity = source_identity(source, require_terminal=True)
            output = folder / "mip.json"
            output.write_text(json.dumps({
                "status_name": "TIME_LIMIT",
                "buses": 40,
                "source_result_sha256": identity["result_sha256"],
                "source_journal_sha256": identity["journal_sha256"],
                "mip_provenance": {"git_commit": "expected"},
            }))
            self.assertTrue(output_complete(
                "MC", source, output, expected_commit="expected"
            ))
            self.assertFalse(output_complete(
                "MC", source, output, expected_commit="different"
            ))

    def test_prepared_control_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = self.make_source(folder)
            identity = source_identity(source, require_terminal=True)
            output = folder / "control.json"
            output.write_text(json.dumps({
                "stop_reason": "prepared_snapshot_resume",
                "wall_s": 21600,
                "resume_parent": {
                    "snapshot_sha256": identity["result_sha256"],
                    "journal_sha256": identity["journal_sha256"],
                },
                "provenance": {"git_commit": "expected"},
            }))
            self.assertFalse(output_complete(
                "CC", source, output, expected_commit="expected"
            ))


if __name__ == "__main__":
    unittest.main()
