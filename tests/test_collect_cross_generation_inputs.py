import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from collect_cross_generation_inputs import collect, main  # noqa: E402


class CrossGenerationCollectorTests(unittest.TestCase):
    def test_collector_hashes_explicit_matches_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            artifact = source / "run1" / "trace.iters.csv"
            artifact.parent.mkdir()
            artifact.write_text("elapsed_s,iteration\n1,1\n")
            before = artifact.read_bytes()
            template = root / "template.json"
            template.write_text(json.dumps({
                "schema": "evsp-dr-cross-generation-input-manifest-v1",
                "artifacts": [],
                "collection_requests": [{
                    "request_id": "exact",
                    "root_alias": "exact",
                    "glob": "**/*.iters.csv",
                    "artifact_type": "exact_cg_iterations_csv",
                    "run_id_namespace": "exact",
                    "run_id_regex": (
                        "(?P<run_id>[^/]+)/trace\\.iters\\.csv$"
                    ),
                    "metadata": {
                        "algorithm_family": "exact_expanded_network"
                    },
                }],
            }))
            result = collect(template, {"exact": source})
            self.assertEqual(len(result["artifacts"]), 1)
            entry = result["artifacts"][0]
            self.assertEqual(entry["run_id"], "exact:run1")
            self.assertEqual(
                entry["expected_sha256"],
                hashlib.sha256(before).hexdigest(),
            )
            self.assertEqual(artifact.read_bytes(), before)

    def test_missing_root_is_explicit_and_output_is_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.json"
            template.write_text(json.dumps({
                "schema": "evsp-dr-cross-generation-input-manifest-v1",
                "artifacts": [],
                "collection_requests": [{
                    "request_id": "missing",
                    "root_alias": "missing",
                    "glob": "**/*.csv",
                    "artifact_type": "heuristic_dp_current_csv",
                }],
            }))
            result = collect(template, {})
            self.assertEqual(
                result["collection_report"][0]["status"],
                "root_not_supplied",
            )
            output = root / "resolved.json"
            rc = main([
                "--template", str(template),
                "--out-manifest", str(output),
            ])
            self.assertEqual(rc, 0)
            with self.assertRaises(FileExistsError):
                main([
                    "--template", str(template),
                    "--out-manifest", str(output),
                ])

    def test_collector_rejects_escape_glob_and_symlinked_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            outside = root / "outside"
            source.mkdir()
            outside.mkdir()
            (outside / "escape.csv").write_text("x\n")
            template = root / "template.json"
            template.write_text(json.dumps({
                "schema": "evsp-dr-cross-generation-input-manifest-v1",
                "artifacts": [],
                "collection_requests": [{
                    "request_id": "escape",
                    "root_alias": "source",
                    "glob": "../outside/*.csv",
                    "artifact_type": "heuristic_dp_current_csv",
                }],
            }))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                collect(template, {"source": source})

            template.write_text(json.dumps({
                "schema": "evsp-dr-cross-generation-input-manifest-v1",
                "artifacts": [],
                "collection_requests": [{
                    "request_id": "safe",
                    "root_alias": "source",
                    "glob": "*.csv",
                    "artifact_type": "heuristic_dp_current_csv",
                }],
            }))
            (source / "link.csv").symlink_to(outside / "escape.csv")
            result = collect(template, {"source": source})
            self.assertEqual(result["artifacts"], [])
            self.assertEqual(
                result["collection_report"][0]["status"], "no_matches"
            )


if __name__ == "__main__":
    unittest.main()
