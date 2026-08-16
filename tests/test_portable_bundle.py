import errno
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from portable_bundle import (  # noqa: E402
    BundleExistsError,
    IncompleteBundleError,
    capability_probe,
    inspect_bundle,
    publish_bundle,
)


class PortableBundleTests(unittest.TestCase):
    def test_probe_succeeds_when_renameat2_is_unsupported(self):
        for error in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp:
                result = capability_probe(
                    Path(tmp),
                    renameat2_probe=lambda _path, error=error: {
                        "supported": False,
                        "errno": errno.errorcode[error],
                    },
                )
                self.assertEqual(
                    result["portable_protocol"], "complete_valid"
                )
                self.assertEqual(
                    result["legacy_renameat2"]["errno"],
                    errno.errorcode[error],
                )

    def test_completion_marker_is_last_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            result = publish_bundle(
                bundle,
                members={"result.json": b'{"ok":true}\n'},
                metadata={"kind": "test"},
            )
            self.assertEqual(result["state"], "complete_valid")
            self.assertTrue((bundle / "result.json").is_file())
            self.assertTrue((bundle / "completion.json").is_file())
            with self.assertRaises(BundleExistsError):
                publish_bundle(
                    bundle,
                    members={"result.json": b'{"ok":true}\n'},
                    metadata={"kind": "test"},
                )

    def test_interruption_before_and_after_result_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = root / "before"
            with self.assertRaisesRegex(RuntimeError, "reservation"):
                publish_bundle(
                    before,
                    members={"result.json": b'{"ok":true}\n'},
                    metadata={},
                    fault_at="after_reservation",
                )
            state = inspect_bundle(before)
            self.assertEqual(state["state"], "incomplete_publication")
            self.assertFalse((before / "completion.json").exists())

            after = root / "after"
            with self.assertRaisesRegex(RuntimeError, "result.json"):
                publish_bundle(
                    after,
                    members={"result.json": b'{"ok":true}\n'},
                    metadata={},
                    fault_at="after_member:result.json",
                )
            state = inspect_bundle(
                after,
                required_members=("result.json",),
                recoverable_validator=lambda payload: (
                    None if payload.get("ok") is True
                    else (_ for _ in ()).throw(ValueError("not ok"))
                ),
            )
            self.assertEqual(state["state"], "recoverable_validated")
            self.assertFalse((after / "completion.json").exists())

    def test_malformed_and_hash_mismatched_results_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            (bundle / "result.json").write_text("not json")
            state = inspect_bundle(
                bundle,
                recoverable_validator=lambda _payload: None,
            )
            self.assertEqual(state["state"], "invalid")
            self.assertFalse(state["recoverable"])

            (bundle / "result.json").write_text('{"ok":true}\n')
            publish_bundle(
                bundle,
                members={"result.json": b'{"ok":true}\n'},
                metadata={},
                allow_existing_incomplete=True,
            )
            (bundle / "result.json").write_text('{"ok":false}\n')
            state = inspect_bundle(bundle, required_members=("result.json",))
            self.assertEqual(state["state"], "invalid")
            self.assertTrue(any(
                "hash/size mismatch" in error for error in state["errors"]
            ))

    def test_safe_idempotent_recovery_of_incomplete_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            with self.assertRaises(RuntimeError):
                publish_bundle(
                    bundle,
                    members={"result.json": b'{"value":1}\n'},
                    metadata={"attempt": 1},
                    fault_at="before_completion",
                )
            result = publish_bundle(
                bundle,
                members={"result.json": b'{"value":1}\n'},
                metadata={"attempt": 1},
                allow_existing_incomplete=True,
            )
            self.assertEqual(result["state"], "complete_valid")
            with self.assertRaises(BundleExistsError):
                publish_bundle(
                    bundle,
                    members={"result.json": b'{"value":1}\n'},
                    metadata={"attempt": 1},
                    allow_existing_incomplete=True,
                )

    def test_existing_different_incomplete_member_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            (bundle / "result.json").write_text('{"value":1}\n')
            with self.assertRaises(IncompleteBundleError):
                publish_bundle(
                    bundle,
                    members={"result.json": b'{"value":2}\n'},
                    metadata={},
                    allow_existing_incomplete=True,
                )


if __name__ == "__main__":
    unittest.main()
