import errno
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_missing_required_member_is_incomplete_not_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            (bundle / "result.json").write_text('{"ok":true}\n')
            state = inspect_bundle(
                bundle,
                required_members=("result.json", "metrics.csv"),
                recoverable_validator=lambda _payload: None,
            )
            self.assertEqual(state["state"], "incomplete_publication")
            self.assertFalse(state["recoverable"])

    def test_invalid_completion_is_rejected_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            (bundle / "completion.json").write_text("{}")
            with self.assertRaises(IncompleteBundleError):
                publish_bundle(
                    bundle,
                    members={"result.json": b'{"ok":true}\n'},
                    metadata={},
                    allow_existing_incomplete=True,
                )
            self.assertFalse((bundle / "result.json").exists())

    def test_symlinked_destination_member_and_completion_are_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "target"
            target.mkdir()
            destination = root / "bundle"
            destination.symlink_to(target, target_is_directory=True)
            with self.assertRaises(IncompleteBundleError):
                publish_bundle(
                    destination,
                    members={"result.json": b'{"ok":true}\n'},
                    metadata={},
                )
            self.assertFalse((target / "result.json").exists())

            destination.unlink()
            destination.mkdir()
            outside = root / "outside.json"
            outside.write_text('{"ok":true}\n')
            (destination / "result.json").symlink_to(outside)
            state = inspect_bundle(
                destination,
                recoverable_validator=lambda _payload: None,
            )
            self.assertEqual(state["state"], "invalid")

            (destination / "result.json").unlink()
            (destination / "result.json").write_text('{"ok":true}\n')
            (destination / "completion.json").symlink_to(outside)
            state = inspect_bundle(destination)
            self.assertEqual(state["state"], "invalid")
            state = inspect_bundle(
                destination, result_member="../outside.json"
            )
            self.assertEqual(state["state"], "invalid")

            lock_bundle = root / "lock-bundle"
            lock_bundle.mkdir()
            (lock_bundle / ".publication.lock").symlink_to(outside)
            state = inspect_bundle(lock_bundle)
            self.assertEqual(state["state"], "invalid")

    def test_concurrent_member_creation_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            original_link = __import__("os").link

            def race_link(source, destination, **kwargs):
                if destination == "result.json":
                    descriptor = os.open(
                        destination,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=kwargs["dst_dir_fd"],
                    )
                    with os.fdopen(descriptor, "w") as handle:
                        handle.write('{"racer":true}\n')
                return original_link(source, destination, **kwargs)

            with (
                patch("portable_bundle.os.link", side_effect=race_link),
                self.assertRaises(FileExistsError),
            ):
                publish_bundle(
                    bundle,
                    members={"result.json": b'{"ours":true}\n'},
                    metadata={},
                )
            self.assertEqual(
                (bundle / "result.json").read_text(),
                '{"racer":true}\n',
            )
            self.assertFalse((bundle / "completion.json").exists())

    def test_reserved_names_and_non_strict_completion_types_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                publish_bundle(
                    root / "reserved",
                    members={".publication.lock": b"x"},
                    metadata={},
                )
            bundle = root / "typed"
            bundle.mkdir()
            (bundle / "result.json").write_text("{}")
            (bundle / "completion.json").write_text(json.dumps({
                "schema": "evsp-dr-portable-bundle-completion-v1",
                "protocol": {
                    "destination_reserved_by": "mkdir",
                    "member_publication":
                        "same-directory-temp-plus-hardlink-noreplace",
                    "commit_marker": "completion.json-published-last",
                    "renameat2_required": 0,
                },
                "members": {
                    "result.json": {
                        "sha256": hashlib.sha256(b"{}").hexdigest(),
                        "size": True,
                    }
                },
                "metadata": {"bad": float("nan")},
            }))
            self.assertEqual(inspect_bundle(bundle)["state"], "invalid")

    def test_recoverable_validator_exception_is_classified_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            bundle.mkdir()
            (bundle / "result.json").write_text("{}")
            state = inspect_bundle(
                bundle,
                required_members=("result.json",),
                recoverable_validator=lambda payload: payload["required"],
            )
            self.assertEqual(state["state"], "invalid")
            self.assertFalse(state["recoverable"])
            (bundle / "completion.json").write_text(
                '{"schema":"evsp-dr-portable-bundle-completion-v1",'
                '"protocol":{"destination_reserved_by":"mkdir",'
                '"member_publication":"same-directory-temp-plus-hardlink-noreplace",'
                '"commit_marker":"completion.json-published-last",'
                '"renameat2_required":true,"renameat2_required":false},'
                '"members":{"result.json":{"sha256":"'
                + hashlib.sha256(b"{}").hexdigest()
                + '","size":2}},"metadata":{}}'
            )
            self.assertEqual(inspect_bundle(bundle)["state"], "invalid")


if __name__ == "__main__":
    unittest.main()
