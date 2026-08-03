import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from unicorn_preflight import validate_python_runtime  # noqa: E402


class PythonRuntimeTests(unittest.TestCase):
    def test_accepts_python_312_patch_releases(self):
        self.assertEqual(
            validate_python_runtime((3, 12, 0), "3.12.0"),
            (3, 12),
        )
        self.assertEqual(
            validate_python_runtime((3, 12, 13), "3.12.13"),
            (3, 12),
        )

    def test_rejects_older_and_unvalidated_newer_minor_versions(self):
        for version_info, version_text in (
            ((3, 10, 6), "3.10.6"),
            ((3, 11, 9), "3.11.9"),
            ((3, 13, 0), "3.13.0"),
        ):
            with self.subTest(version=version_text):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"Python 3\.12\.x required; found {version_text}",
                ):
                    validate_python_runtime(version_info, version_text)


if __name__ == "__main__":
    unittest.main()
