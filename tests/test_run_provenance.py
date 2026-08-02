import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_provenance import mismatches, worktree_content_fingerprint  # noqa: E402


class WorktreeFingerprintTests(unittest.TestCase):
    def git(self, root: Path, *args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def make_repo(self, root: Path) -> None:
        self.git(root, "init", "-q")
        self.git(root, "config", "user.email", "test@example.com")
        self.git(root, "config", "user.name", "Test")
        (root / ".gitignore").write_text("ignored/\n", encoding="utf-8")
        (root / "tracked.txt").write_text("base\n", encoding="utf-8")
        self.git(root, "add", ".gitignore", "tracked.txt")
        self.git(root, "commit", "-qm", "base")

    def test_fingerprint_covers_tracked_and_nonignored_untracked_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_repo(root)
            clean = worktree_content_fingerprint(root)
            self.assertEqual(clean, worktree_content_fingerprint(root))

            (root / "tracked.txt").write_text("changed\n", encoding="utf-8")
            tracked_changed = worktree_content_fingerprint(root)
            self.assertNotEqual(clean, tracked_changed)

            (root / "note.txt").write_text("one\n", encoding="utf-8")
            untracked_one = worktree_content_fingerprint(root)
            self.assertNotEqual(tracked_changed, untracked_one)
            (root / "note.txt").write_text("two\n", encoding="utf-8")
            self.assertNotEqual(untracked_one, worktree_content_fingerprint(root))

    def test_ignored_outputs_do_not_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_repo(root)
            before = worktree_content_fingerprint(root)
            (root / "ignored").mkdir()
            (root / "ignored" / "pool.json").write_text("large output\n", encoding="utf-8")
            self.assertEqual(before, worktree_content_fingerprint(root))

    def test_mismatch_policy_can_distinguish_missing_fields(self):
        checkpoint = {"commit": "old"}
        expected = {"commit": "new", "worktree_fingerprint": "abc"}
        self.assertEqual(
            mismatches(checkpoint, expected),
            {"commit": ("old", "new")},
        )
        self.assertEqual(
            mismatches(checkpoint, expected, compare_missing=True),
            {
                "commit": ("old", "new"),
                "worktree_fingerprint": (None, "abc"),
            },
        )


if __name__ == "__main__":
    unittest.main()
