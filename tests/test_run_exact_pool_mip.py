import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_exact_pool_mip import load_pool, main  # noqa: E402


class ExactPoolMipTests(unittest.TestCase):
    def test_copied_snapshot_finds_adjacent_recorded_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = folder / "sample.snapshot.json"
            journal = folder / "sample.columns.jsonl"
            result.write_text(json.dumps({
                "csv": "sample.csv",
                "soc_step": 5,
                "trip_ids": [11, 12],
                "columns_journal": "/unavailable/cluster/sample.columns.jsonl",
            }))
            journal.write_text(
                json.dumps({"trips": [11], "cost": 1.0}) + "\n" +
                json.dumps({"trips": [12], "cost": 1.0}) + "\n"
            )

            _, routes, trips = load_pool(result)

            self.assertEqual(trips, [11, 12])
            self.assertEqual(len(routes), 2)

    def test_snapshot_prefers_frozen_sibling_over_recorded_live_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            live_folder = folder / "live"
            frozen_folder = folder / "frozen"
            live_folder.mkdir()
            frozen_folder.mkdir()
            live_journal = live_folder / "sample.columns.jsonl"
            frozen_journal = frozen_folder / "sample.columns.jsonl"
            result = frozen_folder / "sample.snapshot.json"
            result.write_text(json.dumps({
                "csv": "sample.csv",
                "soc_step": 5,
                "trip_ids": [11],
                "columns_journal": str(live_journal),
            }))
            live_journal.write_text(
                json.dumps({"trips": [11], "cost": 99.0}) + "\n"
            )
            frozen_journal.write_text(
                json.dumps({"trips": [11], "cost": 1.0}) + "\n"
            )

            _, routes, _ = load_pool(result)

            self.assertEqual(routes[0]["cost"], 1.0)

    def test_runner_refuses_to_overwrite_input_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "sample.json"
            result.write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--result", str(result), "--out", str(result),
                          "--validate-only"])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
