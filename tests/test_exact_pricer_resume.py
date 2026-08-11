import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import exact_pricer_expanded as exact  # noqa: E402
from durable_io import DurableFileError  # noqa: E402


class ExactPricerResumeTests(unittest.TestCase):
    def _args(self, out: Path, *, snapshot_marks=""):
        return Namespace(
            csv="instance.csv",
            prices_csv="prices.csv",
            soc_step=5.0,
            block_min=10,
            max_iters=1,
            columns_per_iter=1,
            rc_eps=1e-4,
            master_sense="partition",
            stall_window_min=None,
            stall_rc_frac=0.05,
            stall_obj_frac=1e-5,
            wall_limit_s=3600,
            checkpoint_every=25,
            g_kwh=300.0,
            charge_kw=300.0,
            min_soc_frac=0.0,
            diversify_rounds=0,
            diversify_delta=0.15,
            snapshot_at_minutes=snapshot_marks,
            resume=True,
            out=out,
        )

    @staticmethod
    def _status(*, stop_reason="running", include_hashes=True):
        provenance = {}
        if include_hashes:
            provenance = {
                "instance_sha256": "instance-hash",
                "prices_sha256": "prices-hash",
            }
        return {
            "csv": "instance.csv",
            "prices_csv": "prices.csv",
            "soc_step": 5.0,
            "block_min": 10,
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "min_soc_frac": 0.0,
            "master_sense": "partition",
            "trip_ids": [],
            "iterations": 1,
            "columns": 1,
            "wall_s": 3600.0,
            "stop_reason": stop_reason,
            "final_lp": None,
            "provenance": provenance,
        }

    @staticmethod
    def _record():
        return {
            "trips": [],
            "cost": 100000.0,
            "route_nodes": ["Depot", "Depot"],
            "charging_stops": {
                "stations": [], "cst": [], "cet": [], "kwh": [],
            },
            "charges_started": 0,
        }

    def _run_with_lightweight_problem(self, args, *, trips=None):
        trip_ids = list(trips or [])
        problem = SimpleNamespace(trips=trip_ids, adjacency={})
        network = SimpleNamespace(node_meta=[], n_arcs=0)
        provenance = {
            "instance_sha256": "instance-hash",
            "prices_sha256": "prices-hash",
        }
        with (
            patch.object(exact, "build_problem", return_value=problem),
            patch.object(exact, "load_station_hourly_prices", return_value={}),
            patch.object(exact, "ExpandedNetwork", return_value=network),
            patch.object(exact, "_provenance", return_value=provenance),
            patch.object(
                exact, "direct_singleton_seed_records",
                return_value=([], trip_ids),
            ),
            patch.object(
                exact,
                "solve_restricted_master_lp",
                side_effect=RuntimeError("not needed in resume test"),
            ),
        ):
            return exact.run_cg(args)

    def test_resume_refuses_nonempty_journal_without_prior_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            journal = Path(str(out) + ".columns.jsonl")
            original = json.dumps(self._record()) + "\n"
            journal.write_text(original)

            with self.assertRaisesRegex(
                    DurableFileError, "prior status is missing or incompatible"):
                self._run_with_lightweight_problem(self._args(out))

            self.assertEqual(journal.read_text(), original)

    def test_resume_requires_both_exact_input_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            out.write_text(json.dumps(self._status(include_hashes=False)))
            journal = Path(str(out) + ".columns.jsonl")
            journal.write_text(json.dumps(self._record()) + "\n")

            with self.assertRaisesRegex(
                    DurableFileError, "prior status is missing or incompatible"):
                self._run_with_lightweight_problem(self._args(out))

    def test_resume_rejects_a_changed_input_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            status = self._status()
            status["provenance"]["prices_sha256"] = "changed-price-hash"
            out.write_text(json.dumps(status))
            journal = Path(str(out) + ".columns.jsonl")
            journal.write_text(json.dumps(self._record()) + "\n")

            with self.assertRaisesRegex(
                    DurableFileError, "prior status is missing or incompatible"):
                self._run_with_lightweight_problem(self._args(out))

    def test_resume_accepts_journal_ahead_of_last_status_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            status = self._status()
            status["trip_ids"] = [1, 2]
            out.write_text(json.dumps(status))
            journal = Path(str(out) + ".columns.jsonl")
            first = dict(self._record())
            first["trips"] = [1]
            extra = dict(self._record())
            extra["trips"] = [2]
            journal.write_text(
                json.dumps(first) + "\n" + json.dumps(extra) + "\n"
            )

            result = self._run_with_lightweight_problem(
                self._args(out), trips=[1, 2]
            )

            self.assertEqual(result["columns"], 2)

    def test_recovers_orphan_snapshot_only_from_matching_partial_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            status = self._status(stop_reason="snapshot_m60")
            out.write_text(json.dumps(status))
            record_text = json.dumps(self._record()) + "\n"
            Path(str(out) + ".columns.jsonl").write_text(record_text)
            snapshot = Path(tmp) / "run.m60.snapshot.json"
            orphan_journal = Path(str(snapshot) + ".columns.jsonl")
            orphan_journal.write_text(record_text)

            self._run_with_lightweight_problem(
                self._args(out, snapshot_marks="60")
            )

            recovered = json.loads(snapshot.read_text())
            self.assertEqual(recovered["stop_reason"], "snapshot_m60")
            self.assertEqual(recovered["snapshot_mark_minutes"], 60.0)
            self.assertEqual(
                recovered["columns_journal"], str(orphan_journal)
            )
            self.assertEqual(orphan_journal.read_text(), record_text)

    def test_refuses_orphan_snapshot_with_nonmatching_partial_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            out.write_text(json.dumps(self._status(stop_reason="running")))
            record_text = json.dumps(self._record()) + "\n"
            Path(str(out) + ".columns.jsonl").write_text(record_text)
            snapshot = Path(tmp) / "run.m60.snapshot.json"
            Path(str(snapshot) + ".columns.jsonl").write_text(record_text)

            with self.assertRaisesRegex(
                    DurableFileError, "orphan snapshot journal"):
                self._run_with_lightweight_problem(
                    self._args(out, snapshot_marks="60")
                )
            self.assertFalse(snapshot.exists())


if __name__ == "__main__":
    unittest.main()
