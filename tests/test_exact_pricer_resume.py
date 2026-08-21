import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
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
            initial_pool="singletons",
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
            phase_telemetry=None,
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
            "initial_pool": "singletons",
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
                    DurableFileError, "before modifying persisted artifacts"):
                self._run_with_lightweight_problem(self._args(out))

            self.assertEqual(journal.read_text(), original)

    def test_resume_requires_both_exact_input_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            out.write_text(json.dumps(self._status(include_hashes=False)))
            journal = Path(str(out) + ".columns.jsonl")
            journal.write_text(json.dumps(self._record()) + "\n")

            with self.assertRaisesRegex(
                    DurableFileError, "before modifying persisted artifacts"):
                self._run_with_lightweight_problem(self._args(out))

    def test_attested_legacy_resume_requires_commit_identity(self):
        status = self._status()
        status["resume_parent"] = {
            "schema": "evsp-dr-legacy-exact-pool-migration-v1"
        }
        args = self._args(Path("run.json"))

        mismatches = exact.resume_identity_mismatches(
            status,
            args,
            [],
            {
                "instance_sha256": "instance-hash",
                "prices_sha256": "prices-hash",
            },
        )

        self.assertIn(
            "attested legacy migration is missing saved or current "
            "git_commit identity",
            mismatches,
        )

    def test_resume_rejects_changed_initial_pool_mode(self):
        status = self._status()
        args = self._args(Path("run.json"))
        args.initial_pool = "artificial"

        mismatches = exact.resume_identity_mismatches(
            status,
            args,
            [],
            {
                "instance_sha256": "instance-hash",
                "prices_sha256": "prices-hash",
            },
        )

        self.assertIn(
            "initial_pool differs (saved='singletons', "
            "current='artificial')",
            mismatches,
        )

    def test_legacy_missing_initial_pool_is_singleton_only(self):
        status = self._status()
        del status["initial_pool"]
        args = self._args(Path("run.json"))
        provenance = {
            "instance_sha256": "instance-hash",
            "prices_sha256": "prices-hash",
        }

        self.assertEqual(
            exact.resume_identity_mismatches(status, args, [], provenance),
            [],
        )

        args.initial_pool = "artificial"
        self.assertIn(
            "initial_pool differs (saved='singletons', "
            "current='artificial')",
            exact.resume_identity_mismatches(status, args, [], provenance),
        )

    def test_resume_rejects_changed_hash_without_repairing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            status = self._status()
            status["provenance"]["prices_sha256"] = "changed-price-hash"
            out.write_text(json.dumps(status))
            journal = Path(str(out) + ".columns.jsonl")
            journal_original = (
                json.dumps(self._record()) + "\n" + '{"trips":'
            )
            journal.write_text(journal_original)
            iters = Path(str(out) + ".iters.csv")
            iters_original = (
                exact.ITERATION_LOG_HEADER + "\n"
                "10,1,100000,1,0,-1,1\n"
                "11,2,"
            )
            iters.write_text(iters_original)

            with self.assertRaisesRegex(
                    DurableFileError, "prices_sha256 differs"):
                self._run_with_lightweight_problem(self._args(out))

            self.assertEqual(journal.read_text(), journal_original)
            self.assertEqual(iters.read_text(), iters_original)

    def test_resume_rejects_seed_hash_before_modifying_status_or_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            status = self._status(stop_reason="certified")
            status.update({
                "trip_ids": [1],
                "initial_pool_sha256": "0" * 64,
            })
            status_original = json.dumps(status)
            out.write_text(status_original)
            record = dict(self._record())
            record.update({
                "trips": [1],
                "origin": "exact_direct_singleton_seed",
                "cost_tariff_sha256": "prices-hash",
            })
            journal = Path(str(out) + ".columns.jsonl")
            journal_original = json.dumps(record) + "\n" + '{"trips":'
            journal.write_text(journal_original)

            with self.assertRaisesRegex(
                    DurableFileError, "journal hash differs"):
                self._run_with_lightweight_problem(
                    self._args(out), trips=[1]
                )

            self.assertEqual(out.read_text(), status_original)
            self.assertEqual(journal.read_text(), journal_original)

    def test_resume_rejects_regenerated_seed_before_any_repair_or_status_write(self):
        from exact_initial_pools import pool_sha256

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            persisted = dict(self._record())
            persisted.update({
                "trips": [1], "origin": "exact_matching_initial_seed",
                "initial_pool": "matching",
            })
            regenerated = {**persisted, "cost": 90000.0}
            status = self._status(stop_reason="certified")
            status.update({
                "trip_ids": [1],
                "initial_pool": "matching",
                "initial_pool_sha256": pool_sha256([persisted]),
            })
            status_original = json.dumps(status)
            out.write_text(status_original)
            journal = Path(str(out) + ".columns.jsonl")
            journal_original = json.dumps(persisted) + "\n" + '{"trips":'
            journal.write_text(journal_original)
            iters = Path(str(out) + ".iters.csv")
            iters_original = exact.ITERATION_LOG_HEADER + "\n1,1,"
            iters.write_text(iters_original)
            args = self._args(out)
            args.initial_pool = "matching"
            regenerated_provenance = {
                "generated_pool_sha256": pool_sha256([regenerated])
            }

            with patch(
                "exact_initial_pools.build_heuristic_initial_pool",
                return_value=([regenerated], regenerated_provenance),
            ), self.assertRaisesRegex(
                DurableFileError, "regenerated initial pool differs"
            ):
                self._run_with_lightweight_problem(args, trips=[1])

            self.assertEqual(out.read_text(), status_original)
            self.assertEqual(journal.read_text(), journal_original)
            self.assertEqual(iters.read_text(), iters_original)

    def test_initializing_null_hash_resume_does_not_publish_unresumable_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            status = self._status(stop_reason="initializing")
            status.update({
                "initial_pool_sha256": None,
                "columns": 0,
            })
            original = json.dumps(status)
            out.write_text(original)
            for _attempt in range(2):
                self._run_with_lightweight_problem(self._args(out))
                self.assertEqual(out.read_text(), original)

    def test_resume_rejects_status_ahead_of_missing_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            status = self._status()
            status["columns"] = 3
            out.write_text(json.dumps(status))

            with self.assertRaisesRegex(
                    DurableFileError, "journal contains only 0"):
                self._run_with_lightweight_problem(self._args(out))

            self.assertFalse(Path(str(out) + ".columns.jsonl").exists())

    def test_fresh_run_publishes_identity_before_first_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            args = self._args(out)
            args.wall_limit_s = 1

            self._run_with_lightweight_problem(args)

            initial = json.loads(out.read_text())
            self.assertEqual(initial["stop_reason"], "initializing")
            self.assertEqual(initial["columns"], 0)
            self.assertEqual(initial["initial_pool"], "singletons")
            self.assertEqual(
                initial["provenance"]["instance_sha256"], "instance-hash"
            )

    def test_artificial_initial_pool_skips_singleton_construction(self):
        args = self._args(Path("unused.json"))
        args.out = None
        args.resume = False
        args.initial_pool = "artificial"
        problem = SimpleNamespace(trips=[1], adjacency={})
        network = SimpleNamespace(
            node_meta=[], n_arcs=0,
            k_best_routes=lambda _duals, *, k: [],
        )
        provenance = {
            "instance_sha256": "instance-hash",
            "prices_sha256": "prices-hash",
        }
        output = io.StringIO()

        with (
            patch.object(exact, "build_problem", return_value=problem),
            patch.object(exact, "load_station_hourly_prices", return_value={}),
            patch.object(exact, "ExpandedNetwork", return_value=network),
            patch.object(exact, "_provenance", return_value=provenance),
            patch.object(
                exact, "direct_singleton_seed_records",
                side_effect=AssertionError("singletons must not be built"),
            ),
            patch.object(
                exact, "solve_restricted_master_lp",
                side_effect=AssertionError("empty real pool needs no master"),
            ),
            redirect_stdout(output),
        ):
            result = exact.run_cg(args)

        self.assertEqual(result["initial_pool"], "artificial")
        self.assertEqual(result["columns"], 0)
        self.assertEqual(result["final"]["route_weight"], 0.0)
        self.assertEqual(result["final"]["artificials"], 1.0)
        self.assertIn("initial pool: artificial-only", output.getvalue())

    def test_phase_telemetry_does_not_change_result_or_certification(self):
        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "phases.jsonl"

            def execute(path):
                args = self._args(Path("unused.json"))
                args.out = None
                args.resume = False
                args.initial_pool = "artificial"
                args.phase_telemetry = path
                problem = SimpleNamespace(trips=[1], adjacency={})

                def k_best(_duals, *, k, phase_callback=None):
                    if phase_callback is not None:
                        phase_callback(
                            "pricing_shortest_path", 0.1,
                            {"path_found": False},
                        )
                        phase_callback(
                            "pricing_extra_columns", 0.0,
                            {"sink_candidates": 0, "returned_routes": 0},
                        )
                    return []

                network = SimpleNamespace(
                    node_meta=[], n_arcs=0, k_best_routes=k_best,
                )
                provenance = {
                    "instance_sha256": "instance-hash",
                    "prices_sha256": "prices-hash",
                    "git_commit": "a" * 40,
                }
                with (
                    patch.object(exact, "build_problem", return_value=problem),
                    patch.object(
                        exact, "load_station_hourly_prices", return_value={}
                    ),
                    patch.object(
                        exact, "ExpandedNetwork", return_value=network
                    ),
                    patch.object(
                        exact, "_provenance", return_value=provenance
                    ),
                    patch.object(
                        exact,
                        "direct_singleton_seed_records",
                        side_effect=AssertionError(
                            "artificial mode must not build singletons"
                        ),
                    ),
                ):
                    return exact.run_cg(args)

            without = execute(None)
            with_telemetry = execute(telemetry_path)

            for payload in (without, with_telemetry):
                payload.pop("wall_s")
                payload.pop("attempt_wall_s")
            self.assertEqual(without, with_telemetry)
            self.assertFalse(with_telemetry["certified_rc_optimal"])
            records = [
                json.loads(line)
                for line in telemetry_path.read_text().splitlines()
            ]
            phases = {
                record.get("phase") for record in records
                if record.get("record_type") == "phase"
            }
            self.assertIn("network_build", phases)
            self.assertIn("pricing_shortest_path", phases)
            self.assertIn("pricing_extra_columns", phases)

    def test_telemetry_io_time_does_not_cross_wall_boundary(self):
        class _FakeTime:
            def __init__(self):
                self.now = 1000.0

            def time(self):
                return self.now

            def perf_counter(self):
                return self.now

        def execute(enabled):
            fake_time = _FakeTime()

            class SlowTelemetry:
                def __init__(self, *_args, **_kwargs):
                    self.overhead_s = 0.0

                def phase(self, *_args, **_kwargs):
                    fake_time.now += 120.0
                    self.overhead_s += 120.0

            args = self._args(Path("unused.json"))
            args.out = None
            args.resume = False
            args.initial_pool = "artificial"
            args.wall_limit_s = 120
            args.phase_telemetry = (
                Path("slow.phases.jsonl") if enabled else None
            )
            problem = SimpleNamespace(trips=[1], adjacency={})

            def k_best(_duals, *, k, phase_callback=None):
                if phase_callback is not None:
                    phase_callback(
                        "pricing_shortest_path", 0.0,
                        {"path_found": False},
                    )
                    phase_callback(
                        "pricing_extra_columns", 0.0,
                        {"sink_candidates": 0, "returned_routes": 0},
                    )
                return []

            network = SimpleNamespace(
                node_meta=[], n_arcs=0, k_best_routes=k_best,
            )
            provenance = {
                "instance_sha256": "instance-hash",
                "prices_sha256": "prices-hash",
                "git_commit": "a" * 40,
            }
            with (
                patch.object(exact, "time", fake_time),
                patch.object(exact, "PhaseTelemetry", SlowTelemetry),
                patch.object(exact, "build_problem", return_value=problem),
                patch.object(
                    exact, "load_station_hourly_prices", return_value={}
                ),
                patch.object(exact, "ExpandedNetwork", return_value=network),
                patch.object(exact, "_provenance", return_value=provenance),
                patch.object(
                    exact,
                    "direct_singleton_seed_records",
                    side_effect=AssertionError(
                        "artificial mode must not build singletons"
                    ),
                ),
            ):
                return exact.run_cg(args)

        without = execute(False)
        with_slow_telemetry = execute(True)

        self.assertEqual(without["stop_reason"], "no_path")
        self.assertEqual(with_slow_telemetry["stop_reason"], "no_path")
        self.assertEqual(without["iterations"], with_slow_telemetry["iterations"])
        self.assertEqual(without["final"], with_slow_telemetry["final"])
        self.assertEqual(without["wall_s"], with_slow_telemetry["wall_s"])

    def test_network_telemetry_failure_does_not_abort_cg(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run.json"
            status = self._status()
            status["trip_ids"] = [1]
            status["columns"] = 1
            out.write_text(json.dumps(status))
            record = dict(self._record())
            record["trips"] = [1]
            Path(str(out) + ".columns.jsonl").write_text(
                json.dumps(record) + "\n"
            )
            args = self._args(out)
            args.phase_telemetry = Path(tmp) / "phases.jsonl"
            args.wall_limit_s = None
            problem = SimpleNamespace(trips=[1], adjacency={})
            network = SimpleNamespace(
                node_meta=[],
                n_arcs=0,
                k_best_routes=lambda _duals, *, k, **_kwargs: [],
            )
            provenance = {
                "instance_sha256": "instance-hash",
                "prices_sha256": "prices-hash",
            }
            backend = SimpleNamespace(method="highs-ds")
            lp = SimpleNamespace(
                objective=100000.0,
                route_weight=1.0,
                artificial_total=0.0,
                trip_duals={1: 100000.0},
                route_values=[1.0],
                max_row_violation=0.0,
                max_bound_violation=0.0,
                feasibility_tolerance=1e-6,
                backend=backend,
                runtime_s=0.01,
            )

            class FailingTelemetry:
                overhead_s = 0.0

                def __init__(self, *_args, **_kwargs):
                    pass

                def phase(self, name, *_args, **_kwargs):
                    if name == "network_build":
                        raise OSError("synthetic telemetry failure")

            with (
                patch.object(exact, "PhaseTelemetry", FailingTelemetry),
                patch.object(exact, "build_problem", return_value=problem),
                patch.object(
                    exact, "load_station_hourly_prices", return_value={}
                ),
                patch.object(exact, "ExpandedNetwork", return_value=network),
                patch.object(exact, "_provenance", return_value=provenance),
                patch.object(
                    exact, "direct_singleton_seed_records",
                    return_value=([], [1]),
                ),
                patch.object(
                    exact, "build_route_incidence",
                    return_value=SimpleNamespace(
                        nnz=1, shape=(1, 1),
                    ),
                ),
                patch.object(
                    exact, "solve_restricted_master_lp",
                    return_value=lp,
                ),
            ):
                result = exact.run_cg(args)

            self.assertEqual(result["stop_reason"], "no_path")

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

    def _run_master_labeling_case(self, tmp, *, wall_limit_s,
                                  master_seconds_per_attempt):
        """Resume a one-column pool against a failing master under fake time.

        Returns the terminal result written by ``run_cg``.  Every master
        attempt raises after advancing the fake clock, so the run must decide
        between the honest ``wall_limit`` and ``master_failed`` labels.
        """

        out = Path(tmp) / "run.json"
        status = self._status()
        status["trip_ids"] = [1]
        status["wall_s"] = 0.0
        status["iterations"] = 0
        out.write_text(json.dumps(status))
        record = dict(self._record())
        record["trips"] = [1]
        Path(str(out) + ".columns.jsonl").write_text(
            json.dumps(record) + "\n"
        )
        args = self._args(out)
        args.wall_limit_s = wall_limit_s

        class _FakeTime:
            def __init__(self):
                self.now = 1000.0

            def time(self):
                return self.now

            def perf_counter(self):
                return self.now

        fake_time = _FakeTime()

        def _failing_master(*_args, **_kwargs):
            fake_time.now += master_seconds_per_attempt
            raise RuntimeError("synthetic master failure")

        problem = SimpleNamespace(trips=[1], adjacency={})
        network = SimpleNamespace(node_meta=[], n_arcs=0)
        provenance = {
            "instance_sha256": "instance-hash",
            "prices_sha256": "prices-hash",
        }
        with (
            patch.object(exact, "time", fake_time),
            patch.object(exact, "build_problem", return_value=problem),
            patch.object(exact, "load_station_hourly_prices", return_value={}),
            patch.object(exact, "ExpandedNetwork", return_value=network),
            patch.object(exact, "_provenance", return_value=provenance),
            patch.object(
                exact, "direct_singleton_seed_records",
                return_value=([], [1]),
            ),
            patch.object(
                exact, "build_route_incidence", return_value=None,
            ),
            patch.object(
                exact, "solve_restricted_master_lp",
                side_effect=_failing_master,
            ),
        ):
            return exact.run_cg(args)

    def test_wall_expiry_between_master_attempts_is_labeled_wall_limit(self):
        # First attempt consumes the whole budget and fails; the second
        # method sees no remaining wall time.  The stop must be recorded as
        # the graceful timed stop it is, not as a solver failure.
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_master_labeling_case(
                tmp, wall_limit_s=200, master_seconds_per_attempt=180.0,
            )
        self.assertEqual(result["stop_reason"], "wall_limit")

    def test_wall_expiry_during_final_master_attempt_is_labeled_wall_limit(self):
        # Boundary case: every method still receives a positive time limit
        # (170s, 110s, 50s), so no mid-loop exhaustion break ever fires; the
        # third and FINAL attempt consumes the remaining budget and then
        # raises.  There is no later loop iteration to notice the expiry, so
        # the exit path itself must classify this as a wall stop.
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_master_labeling_case(
                tmp, wall_limit_s=200, master_seconds_per_attempt=60.0,
            )
        self.assertEqual(result["stop_reason"], "wall_limit")

    def test_exhausting_all_master_methods_stays_labeled_master_failed(self):
        # With ample wall budget, failing every master method is a genuine
        # master failure and must keep its uncertified label.
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_master_labeling_case(
                tmp, wall_limit_s=100000, master_seconds_per_attempt=1.0,
            )
        self.assertEqual(result["stop_reason"], "master_failed")

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

    def _run_spanning_master_snapshot_case(self, tmp, *, fail_first):
        out = Path(tmp) / "run.json"
        snapshot = Path(tmp) / "run.m1440.snapshot.json"
        status = self._status()
        status.update({
            "trip_ids": [1],
            "wall_s": 86390.0,
            "iterations": 1,
            "final_lp": {
                "objective": 100000.0,
                "route_weight": 1.0,
                "artificial_total": 0.0,
                "positive_routes": [
                    {"trips": [1], "value": 1.0, "cost": 100000.0}
                ],
                "trip_duals": {"1": 100000.0},
                "source": "prior_durable_lp",
            },
        })
        out.write_text(json.dumps(status))
        record = dict(self._record())
        record["trips"] = [1]
        Path(str(out) + ".columns.jsonl").write_text(
            json.dumps(record) + "\n"
        )
        args = self._args(out, snapshot_marks="1440")
        args.wall_limit_s = 200000

        class _FakeTime:
            def __init__(self):
                self.now = 1000.0

            def time(self):
                return self.now

            def perf_counter(self):
                return self.now

        fake_time = _FakeTime()
        limits = []
        methods = []
        retry_saw_snapshot = []
        calls = 0
        solved_lp = SimpleNamespace(
            objective=90000.0,
            route_weight=1.0,
            artificial_total=0.0,
            trip_duals={1: 90000.0},
            route_values=[1.0],
            max_row_violation=0.0,
            max_bound_violation=0.0,
            feasibility_tolerance=1e-7,
            backend=SimpleNamespace(method="synthetic"),
        )

        def _master(*_args, **kwargs):
            nonlocal calls
            calls += 1
            limits.append(kwargs.get("time_limit_s"))
            methods.append(kwargs.get("method"))
            if calls == 1:
                self.assertFalse(snapshot.exists())
                fake_time.now += 15.0
                if fail_first:
                    raise RuntimeError("synthetic master timeout")
                return solved_lp
            retry_saw_snapshot.append(snapshot.exists())
            if fail_first:
                raise RuntimeError("synthetic retry failure")
            return solved_lp

        problem = SimpleNamespace(trips=[1], adjacency={})
        network = SimpleNamespace(
            node_meta=[], n_arcs=0,
            k_best_routes=lambda _duals, *, k: [],
        )
        provenance = {
            "instance_sha256": "instance-hash",
            "prices_sha256": "prices-hash",
        }
        with (
            patch.object(exact, "time", fake_time),
            patch.object(exact, "build_problem", return_value=problem),
            patch.object(exact, "load_station_hourly_prices", return_value={}),
            patch.object(exact, "ExpandedNetwork", return_value=network),
            patch.object(exact, "_provenance", return_value=provenance),
            patch.object(
                exact, "direct_singleton_seed_records",
                return_value=([], [1]),
            ),
            patch.object(exact, "build_route_incidence", return_value=None),
            patch.object(
                exact, "solve_restricted_master_lp", side_effect=_master,
            ),
        ):
            result = exact.run_cg(args)

        return result, snapshot, limits, methods, retry_saw_snapshot

    def test_master_exception_crossing_m1440_freezes_pre_attempt_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, snapshot, limits, methods, retry_saw_snapshot = (
                self._run_spanning_master_snapshot_case(
                    tmp, fail_first=True,
                )
            )
            frozen = json.loads(snapshot.read_text())
            frozen_journal = Path(frozen["columns_journal"])

            self.assertEqual(frozen["stop_reason"], "snapshot_m1440")
            self.assertEqual(frozen["columns"], 1)
            self.assertEqual(frozen["final_lp"]["objective"], 100000.0)
            self.assertEqual(
                frozen["final_lp"]["source"], "compatible_prior_result"
            )
            self.assertEqual(
                len(frozen_journal.read_text().splitlines()), 1
            )

        self.assertEqual(result["stop_reason"], "master_failed")
        self.assertGreater(limits[0], 0.0)
        self.assertLessEqual(limits[0], 10.0)
        self.assertEqual(methods[:2], ["highs-ds", "highs-ds"])
        self.assertTrue(retry_saw_snapshot)
        self.assertTrue(retry_saw_snapshot[0])

    def test_successful_master_crossing_mark_freezes_pre_solve_lp(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, snapshot, limits, methods, _ = (
                self._run_spanning_master_snapshot_case(
                    tmp, fail_first=False,
                )
            )
            frozen = json.loads(snapshot.read_text())

        self.assertLessEqual(limits[0], 10.0)
        self.assertEqual(result["stop_reason"], "no_path")
        # One successful main-loop attempt plus the expected final re-solve;
        # neither falls back to IPM/auto. ``solved_lp`` deliberately has no
        # runtime_s, proving disabled telemetry does not eagerly access it.
        self.assertEqual(methods, ["highs-ds", "highs-ds"])
        self.assertEqual(frozen["final_lp"]["objective"], 100000.0)
        self.assertEqual(
            frozen["final_lp"]["source"], "compatible_prior_result"
        )
        self.assertEqual(result["final_lp"]["objective"], 90000.0)


if __name__ == "__main__":
    unittest.main()
