import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mip_convergence import (  # noqa: E402
    GurobiProgressObserver,
    MIPProgressRecorder,
    checkpoint_schedule_s,
    route_vector_hash,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class MIPConvergenceTests(unittest.TestCase):
    def _recorder(self, root, *, limit=7200):
        clock = FakeClock()
        recorder = MIPProgressRecorder(
            root,
            time_limit_s=limit,
            metadata={
                "source_result_sha256": "a" * 64,
                "source_journal_sha256": "b" * 64,
                "source_initial_partition_sha256": "c" * 64,
                "gurobi_version": "test",
                "parameters": {"threads": 8},
                "git_commit": "d" * 40,
                "experiment_arm": "D",
            },
            clock=clock,
        )
        return recorder, clock

    def test_exact_checkpoint_cadence(self):
        self.assertEqual(
            checkpoint_schedule_s(4 * 3600),
            [0.0, 300.0, 900.0, 1800.0, 3600.0,
             7200.0, 10800.0, 14400.0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            recorder, clock = self._recorder(Path(tmp) / "progress")
            recorder.transition_stage("fleet", elapsed_s=0.0)
            recorder.emit_zero()
            clock.now = 7200.0
            recorder.observe_stats(
                elapsed_s=7200.0,
                stage_elapsed_s=7200.0,
                fleet_bound=3.0,
                node_count=100,
                solution_count=0,
            )
            recorder.finalize(
                elapsed_s=7200.0,
                final={"status_name": "TIME_LIMIT"},
            )
            checkpoints = sorted(
                recorder.directory.glob("checkpoint_*.json")
            )
            self.assertEqual(
                [path.name for path in checkpoints],
                [
                    "checkpoint_0000m.json",
                    "checkpoint_0005m.json",
                    "checkpoint_0015m.json",
                    "checkpoint_0030m.json",
                    "checkpoint_0060m.json",
                    "checkpoint_0120m.json",
                ],
            )
            early = json.loads(
                (recorder.directory / "checkpoint_0005m.json").read_text()
            )
            self.assertIsNone(
                early["latest_statistics"]["fleet_bound"]
            )
            self.assertEqual(early["latest_statistics_observed_s"], 0.0)
            current = json.loads(
                (recorder.directory / "checkpoint_0120m.json").read_text()
            )
            self.assertEqual(
                current["latest_statistics_observed_s"], 7200.0
            )

    def test_multiple_fleet_improvements_between_marks(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder, clock = self._recorder(Path(tmp) / "progress", limit=300)
            recorder.transition_stage("fleet", elapsed_s=0.0)
            recorder.emit_zero()
            for elapsed, fleet, indices in (
                (100.0, 5, [0, 1, 2, 3, 4]),
                (200.0, 4, [0, 1, 2, 3]),
                (250.0, 3, [0, 1, 2]),
            ):
                clock.now = elapsed
                recorder.record_incumbent(
                    indices,
                    fleet=fleet,
                    objective=fleet * 100000.0,
                    elapsed_s=elapsed,
                    stage_elapsed_s=elapsed,
                )
            clock.now = 300.0
            recorder.observe_stats(
                elapsed_s=300.0,
                stage_elapsed_s=300.0,
                fleet_bound=2.0,
                fleet_gap=1 / 3,
                node_count=20,
                solution_count=3,
            )
            checkpoint = json.loads(
                (recorder.directory / "checkpoint_0005m.json").read_text()
            )
            self.assertEqual(
                [event["fleet"] for event in checkpoint[
                    "incumbent_improvements"
                ]],
                [5, 4, 3],
            )
            self.assertEqual(
                checkpoint["incumbent_state"],
                "reused_most_recent_earlier_incumbent",
            )
            self.assertEqual(checkpoint["incumbent"]["fleet"], 3)

    def test_no_incumbent_and_interrupted_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder, clock = self._recorder(Path(tmp) / "progress", limit=900)
            recorder.transition_stage("fleet", elapsed_s=0.0)
            recorder.emit_zero()
            clock.now = 120.0
            recorder.finalize(
                elapsed_s=120.0,
                final={
                    "status_name": "INTERRUPTED",
                    "incumbent_found": False,
                    "termination_signal": "SIGUSR1",
                },
            )
            final = json.loads(
                (recorder.directory / "final.json").read_text()
            )
            self.assertEqual(final["incumbent_state"], "no_incumbent_yet")
            future = json.loads(
                (recorder.directory / "checkpoint_0015m.json").read_text()
            )
            self.assertTrue(future["solver_ended_before_checkpoint"])
            self.assertFalse(future["gurobi_tree_restart_supported"])

    def test_validated_initial_incumbent_and_stage_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder, clock = self._recorder(Path(tmp) / "progress", limit=300)
            recorder.transition_stage("fleet", elapsed_s=0.0)
            recorder.record_initial_incumbent(
                [3, 1],
                objective=200005.0,
                fleet=2,
                kind="validated_partition_at_t0",
            )
            recorder.emit_zero()
            zero = json.loads(
                (recorder.directory / "checkpoint_0000m.json").read_text()
            )
            self.assertEqual(
                zero["incumbent_state"],
                "current_incumbent_at_checkpoint",
            )
            self.assertEqual(
                zero["incumbent"]["route_vector_sha256"],
                route_vector_hash([1, 3]),
            )
            clock.now = 100.0
            recorder.transition_stage("cost", elapsed_s=100.0)
            latest = json.loads(
                (recorder.directory / "latest.json").read_text()
            )
            self.assertEqual(latest["stage"], "cost")
            self.assertEqual(latest["incumbent"]["stage"], "fleet")

    def test_atomic_publication_and_no_clobber_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "progress"
            recorder, _clock = self._recorder(directory, limit=0)
            recorder.transition_stage("fleet", elapsed_s=0.0)
            recorder.emit_zero()
            self.assertTrue((directory / "latest.json").is_file())
            self.assertFalse(list(directory.glob("*.tmp*")))
            with self.assertRaises(FileExistsError):
                self._recorder(directory, limit=0)

    def test_fake_mipsol_callback_caches_route_vector(self):
        class Callback:
            MIPSOL = 1
            MIPSOL_OBJBND = 11
            MIPSOL_NODCNT = 12
            MIPSOL_SOLCNT = 13

        class GRB:
            pass

        GRB.Callback = Callback

        class Model:
            values = {
                Callback.MIPSOL_OBJBND: 1.0,
                Callback.MIPSOL_NODCNT: 7.0,
                Callback.MIPSOL_SOLCNT: 1,
            }

            def cbGet(self, key):
                return self.values[key]

            def cbGetSolution(self, _variables):
                return [1.0, 0.0, 1.0]

        with tempfile.TemporaryDirectory() as tmp:
            recorder, clock = self._recorder(Path(tmp) / "progress", limit=300)
            recorder.transition_stage("fleet", elapsed_s=0.0)
            recorder.emit_zero()
            clock.now = 10.0
            observer = GurobiProgressObserver(
                recorder,
                GRB=GRB,
                variables=[object(), object(), object()],
                routes=[
                    {"cost": 100001.0},
                    {"cost": 100002.0},
                    {"cost": 100003.0},
                ],
                bus_cost=100000.0,
                stage="fleet",
            )
            observer(Model(), Callback.MIPSOL)
            self.assertEqual(recorder.latest_incumbent["fleet"], 2)
            self.assertEqual(
                recorder.latest_incumbent["selected_route_indices"], [0, 2]
            )
            self.assertEqual(
                recorder.latest_stats["node_count"], 7.0
            )

    def test_dict_solution_aligns_routes_and_separates_statistics_incumbent(self):
        class Callback:
            MIPSOL = 1
            MIP = 2
            MIPSOL_OBJBST = 10
            MIPSOL_OBJBND = 11
            MIPSOL_NODCNT = 12
            MIPSOL_SOLCNT = 13
            MIP_OBJBST = 20
            MIP_OBJBND = 21
            MIP_NODCNT = 22
            MIP_SOLCNT = 23

        class GRB:
            pass

        GRB.Callback = Callback

        class Model:
            values = {
                Callback.MIPSOL_OBJBST: 2.0,
                Callback.MIPSOL_OBJBND: 1.0,
                Callback.MIPSOL_NODCNT: 7.0,
                Callback.MIPSOL_SOLCNT: 1,
                Callback.MIP_OBJBST: 2.0,
                Callback.MIP_OBJBND: 1.0,
                Callback.MIP_NODCNT: 9.0,
                Callback.MIP_SOLCNT: 1,
            }

            def cbGet(self, key):
                return self.values[key]

            def cbGetSolution(self, variables):
                self.observed_variables = variables
                return {2: 1.0, 0: 0.0, 1: 1.0}

        with tempfile.TemporaryDirectory() as tmp:
            recorder, clock = self._recorder(
                Path(tmp) / "progress", limit=300
            )
            recorder.transition_stage("fleet", elapsed_s=0.0)
            recorder.record_initial_incumbent(
                [0, 1, 2],
                objective=300006.0,
                fleet=3,
                kind="validated_partition_at_t0",
            )
            recorder.emit_zero()
            variables = {
                2: object(),
                0: object(),
                1: object(),
            }
            observer = GurobiProgressObserver(
                recorder,
                GRB=GRB,
                variables=variables,
                routes=[
                    {"cost": 100001.0},
                    {"cost": 100002.0},
                    {"cost": 100003.0},
                ],
                bus_cost=100000.0,
                stage="fleet",
                statistics_throttle_s=0.0,
            )
            model = Model()
            clock.now = 10.0
            observer(model, Callback.MIPSOL)
            self.assertIs(model.observed_variables, variables)
            self.assertEqual(
                recorder.latest_incumbent["selected_route_indices"], [1, 2]
            )
            self.assertEqual(recorder.latest_incumbent["fleet"], 2)
            self.assertEqual(
                [event["fleet"] for event in recorder.incumbent_events],
                [3, 2],
            )

            clock.now = 20.0
            observer(model, Callback.MIP)
            self.assertEqual(
                recorder.latest_stats["statistics_incumbent_fleet"], 2.0
            )
            self.assertEqual(recorder.latest_stats["fleet_bound"], 1.0)
            self.assertAlmostEqual(recorder.latest_stats["fleet_gap"], 0.5)

            latest = json.loads(
                (recorder.directory / "latest.json").read_text()
            )
            self.assertEqual(latest["incumbent"]["fleet"], 2)
            self.assertEqual(
                latest["latest_statistics"]["statistics_incumbent_fleet"],
                2.0,
            )
            self.assertAlmostEqual(
                latest["latest_statistics"]["fleet_gap"],
                (
                    latest["latest_statistics"][
                        "statistics_incumbent_fleet"
                    ]
                    - latest["latest_statistics"]["fleet_bound"]
                )
                / latest["latest_statistics"][
                    "statistics_incumbent_fleet"
                ],
            )

    def test_gurobi_infinity_sentinel_is_not_a_finite_bound(self):
        self.assertIsNone(MIPProgressRecorder._finite(1e100))
        self.assertIsNone(MIPProgressRecorder._finite(-1e100))


if __name__ == "__main__":
    unittest.main()
