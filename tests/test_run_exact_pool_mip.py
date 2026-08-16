import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_exact_pool_mip import (  # noqa: E402
    fleet_bound_proves_incumbent,
    finite_solver_value,
    greedy_partition_start_indices,
    load_pool,
    main,
    merge_validated_partition_start,
    optimal_scope,
    singleton_partition_indices,
    validate_injected_route,
    verified_mip_code_identity,
)
from audit_giro_known_columns import DEPOT  # noqa: E402


class ExactPoolMipTests(unittest.TestCase):
    @staticmethod
    def validation_problem(*, final_trip_end=100.0, return_minutes=10.0):
        return SimpleNamespace(
            adjacency={
                DEPOT: [(1, 0.0, 0.0, "depot_trip")],
                1: [(DEPOT, return_minutes, 0.0, "trip_depot")],
            },
            start_min={1: 0.0},
            end_min={1: final_trip_end},
            trip_energy={1: 0.0},
        )

    def test_singletons_are_a_strict_partition_seed(self):
        routes = [
            {"trips": [1, 2], "cost": 1.0},
            {"trips": [1], "cost": 2.0},
            {"trips": [2], "cost": 2.0},
        ]
        self.assertEqual(singleton_partition_indices(routes, [1, 2]), [1, 2])
        self.assertEqual(singleton_partition_indices(routes[:-1], [1, 2]), [])

    def test_solver_infinity_is_serialized_as_null(self):
        self.assertIsNone(finite_solver_value(float("inf")))
        self.assertIsNone(finite_solver_value(1.7976931348623157e308))
        self.assertEqual(finite_solver_value(42.5), 42.5)

    def test_submitted_solver_rejects_commit_dirty_and_branch_mismatches(self):
        def git_state(
            commit="a" * 40,
            status="",
            branch="",
            *,
            status_returncode=0,
        ):
            def value(*args):
                if args[:2] == ("rev-parse", "--verify"):
                    return subprocess.CompletedProcess(
                        args, 0, stdout=commit + "\n", stderr=""
                    )
                if args and args[0] == "status":
                    return subprocess.CompletedProcess(
                        args, status_returncode, stdout=status, stderr=""
                    )
                if args[:2] == ("symbolic-ref", "-q"):
                    return subprocess.CompletedProcess(
                        args,
                        0 if branch else 1,
                        stdout=(branch + "\n") if branch else "",
                        stderr="",
                    )
                return subprocess.CompletedProcess(
                    args, 1, stdout="", stderr="unsupported"
                )
            return value

        with (
            patch.dict(os.environ, {
                "SLURM_JOB_ID": "123",
                "EVSP_EXPECTED_COMMIT": "b" * 40,
                "EVSP_REQUIRE_DETACHED": "1",
            }, clear=False),
            patch(
                "run_exact_pool_mip.git_result",
                side_effect=git_state(),
            ),
            self.assertRaisesRegex(SystemExit, "commit mismatch"),
        ):
            verified_mip_code_identity()

        with (
            patch.dict(os.environ, {
                "SLURM_JOB_ID": "123",
                "EVSP_EXPECTED_COMMIT": "a" * 40,
                "EVSP_REQUIRE_DETACHED": "0",
            }, clear=False),
            patch(
                "run_exact_pool_mip.git_result",
                side_effect=git_state(status=" M src/run_exact_pool_mip.py"),
            ),
            self.assertRaisesRegex(SystemExit, "tracked modifications"),
        ):
            verified_mip_code_identity()

        with (
            patch.dict(os.environ, {
                "SLURM_JOB_ID": "123",
                "EVSP_EXPECTED_COMMIT": "a" * 40,
                "EVSP_REQUIRE_DETACHED": "0",
            }, clear=False),
            patch(
                "run_exact_pool_mip.git_result",
                side_effect=git_state(status_returncode=2),
            ),
            self.assertRaisesRegex(
                SystemExit, "could not verify solver worktree"
            ),
        ):
            verified_mip_code_identity()

        with (
            patch.dict(os.environ, {
                "SLURM_JOB_ID": "123",
                "EVSP_EXPECTED_COMMIT": "a" * 40,
                "EVSP_REQUIRE_DETACHED": "0",
            }, clear=False),
            patch(
                "run_exact_pool_mip.git_result",
                side_effect=git_state(branch="cursor/recovery-audit"),
            ),
            self.assertRaisesRegex(SystemExit, "must run detached"),
        ):
            verified_mip_code_identity()

    def test_integer_fleet_bound_can_prove_timeout_incumbent(self):
        self.assertTrue(fleet_bound_proves_incumbent(40, 39.001, 9))
        self.assertFalse(fleet_bound_proves_incumbent(40, 38.999, 9))
        self.assertFalse(fleet_bound_proves_incumbent(40, None, 2))
        self.assertTrue(fleet_bound_proves_incumbent(40, 40.0, 2))

    def test_optimal_scope_never_overstates_unproven_fleet(self):
        self.assertEqual(
            optimal_scope(
                two_stage=True,
                fleet_proven=False,
                cost_stage_executed=False,
                final_status=9,
            ),
            "none",
        )
        self.assertEqual(
            optimal_scope(
                two_stage=True,
                fleet_proven=True,
                cost_stage_executed=False,
                final_status=2,
            ),
            "fleet_only",
        )
        self.assertEqual(
            optimal_scope(
                two_stage=True,
                fleet_proven=True,
                cost_stage_executed=True,
                final_status=2,
            ),
            "full_pool_lexicographic",
        )

    def test_greedy_start_replaces_singletons_with_disjoint_routes(self):
        routes = [
            {"trips": [1], "cost": 100000.0},
            {"trips": [2], "cost": 100000.0},
            {"trips": [3], "cost": 100000.0},
            {"trips": [1, 2], "cost": 100010.0},
            {"trips": [2, 3], "cost": 100020.0},
        ]

        start = greedy_partition_start_indices(routes, [1, 2, 3], [0, 1, 2])

        self.assertEqual(start, [3, 2])
        covered = [trip for index in start for trip in routes[index]["trips"]]
        self.assertEqual(sorted(covered), [1, 2, 3])

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

    def test_pool_loader_rejects_repeated_trip_incidences(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "malformed.json"
            journal = Path(str(result) + ".columns.jsonl")
            result.write_text(json.dumps({
                "trip_ids": [1],
                "columns_journal": str(journal),
            }))
            journal.write_text(json.dumps({
                "trips": [1, 1],
                "cost": 100000.0,
            }) + "\n")

            with self.assertRaisesRegex(SystemExit, "repeats a trip"):
                load_pool(result)

    def test_runner_refuses_to_overwrite_input_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "sample.json"
            result.write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(["--result", str(result), "--out", str(result),
                          "--validate-only"])
            self.assertEqual(raised.exception.code, 2)

    def test_runner_enforces_submission_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "sample.snapshot.json"
            journal = Path(str(result) + ".columns.jsonl")
            result.write_text(json.dumps({
                "trip_ids": [1],
                "columns_journal": str(journal),
            }))
            journal.write_text(json.dumps({
                "trips": [1], "cost": 100000.0,
            }) + "\n")

            with (
                patch.dict(os.environ, {
                    "EVSP_MIP_EXPECTED_RESULT_SHA256": "0" * 64,
                }, clear=False),
                self.assertRaisesRegex(
                    SystemExit, "submission-manifest hash"
                ),
            ):
                main(["--result", str(result), "--validate-only"])

            identity = {
                "expected_commit": "a" * 40,
                "observed_commit": "a" * 40,
                "branch": "",
                "detached": True,
                "tracked_clean": True,
                "enforced": True,
            }
            with (
                patch.dict(os.environ, {
                    "SLURM_JOB_ID": "123",
                    "EVSP_EXPECTED_COMMIT": "a" * 40,
                    "EVSP_REQUIRE_DETACHED": "1",
                    "EVSP_MIP_EXPECTED_RESULT_SHA256": "",
                    "EVSP_MIP_EXPECTED_JOURNAL_SHA256": "",
                }, clear=False),
                patch(
                    "run_exact_pool_mip.verified_mip_code_identity",
                    return_value=identity,
                ),
                self.assertRaisesRegex(
                    SystemExit, "lacks required input hashes"
                ),
            ):
                main(["--result", str(result), "--validate-only"])

    def test_required_singleton_partition_rejects_coverage_only_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            result = folder / "triangle.json"
            journal = Path(str(result) + ".columns.jsonl")
            result.write_text(json.dumps({
                "csv": "triangle.csv",
                "soc_step": 5,
                "trip_ids": [1, 2, 3],
                "columns_journal": str(journal),
            }))
            journal.write_text("\n".join(json.dumps(route) for route in (
                {"trips": [1, 2], "cost": 1.0},
                {"trips": [1, 3], "cost": 1.0},
                {"trips": [2, 3], "cost": 1.0},
            )) + "\n")

            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    SystemExit, "singleton partition required"
                ):
                    main([
                        "--result", str(result),
                        "--require-singleton-partition",
                        "--validate-only",
                ])

    def test_validator_counts_final_travel_to_depot_against_horizon(self):
        problem = self.validation_problem(final_trip_end=100.0,
                                          return_minutes=11.0)
        verdict = validate_injected_route(
            problem,
            {"route_nodes": [DEPOT, 1, DEPOT], "charging_stops": {}},
            g_kwh=300.0,
            charge_kw=300.0,
            reserve_kwh=0.0,
            horizon_min=110.0,
        )
        self.assertRegex(verdict, r"ends at 111.*past horizon")

    def test_validator_rejects_unconsumed_charging_stop(self):
        problem = self.validation_problem()
        verdict = validate_injected_route(
            problem,
            {
                "route_nodes": [DEPOT, 1, DEPOT],
                "charging_stops": {
                    "stations": ["unused"],
                    "cst": [10.0],
                    "cet": [11.0],
                    "kwh": [1.0],
                },
            },
            g_kwh=300.0,
            charge_kw=300.0,
            reserve_kwh=0.0,
            horizon_min=200.0,
        )
        self.assertRegex(verdict, r"1 charging stop record.*not consumed")

    def test_validator_requires_depot_endpoints(self):
        problem = self.validation_problem()
        verdict = validate_injected_route(
            problem,
            {"route_nodes": ["elsewhere", 1, DEPOT], "charging_stops": {}},
            g_kwh=300.0,
            charge_kw=300.0,
            reserve_kwh=0.0,
            horizon_min=200.0,
        )
        self.assertEqual(verdict, "route must start and end at the depot")

    def test_validator_does_not_turn_arrival_grace_into_free_energy(self):
        problem = SimpleNamespace(
            adjacency={
                DEPOT: [(1, 0.0, 0.0, "depot_trip")],
                1: [("station", 0.0, 0.0, "trip_station")],
                "station": [(DEPOT, 0.0, 0.0, "station_depot")],
            },
            start_min={1: 0.0},
            end_min={1: 10.0},
            trip_energy={1: 10.0},
        )
        verdict = validate_injected_route(
            problem,
            {
                "route_nodes": [DEPOT, 1, "station", DEPOT],
                "charging_stops": {
                    "stations": ["station"],
                    "cst": [10.0],
                    "cet": [20.0],
                    "kwh": [55.0],
                },
            },
            g_kwh=300.0,
            charge_kw=300.0,
            reserve_kwh=0.0,
            horizon_min=100.0,
        )
        self.assertRegex(verdict, r"55.0 kWh exceeds 300 kW in 10 min")

    def test_validator_rejects_charging_timestamp_outside_horizon(self):
        problem = self.validation_problem()
        verdict = validate_injected_route(
            problem,
            {
                "route_nodes": [DEPOT, 1, DEPOT],
                "charging_stops": {
                    "stations": ["unused"],
                    "cst": [-1.0],
                    "cet": [1.0],
                    "kwh": [0.0],
                },
            },
            g_kwh=300.0,
            charge_kw=300.0,
            reserve_kwh=0.0,
            horizon_min=200.0,
        )
        self.assertRegex(verdict, r"starts before the horizon")

    def test_explicit_partition_loader_validates_and_hashes_every_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data_dir = folder / "data"
            data_dir.mkdir()
            instance = data_dir / "tiny.csv"
            prices_path = data_dir / "prices.csv"
            instance.write_text("instance bytes\n")
            prices_path.write_text("price bytes\n")
            path = folder / "partition.json"
            path.write_text(json.dumps({"routes": [
                {
                    "route": [DEPOT, 1, DEPOT],
                    "charging_stops": {},
                    "deadhead_kwh": -999999.0,
                },
                {"route": [DEPOT, 2, DEPOT], "charging_stops": {}},
            ]}))
            problem = SimpleNamespace(
                adjacency={
                    DEPOT: [
                        (1, 0.0, 2.0, "depot_trip"),
                        (2, 0.0, 4.0, "depot_trip"),
                    ],
                    1: [(DEPOT, 0.0, 3.0, "trip_depot")],
                    2: [(DEPOT, 0.0, 5.0, "trip_depot")],
                },
                start_min={1: 0.0, 2: 10.0},
                end_min={1: 5.0, 2: 15.0},
                trip_energy={1: 0.0, 2: 0.0},
                trips=[1, 2],
            )
            status = {
                "csv": "tiny.csv",
                "prices_csv": "prices.csv",
                "soc_step": 15.0,
                "block_min": 10,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "provenance": {
                    "instance_sha256": hashlib.sha256(
                        instance.read_bytes()
                    ).hexdigest(),
                    "prices_sha256": hashlib.sha256(
                        prices_path.read_bytes()
                    ).hexdigest(),
                },
            }
            pool = [
                {"trips": [1], "cost": 100003.0},
                {"trips": [2], "cost": 100004.0},
            ]
            priced_deadhead = []

            def priced_route(route, *_args, **_kwargs):
                priced_deadhead.append(route["deadhead_kwh"])
                return 100000.0

            with (
                patch(
                    "audit_giro_known_columns.build_problem",
                    return_value=problem,
                ),
                patch(
                    "utils_v2.load_station_hourly_prices",
                    return_value={"PARX": {0: 0.0}},
                ),
                patch(
                    "utils_v2.calculate_truck_route_cost_accurate",
                    side_effect=priced_route,
                ),
            ):
                merged, start, detail = merge_validated_partition_start(
                    pool, [1, 2], path, "prices.csv", status,
                    data_dir=data_dir,
                )

            self.assertEqual(len(merged), 2)
            self.assertEqual(len(start), 2)
            self.assertEqual(
                sorted(merged[index]["trips"] for index in start),
                [[1], [2]],
            )
            self.assertEqual(detail["kind"], "validated_exact_partition")
            self.assertEqual(detail["validated_bus_count"], 2)
            self.assertEqual(detail["pool_columns_replaced"], 2)
            self.assertEqual(priced_deadhead, [5.0, 9.0])
            self.assertEqual(
                detail["source_sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_explicit_partition_loader_fails_closed_on_coverage_or_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data_dir = folder / "data"
            data_dir.mkdir()
            instance = data_dir / "tiny.csv"
            prices_path = data_dir / "prices.csv"
            instance.write_text("instance bytes\n")
            prices_path.write_text("price bytes\n")
            problem = SimpleNamespace(
                adjacency={
                    DEPOT: [
                        (1, 0.0, 0.0, "depot_trip"),
                        (2, 0.0, 0.0, "depot_trip"),
                    ],
                    1: [(DEPOT, 0.0, 0.0, "trip_depot")],
                    2: [(DEPOT, 0.0, 0.0, "trip_depot")],
                },
                start_min={1: 0.0, 2: 10.0},
                end_min={1: 5.0, 2: 15.0},
                trip_energy={1: 0.0, 2: 0.0},
                trips=[1, 2],
            )
            status = {
                "csv": "tiny.csv",
                "prices_csv": "prices.csv",
                "soc_step": 15.0,
                "block_min": 10,
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "min_soc_frac": 0.0,
                "provenance": {
                    "instance_sha256": hashlib.sha256(
                        instance.read_bytes()
                    ).hexdigest(),
                    "prices_sha256": hashlib.sha256(
                        prices_path.read_bytes()
                    ).hexdigest(),
                },
            }
            missing = folder / "missing.json"
            missing.write_text(json.dumps({"routes": [
                {"route": [DEPOT, 1, DEPOT], "charging_stops": {}},
            ]}))
            invalid = folder / "invalid.json"
            invalid.write_text(json.dumps({"routes": [
                {"route": ["elsewhere", 1, DEPOT], "charging_stops": {}},
                {"route": [DEPOT, 2, DEPOT], "charging_stops": {}},
            ]}))
            repeated = folder / "repeated.json"
            repeated.write_text(json.dumps({"routes": [
                {"route": [DEPOT, 1, DEPOT], "charging_stops": {}},
                {"route": [DEPOT, 1, DEPOT], "charging_stops": {}},
                {"route": [DEPOT, 2, DEPOT], "charging_stops": {}},
            ]}))
            common_patches = (
                patch(
                    "audit_giro_known_columns.build_problem",
                    return_value=problem,
                ),
                patch(
                    "utils_v2.load_station_hourly_prices",
                    return_value={"PARX": {0: 0.0}},
                ),
                patch(
                    "utils_v2.calculate_truck_route_cost_accurate",
                    return_value=100000.0,
                ),
            )
            with common_patches[0], common_patches[1], common_patches[2]:
                with self.assertRaisesRegex(
                    SystemExit, "not an exact partition"
                ):
                    merge_validated_partition_start(
                        [], [1, 2], missing, "prices.csv", status,
                        data_dir=data_dir,
                    )
                with self.assertRaisesRegex(
                    SystemExit, "failed physical validation"
                ):
                    merge_validated_partition_start(
                        [], [1, 2], invalid, "prices.csv", status,
                        data_dir=data_dir,
                    )
                with self.assertRaisesRegex(
                    SystemExit, "not an exact partition"
                ):
                    merge_validated_partition_start(
                        [], [1, 2], repeated, "prices.csv", status,
                        data_dir=data_dir,
                    )
                bad_physics = dict(status, g_kwh=float("nan"))
                with self.assertRaisesRegex(
                    SystemExit, "invalid or non-finite physics"
                ):
                    merge_validated_partition_start(
                        [], [1, 2], missing, "prices.csv", bad_physics,
                        data_dir=data_dir,
                    )
                bad_hash = {
                    **status,
                    "provenance": {
                        **status["provenance"],
                        "instance_sha256": "f" * 64,
                    },
                }
                with self.assertRaisesRegex(
                    SystemExit, "instance hash mismatch"
                ):
                    merge_validated_partition_start(
                        [], [1, 2], missing, "prices.csv", bad_hash,
                        data_dir=data_dir,
                    )
                bad_problem = SimpleNamespace(
                    **{
                        **problem.__dict__,
                        "adjacency": {
                            **problem.adjacency,
                            DEPOT: [
                                (1, 0.0, float("nan"), "depot_trip"),
                                (2, 0.0, 0.0, "depot_trip"),
                            ],
                        },
                    }
                )
                with (
                    patch(
                        "audit_giro_known_columns.build_problem",
                        return_value=bad_problem,
                    ),
                    self.assertRaisesRegex(
                        SystemExit, "invalid arc data"
                    ),
                ):
                    merge_validated_partition_start(
                        [], [1, 2], missing, "prices.csv", status,
                        data_dir=data_dir,
                    )

    def run_fake_gurobi_mip(
        self, stages, *, explicit_start=False, mip_gap=0.0001,
    ):
        class FakeExpression:
            def __init__(self, items):
                self.items = list(items)

            def __eq__(self, other):
                return ("eq", self, other)

            def __ge__(self, other):
                return ("ge", self, other)

        class FakeVariable:
            def __init__(self, index):
                self.index = index
                self.Start = 0.0
                self.X = 0.0

            def __rmul__(self, coefficient):
                return ("term", float(coefficient), self.index)

        class FakeModel:
            def __init__(self, _name):
                self.Params = SimpleNamespace()
                self.variables = {}
                self.optimize_calls = 0
                self.objectives = []
                self.SolCount = 0
                self.ObjVal = 0.0
                self.ObjBound = 0.0
                self.MIPGap = 0.0
                self.Status = 1

            def addVars(self, count, **_kwargs):
                self.variables = {i: FakeVariable(i) for i in range(count)}
                return self.variables

            def addConstr(self, constraint, **_kwargs):
                return constraint

            def setObjective(self, expression, _sense):
                self.objectives.append(expression)

            def cbGet(self, _what):
                return self.callback_message

            def optimize(self, callback=None):
                stage = stages[self.optimize_calls]
                self.optimize_calls += 1
                messages = stage.get("start_messages")
                if messages is None:
                    messages = [stage.get("start_message", "")]
                for message in messages:
                    self.callback_message = message
                    if callback is not None and self.callback_message:
                        callback(self, 6)
                self.Status = stage["status"]
                self.SolCount = stage.get("solutions", 1)
                self.ObjVal = stage["objective"]
                self.ObjBound = stage["bound"]
                self.MIPGap = stage.get("gap", 0.0)
                selected = set(stage.get("selected", []))
                for index, variable in self.variables.items():
                    variable.X = 1.0 if index in selected else 0.0

        models = []
        fake_gp = ModuleType("gurobipy")

        def make_model(name):
            model = FakeModel(name)
            models.append(model)
            return model

        fake_gp.Model = make_model
        fake_gp.quicksum = lambda values: FakeExpression(list(values))
        fake_gp.GRB = SimpleNamespace(
            BINARY=1,
            MINIMIZE=1,
            Callback=SimpleNamespace(MESSAGE=6, MSG_STRING=6001),
        )
        fake_gp.gurobi = SimpleNamespace(version=lambda: (12, 0, 0))

        temporary = tempfile.TemporaryDirectory()
        folder = Path(temporary.name)
        result = folder / "pool.snapshot.json"
        journal = Path(str(result) + ".columns.jsonl")
        routes = [
            {"trips": [1], "cost": 100003.0},
            {"trips": [2], "cost": 100004.0},
        ]
        journal.write_text("".join(json.dumps(route) + "\n" for route in routes))
        result.write_text(json.dumps({
            "csv": "tiny.csv",
            "prices_csv": "hourly_prices_flat.csv",
            "soc_step": 5.0,
            "block_min": 10,
            "g_kwh": 300.0,
            "charge_kw": 300.0,
            "min_soc_frac": 0.0,
            "wall_s": 21637.5,
            "iterations": 1200,
            "snapshot_mark_minutes": 360.0,
            "trip_ids": [1, 2],
            "columns_journal": str(journal),
        }))
        out = folder / "mip.json"
        arguments = [
            "--result", str(result), "--two-stage",
            "--timelimit", "60", "--mipgap", str(mip_gap),
            "--out", str(out),
        ]
        explicit_patch = contextlib.nullcontext()
        if explicit_start:
            partition = folder / "partition.json"
            partition.write_text(json.dumps({"routes": []}))
            merged_routes = [
                *routes,
                {"trips": [1, 2], "cost": 100005.0},
            ]
            detail = {
                "kind": "validated_exact_partition",
                "source": str(partition),
                "source_sha256": "partition-sha",
                "validated": True,
                "validated_bus_count": 1,
                "expected_full_objective": 100005.0,
            }
            explicit_patch = patch(
                "run_exact_pool_mip.merge_validated_partition_start",
                return_value=(merged_routes, [2], detail),
            )
            arguments.extend([
                "--initial-partition-routes", str(partition),
            ])
        with patch.dict(sys.modules, {"gurobipy": fake_gp}), explicit_patch:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = main(arguments)
        payload = json.loads(out.read_text())
        return temporary, models[0], payload, rc

    def test_explicit_partition_is_assigned_and_solver_acceptance_recorded(self):
        temporary, model, payload, rc = self.run_fake_gurobi_mip([{
            "status": 9,
            "objective": 1.0,
            "bound": 0.0,
            "gap": 1.0,
            "selected": [2],
            "start_messages": [
                "Loaded user MIP start with objective 1",
                "User MIP start did not produce a new incumbent solution",
            ],
        }], explicit_start=True, mip_gap=0.0125)
        self.addCleanup(temporary.cleanup)

        self.assertEqual(rc, 0)
        self.assertEqual(payload["experiment_arm"], "D")
        self.assertEqual(payload["requested_mip_gap"], 0.0125)
        self.assertEqual(model.variables[2].Start, 1.0)
        self.assertEqual(model.variables[0].Start, 0.0)
        self.assertEqual(payload["mip_start"]["kind"],
                         "validated_exact_partition")
        self.assertEqual(payload["mip_start"]["validated_bus_count"], 1)
        self.assertEqual(payload["mip_start"]["assigned_variable_count"], 3)
        self.assertEqual(payload["mip_start"]["selected_variable_count"], 1)
        self.assertTrue(
            payload["mip_start"]["solver_acceptance"]["accepted"]
        )
        self.assertTrue(
            payload["mip_start"]["solver_acceptance"][
                "rejection_observed"
            ]
        )
        self.assertEqual(
            payload["mip_start"]["solver_acceptance"]["status"], "accepted"
        )
        self.assertTrue(payload["mip_start_used"])
        self.assertTrue(payload["mip_start_assigned"])

    def test_unproven_fleet_uses_full_primary_stage_and_skips_cost_stage(self):
        temporary, model, payload, rc = self.run_fake_gurobi_mip([{
            "status": 9,
            "objective": 2.0,
            "bound": 0.9,
            "gap": 0.55,
            "selected": [0, 1],
        }])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(rc, 0)
        self.assertEqual(model.optimize_calls, 1)
        self.assertFalse(payload["fleet_proven"])
        self.assertFalse(payload["two_stage"]["stage2_executed"])
        self.assertEqual(payload["status_name"], "TIME_LIMIT")
        self.assertEqual(payload["optimal_scope"], "none")

    def test_proven_fleet_runs_cost_stage_and_reconstructs_full_objective(self):
        temporary, model, payload, rc = self.run_fake_gurobi_mip([
            {
                "status": 9,
                "objective": 2.0,
                "bound": 1.01,
                "gap": 0.5,
                "selected": [0, 1],
            },
            {
                "status": 2,
                "objective": 7.0,
                "bound": 7.0,
                "gap": 0.0,
                "selected": [0, 1],
            },
        ])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(rc, 0)
        self.assertEqual(model.optimize_calls, 2)
        self.assertTrue(payload["fleet_proven"])
        self.assertEqual(payload["optimal_scope"], "full_pool_lexicographic")
        self.assertEqual(payload["mip_obj"], 200007.0)
        self.assertEqual(payload["mip_bound"], 200007.0)
        self.assertEqual(payload["absolute_cost_gap"], 0.0)
        self.assertEqual(payload["source_cg_wall_s"], 21637.5)
        self.assertEqual(payload["source_cg_iterations"], 1200)
        self.assertEqual(payload["source_snapshot_mark_minutes"], 360.0)
        variable_terms = model.objectives[-1].items
        self.assertEqual([term[1] for term in variable_terms], [3.0, 4.0])

    def test_cost_stage_without_solution_preserves_proven_fleet_incumbent(self):
        temporary, model, payload, rc = self.run_fake_gurobi_mip([
            {
                "status": 2,
                "objective": 2.0,
                "bound": 2.0,
                "selected": [0, 1],
            },
            {
                "status": 11,
                "objective": 0.0,
                "bound": 0.0,
                "solutions": 0,
                "selected": [],
            },
        ])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(rc, 0)
        self.assertEqual(model.optimize_calls, 2)
        self.assertEqual(payload["buses"], 2)
        self.assertTrue(payload["fleet_proven"])
        self.assertEqual(payload["optimal_scope"], "fleet_only")
        self.assertEqual(payload["mip_obj"], 200007.0)
        self.assertEqual(
            payload["two_stage"]["stage2_reported_incumbent_source"],
            "stage1_fallback",
        )

    def test_two_stage_no_incumbent_still_writes_final_result(self):
        temporary, model, payload, rc = self.run_fake_gurobi_mip([{
            "status": 9,
            "objective": 0.0,
            "bound": 1.0,
            "gap": 1.0,
            "solutions": 0,
            "selected": [],
        }])
        self.addCleanup(temporary.cleanup)
        self.assertEqual(rc, 0)
        self.assertEqual(model.optimize_calls, 1)
        self.assertEqual(payload["status_name"], "TIME_LIMIT")
        self.assertFalse(payload["incumbent_found"])
        self.assertIsNone(payload["buses"])
        self.assertIsNone(payload["mip_obj"])
        self.assertEqual(
            payload["two_stage"]["stage2_skip_reason"],
            "no_fleet_incumbent",
        )


if __name__ == "__main__":
    unittest.main()
