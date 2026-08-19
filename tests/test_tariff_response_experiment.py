import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_giro_known_columns import DEPOT, STATIONS  # noqa: E402
from build_tariff_response_evidence import (  # noqa: E402
    SCHEMA,
    _validate_cell,
    build as build_evidence,
)
from build_tariff_response_manifest import sha256_file  # noqa: E402
from expanded_path_realization import (  # noqa: E402
    charging_block_schedule_sha256,
)
from exact_pricer_expanded import (  # noqa: E402
    validated_fixed_duty_seed_records,
)
from fixed_duty_expanded_optimizer import (  # noqa: E402
    optimize_fixed_duty,
)
from launch_tariff_response_pilot import (  # noqa: E402
    K40_SUBMISSION_SCOPE,
    MAIN_SUBMISSION_SCOPE,
    build_plan,
    submit,
    tariff_child_spec,
    tariff_gate_spec,
)
import launch_tariff_response_pilot as pilot  # noqa: E402
from assemble_tariff_response_campaign import assemble  # noqa: E402
from assemble_tariff_response_campaign import _aggregate  # noqa: E402
from build_tariff_response_evidence import _schedule_fingerprint  # noqa: E402
from reconcile_tariff_response_gate import reconcile  # noqa: E402
from validate_tariff_response_archive import (  # noqa: E402
    sha as archive_sha,
    validate_reservations,
)
from tariff_response_core import (  # noqa: E402
    PHYSICS,
    evaluate_giro_original,
    load_tariff_manifest,
    reconstruct_giro40_original,
    route_response,
    tariff_prices,
)
from tariff_response_completion import (  # noqa: E402
    SCHEMA as COMPLETION_SCHEMA,
    validate_completion_identity,
)
from preflight_tariff_response_fixed_duties import (  # noqa: E402
    build_preflight as build_fixed_duty_preflight,
)
from slurm_state_contract import (  # noqa: E402
    SlurmContractError,
    release_with_postcondition,
    resolve_exact_job,
    verified_gate_evidence,
    verify_held_receipt,
)


TARIFF_MANIFEST = REPO_ROOT / "data/tariff_response/tariff_manifest.csv"
GIRO_DUTY_MANIFEST = (
    REPO_ROOT / "data/tariff_response/giro40_duty_manifest.csv"
)
FROZEN_INPUT_MANIFEST = (
    REPO_ROOT
    / "data/tariff_response/frozen_instances/frozen_input_manifest.csv"
)


def scheduler_result(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr
    )


class SyntheticScheduler:
    """Explicit scheduler transcript consumed by production parsers."""

    def __init__(
        self, *, live=None, controller=None, accounting=None, release=None,
    ):
        self.live = list(live or [])
        self.controller = list(controller or [])
        self.accounting = list(accounting or [])
        self.release = list(release or [])
        self.commands = []

    @staticmethod
    def _next(queue, source):
        if not queue:
            return scheduler_result(1, stderr=f"no scripted {source} read")
        value = queue.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    def __call__(self, command, **_kwargs):
        self.commands.append(list(command))
        executable = Path(str(command[0])).name
        if executable == "squeue":
            return self._next(self.live, "squeue")
        if executable == "sacct":
            return self._next(self.accounting, "sacct")
        if executable == "scontrol" and command[1] == "release":
            return self._next(self.release, "release")
        if executable == "scontrol":
            return self._next(self.controller, "scontrol")
        raise AssertionError(f"unexpected scheduler command: {command}")


def gate_fixture_spec(job_id="12345"):
    return {
        "job_id": job_id,
        "user": "nathan",
        "job_name": "TRGMabcdef",
        "partition": "default_partition",
        "comment": "TRSPG:abcdef0123456789abcd:m",
        "role": (
            "tariff_response_release_gate:"
            f"{MAIN_SUBMISSION_SCOPE}"
        ),
        "submission_scope": MAIN_SUBMISSION_SCOPE,
    }


def live_gate_row(
    spec, *, state="PENDING", reason="JobHeldUser", comment=None,
):
    return scheduler_result(stdout=(
        f"{spec['job_id']}|{spec['user']}|{spec['job_name']}|{state}|"
        f"{spec['partition']}|{reason}|"
        f"{spec['comment'] if comment is None else comment}\n"
    ))


def accounting_gate_row(
    spec, *, state="COMPLETED", exit_code="0:0",
):
    return scheduler_result(stdout=(
        f"{spec['job_id']}|{spec['user']}|{spec['job_name']}|{state}|"
        f"{spec['partition']}|{spec['comment']}|{exit_code}\n"
    ))


def submitted_child_row(plan, job, gate_id, job_id):
    dependency = f"afterok:{gate_id}"
    spec = tariff_child_spec(
        plan, job, dependency, job_id
    )
    observation = {
        **{
            field: spec[field] for field in (
                "job_id", "user", "job_name", "partition",
                "comment", "dependency",
            )
        },
        "state": "PENDING",
        "reason": "Dependency",
        "exit_code": "0:0",
        "source": "scontrol",
        "live": True,
    }
    return {
        "job_key": job["job_key"],
        **{
            field: spec[field] for field in (
                "job_id", "user", "job_name", "partition",
                "comment", "dependency", "role", "submission_scope",
            )
        },
        "submission_receipt": {
            "verified": True,
            "role": spec["role"],
            "submission_scope": spec["submission_scope"],
            "job_id": str(job_id),
            "attempts": 1,
            "observation": observation,
            "diagnostics": [],
        },
    }


def live_child_row(spec, *, include_dependency):
    fields = [
        spec["job_id"], spec["user"], spec["job_name"], "PENDING",
        spec["partition"], "Dependency", spec["comment"],
    ]
    if include_dependency:
        fields.append(spec["dependency"])
    return scheduler_result(stdout="|".join(fields) + "\n")


def toy_problem(*, boundary=False):
    station = STATIONS[0]
    if boundary:
        energy = {0: 280.0, 1: 80.0}
        starts = {0: 0.0, 1: 70.0}
        ends = {0: 50.0, 1: 80.0}
    else:
        energy = {0: 250.0, 1: 40.0}
        starts = {0: 0.0, 1: 180.0}
        ends = {0: 10.0, 1: 190.0}
    adjacency = {
        DEPOT: [(0, 0.0, 0.0, "depot_trip")],
        0: [(station, 0.0, 0.0, "trip_station")],
        station: [(1, 0.0, 0.0, "station_trip")],
        1: [(DEPOT, 0.0, 0.0, "trip_depot")],
    }
    return SimpleNamespace(
        trips=[0, 1],
        trip_energy=energy,
        start_min=starts,
        end_min=ends,
        adjacency=adjacency,
    )


def price_curves(hour0=10.0, other=1.0):
    return {
        station.rsplit("_", 1)[0]: {
            hour: hour0 if hour == 0 else other
            for hour in range(27)
        }
        for station in STATIONS
    }


class TariffResponseExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tariffs = load_tariff_manifest(TARIFF_MANIFEST)
        cls.tariff_by_id = {
            row["tariff_id"]: row for row in cls.tariffs
        }

    def test_manifest_hashes_alpha_formula_and_spatial_solar(self):
        self.assertEqual(len(self.tariffs), 11)
        flat = tariff_prices(self.tariff_by_id["flat"])["PARX"]
        peak = tariff_prices(self.tariff_by_id["peak12"])["PARX"]
        for alpha, tariff_id in (
            (0.0, "peak12_alpha_0p0"),
            (0.25, "peak12_alpha_0p25"),
            (0.5, "peak12_alpha_0p5"),
            (1.0, "peak12_alpha_1p0"),
            (2.0, "peak12_alpha_2p0"),
        ):
            observed = tariff_prices(
                self.tariff_by_id[tariff_id]
            )["PARX"]
            for hour in range(27):
                self.assertAlmostEqual(
                    observed[hour],
                    flat[hour] + alpha * (peak[hour] - flat[hour]),
                    places=14,
                )
        solar = tariff_prices(
            self.tariff_by_id["solar_parx_midday_free"]
        )
        self.assertEqual(solar["PARX"][12], 0.0)
        self.assertEqual(solar["JON_A"][12], flat[12])
        alpha2 = self.tariff_by_id["peak12_alpha_2p0"]
        self.assertEqual(alpha2["has_negative_prices"], "True")
        self.assertEqual(
            alpha2["negative_price_policy"],
            "allow_feasible_consumption_no_export",
        )
        self.assertEqual(
            alpha2["analysis_role"], "negative_price_stress"
        )
        self.assertEqual(
            alpha2["primary_response_eligible"], "False"
        )
        self.assertTrue(all(
            int(row["coverage_end_hour"]) == 26 for row in self.tariffs
        ))
        with GIRO_DUTY_MANIFEST.open(newline="") as handle:
            duty_rows = list(csv.DictReader(handle))
        self.assertEqual(len(duty_rows), 40)
        self.assertEqual(
            {row["duty_id"] for row in duty_rows}
            & {"13316m", "13324muw"},
            set(),
        )
        with FROZEN_INPUT_MANIFEST.open(newline="") as handle:
            frozen_rows = list(csv.DictReader(handle))
        self.assertEqual(
            sha256_file(FROZEN_INPUT_MANIFEST),
            "5473e8d83c8e7e1f0b6e872125419466bb5044bbbb014df3184254f6a2b601c6",
        )
        self.assertEqual(
            {int(row["scale"]) for row in frozen_rows}, {5, 8, 40}
        )
        for row in frozen_rows:
            path = REPO_ROOT / row["relative_path"]
            self.assertEqual(sha256_file(path), row["file_sha256"])
            with path.open(newline="") as handle:
                trips = [
                    int(float(item["Ordered_Trip_ID"]))
                    for item in csv.DictReader(handle)
                ]
            digest = hashlib.sha256(json.dumps(
                sorted(trips), sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            self.assertEqual(digest, row["trip_set_sha256"])

    def test_giro_original_is_exact_and_ambiguous_costs_stay_null(self):
        original = reconstruct_giro40_original(
            REPO_ROOT / "data/Par_VehicleDetails_Updated.csv"
        )
        self.assertEqual(len(original["duties"]), 40)
        self.assertEqual(len(original["events"]), 344)
        self.assertAlmostEqual(
            sum(event["kwh"] for event in original["events"]),
            15989.5501436,
            places=6,
        )
        self.assertEqual(
            {row["duty_id"] for row in original["duties"]}
            & {"13316m", "13324muw"},
            set(),
        )
        peak, rows = evaluate_giro_original(
            original, self.tariff_by_id["peak12"]
        )
        self.assertIsNone(peak["grid_model_objective"])
        self.assertEqual(peak["scalar_cost_availability"], "unavailable")
        self.assertTrue(any(
            row["availability_reason"]
            == "within_window_energy_allocation_unavailable"
            for row in rows
        ))
        self.assertFalse(peak["continuous_cost_pricing_certified"])
        flat, _ = evaluate_giro_original(
            original, self.tariff_by_id["flat"]
        )
        self.assertEqual(flat["scalar_cost_availability"], "available")
        subset, _ = evaluate_giro_original(
            {
                "events": [],
                "routes": [{"trips": [index]} for index in range(5)],
                "recorded_terminal_soc_policy": "unavailable",
            },
            self.tariff_by_id["flat"],
        )
        self.assertEqual(subset["buses"], 5)
        self.assertEqual(subset["grid_model_objective"], 500000.0)

    def test_fixed_duty_dp_selects_delayed_charging_and_certifies_scope(self):
        result = optimize_fixed_duty(
            toy_problem(),
            [0, 1],
            price_curves(),
            tariff_id="toy",
            tariff_sha256="a" * 64,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(
            result["route"]["charging_stops"]["cst"], [60]
        )
        self.assertEqual(result["expanded_grid_objective"], 100050.0)
        self.assertEqual(
            result["certificate"]["scope"],
            "optimal_discretized_charging_for_fixed_trip_sequence",
        )
        self.assertFalse(
            result["certificate"][
                "continuous_cost_optimality_certified"
            ]
        )

    def test_tariff_boundary_blocks_use_their_own_hour_price(self):
        result = optimize_fixed_duty(
            toy_problem(boundary=True),
            [0, 1],
            price_curves(),
            tariff_id="boundary",
            tariff_sha256="b" * 64,
        )
        blocks = result["route"][
            "continuous_realized_charging_blocks"
        ]
        self.assertEqual(
            [(block["tariff_hour"], block["price_per_kwh"])
             for block in blocks],
            [(0, 10.0), (1, 1.0)],
        )
        self.assertEqual(result["expanded_grid_objective"], 100500.0)

    def test_negative_alpha_terminal_consumption_is_explicitly_reported(self):
        station = STATIONS[0]
        problem = SimpleNamespace(
            trips=[0],
            trip_energy={0: 100.0},
            start_min={0: 0.0},
            end_min={0: 10.0},
            adjacency={
                DEPOT: [(0, 0.0, 0.0, "depot_trip")],
                0: [
                    (DEPOT, 0.0, 0.0, "trip_depot"),
                    (station, 0.0, 0.0, "trip_station"),
                ],
                station: [(DEPOT, 0.0, 0.0, "station_depot")],
            },
        )
        prices = {
            value.rsplit("_", 1)[0]: {
                hour: -1.0 for hour in range(27)
            }
            for value in STATIONS
        }
        result = optimize_fixed_duty(
            problem, [0], prices,
            tariff_id="negative", tariff_sha256="e" * 64,
        )
        self.assertGreater(
            sum(result["route"]["charging_stops"]["kwh"]), 0.0
        )
        self.assertGreater(
            result["route"]["continuous_terminal_soc_kwh"], 200.0
        )

    def test_route_response_ignores_bus_labels(self):
        baseline = [
            {"bus": "A", "trips": [0, 1]},
            {"bus": "B", "trips": [2, 3]},
        ]
        relabeled = [
            {"bus": "totally-different", "trips": [2, 3]},
            {"bus": "x", "trips": [0, 1]},
        ]
        response = route_response(baseline, relabeled)
        self.assertEqual(
            response["percent_trips_assigned_to_different_duty"], 0.0
        )
        self.assertEqual(response["trip_adjacency_jaccard"], 1.0)
        changed = route_response(
            baseline,
            [{"trips": [0, 2]}, {"trips": [1, 3]}],
        )
        self.assertGreater(
            changed["percent_trips_assigned_to_different_duty"], 0.0
        )
        with_terminal = [{
            "trips": [0],
            "continuous_terminal_soc_kwh": 50.0,
        }]
        changed_terminal = copy.deepcopy(with_terminal)
        changed_terminal[0]["continuous_terminal_soc_kwh"] = 999.0
        self.assertNotEqual(
            _schedule_fingerprint(with_terminal),
            _schedule_fingerprint(changed_terminal),
        )

    def test_assembler_reads_terminal_soc_from_physical_realization(self):
        result = optimize_fixed_duty(
            toy_problem(), [0, 1], price_curves(),
            tariff_id="toy", tariff_sha256="a" * 64,
        )
        route = copy.deepcopy(result["route"])
        route.pop("continuous_terminal_soc_kwh")
        tariff = copy.deepcopy(self.tariff_by_id["flat"])
        routes, metrics = _aggregate(
            [route], tariff, toy_problem()
        )
        self.assertEqual(
            routes[0]["continuous_terminal_soc_kwh"],
            route["physical_realization"][
                "continuous_terminal_soc_kwh"
            ],
        )
        self.assertLessEqual(
            metrics["terminal_soc_min_kwh"],
            metrics["terminal_soc_max_kwh"],
        )

    def _seed_payload(self, root):
        tariff_path = root / "tariff.csv"
        tariff_path.write_text(
            "time_block,cost\n"
            + "\n".join(f"{hour},{1 if hour else 10}"
                        for hour in range(27))
            + "\n"
        )
        tariff_sha = sha256_file(tariff_path)
        result = optimize_fixed_duty(
            toy_problem(), [0, 1], price_curves(),
            tariff_id="toy", tariff_sha256=tariff_sha,
        )
        route = result["route"]
        route["duty_id"] = "toy-duty"
        payload = {
            "schema": "evsp-dr-tier1-fixed-duty-partition-v1",
            "routes": [route],
            "tariff": {"tariff_id": "toy", "sha256": tariff_sha},
            "instance_sha256": None,
            "physics": PHYSICS,
            "certificates": [{
                "duty_id": "toy-duty",
                **result["certificate"],
            }],
            "continuous_cost_pricing_certified": False,
        }
        path = root / "seed.json"
        path.write_text(json.dumps(payload))
        station_prices = price_curves()
        return path, tariff_path, tariff_sha, station_prices

    def test_seed_loader_rejects_reused_cost_and_false_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed, tariff, _sha, prices = self._seed_payload(root)
            records, _ = validated_fixed_duty_seed_records(
                seed, toy_problem(), prices, tariff_path=tariff,
                g_kwh=300, charge_kw=300, reserve_kwh=0,
                soc_step=15, block_min=10,
            )
            self.assertEqual(len(records), 1)
            original = json.loads(seed.read_text())
            payload = copy.deepcopy(original)
            payload["routes"][0]["expanded_grid_cost"] -= 123.0
            payload["routes"][0]["cost"] -= 123.0
            seed.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "provenance"):
                validated_fixed_duty_seed_records(
                    seed, toy_problem(), prices, tariff_path=tariff,
                    g_kwh=300, charge_kw=300, reserve_kwh=0,
                    soc_step=15, block_min=10,
                )
            payload = copy.deepcopy(original)
            payload["routes"][0]["cost_tariff_sha256"] = "0" * 64
            seed.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "seed route"):
                validated_fixed_duty_seed_records(
                    seed, toy_problem(), prices, tariff_path=tariff,
                    g_kwh=300, charge_kw=300, reserve_kwh=0,
                    soc_step=15, block_min=10,
                )
            payload["routes"][0]["cost_tariff_sha256"] = _sha
            payload["continuous_cost_pricing_certified"] = True
            seed.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "identity"):
                validated_fixed_duty_seed_records(
                    seed, toy_problem(), prices, tariff_path=tariff,
                    g_kwh=300, charge_kw=300, reserve_kwh=0,
                    soc_step=15, block_min=10,
                )

    def _synthetic_cell(
        self, tariff, tier, treatment, *, route_flexible=False
    ):
        price = tariff_prices(tariff)["PARX"][12]
        factor = (
            1.0 if tier == "TIER0_GIRO_ORIGINAL"
            else 0.8 if tier == "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING"
            else 0.6
        )
        energy = 10.0 * factor
        block = {
            "stop_index": 0,
            "block_index": 0,
            "station": "PARX_1",
            "start_min": 720.0,
            "end_min": 730.0,
            "realized_kwh": energy,
            "expanded_grid_kwh": energy,
            "tariff_hour": 12,
            "tariff_key": "PARX:12",
            "price_per_kwh": price,
        }
        if route_flexible:
            trip_groups = [[0, 1]]
        else:
            trip_groups = [[0], [1]]
        routes = []
        for index, trips in enumerate(trip_groups):
            blocks = [] if index else [block]
            routes.append({
                "route_id": f"{tier}-{index}",
                "trips": trips,
                "trip_blocks": [{
                    "trip_id": trip,
                    "start_min": 100 + 60 * trip,
                    "end_min": 125 + 60 * trip,
                } for trip in trips],
                "charging_stops": {
                    "stations": [] if not blocks else ["PARX_1"],
                    "cst": [] if not blocks else [720],
                    "cet": [] if not blocks else [730],
                    "kwh": [] if not blocks else [energy],
                },
                "continuous_realized_charging_blocks": blocks,
                "recorded_charging_blocks": blocks,
                "continuous_realized_charging_blocks_sha256":
                    charging_block_schedule_sha256(blocks),
                "cost_tariff_sha256": (
                    None if tier == "TIER0_GIRO_ORIGINAL"
                    else tariff["sha256"]
                ),
                "expanded_grid_cost": (
                    100000 + (5 + energy * price if blocks else 0)
                ),
                "continuous_realized_cost": (
                    100000 + (5 + energy * price if blocks else 0)
                ),
                "waiting_min": 20.0 * factor if index == 0 else 0.0,
                "deadhead_min": 5.0 * factor if index == 0 else 0.0,
                "deadhead_kwh": 2.0 * factor if index == 0 else 0.0,
                "continuous_terminal_soc_kwh": 50.0,
            })
        buses = len(routes)
        charging = 5.0 + energy * price
        certificates = []
        if tier == "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING":
            for route in routes:
                payload = {
                    "certified": True,
                    "scope": "synthetic_discretized_fixed_duty",
                    "continuous_cost_optimality_certified": False,
                }
                digest = hashlib.sha256(json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode()).hexdigest()
                route["fixed_duty_certificate_sha256"] = digest
                certificates.append({
                    "route_id": route["route_id"],
                    **payload,
                    "certificate_sha256": digest,
                })
        metrics = {
            "buses": buses,
            "grid_model_objective": buses * 100000 + charging,
            "continuous_replay_objective":
                buses * 100000 + charging,
            "charging_cost": charging,
            "continuous_charging_cost": charging,
            "total_charged_kwh": energy,
            "peak_window_kwh": energy,
            "charging_kwh_by_hour_json": json.dumps({"12": energy}),
            "charging_kwh_by_station_json": json.dumps(
                {"PARX_1": energy}
            ),
            "charging_starts_by_hour_json": json.dumps({"12": 1}),
            "terminal_soc_min_kwh": 50.0,
            "terminal_soc_max_kwh": 50.0,
            "terminal_surplus_total_kwh": 50.0 * buses,
            "waiting_min": 20.0 * factor,
            "deadhead_min": 5.0 * factor,
            "deadhead_kwh": 2.0 * factor,
            "charging_stops": 1,
            "discretized_certification_status":
                "not_applicable" if tier == "TIER0_GIRO_ORIGINAL"
                else "certified",
            "runtime_preprocessing_s": 0.1,
            "runtime_master_s": 0.2 if tier.startswith("TIER2") else 0.0,
            "runtime_pricing_s": 0.3,
            "runtime_postprocessing_s": 0.1,
        }
        cell = {
            "cell_id": f"{tier}-{tariff['tariff_id']}",
            "instance_id": "synthetic-k2",
            "scale": 2,
            "tariff_id": tariff["tariff_id"],
            "tariff_sha256": tariff["sha256"],
            "tier": tier,
            "treatment": treatment,
            "trip_ids": [0, 1],
            "routes": routes,
            "metrics": metrics,
            "physical_replay_status": (
                "unavailable_recorded_power_profile_ambiguous"
                if tier == "TIER0_GIRO_ORIGINAL"
                else "validated_all_routes"
            ),
            "terminal_soc_policy": PHYSICS["terminal_soc_policy"],
            "continuous_cost_pricing_certified": False,
            "certificate_scope": (
                "none_recorded_schedule"
                if tier == "TIER0_GIRO_ORIGINAL"
                else "discretized_fixed_duty"
                if tier.startswith("TIER1")
                else "finite_augmented_pool"
            ),
            "fixed_duty_certificates": certificates,
            "cg_iterations": ([{
                "iteration": 1, "elapsed_s": 1.0,
                "lp_obj": 1.0, "route_weight": buses,
                "artificials": 0.0, "min_rc": -0.1,
                "pool_columns": 3,
            }] if tier.startswith("TIER2") else []),
            "mip_checkpoints": ([{
                "checkpoint_elapsed_s": 60,
                "incumbent_fleet": buses, "fleet_bound": 1,
                "fleet_gap": 0.0, "node_count": 1,
                "solution_count": 1, "route_vector_sha256": "d" * 64,
                "solver_ended_before_checkpoint": False,
            }] if tier.startswith("TIER2") else []),
            "source_artifacts": [],
        }
        return cell

    def test_evidence_builder_rejects_semantic_corruption_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duty_manifest = root / "duties.csv"
            duty_manifest.write_text("duty_id\nD0\nD1\n")
            cells = []
            for tariff_id in (
                "peak12_alpha_0p0", "peak12_alpha_0p25",
                "peak12_alpha_0p5", "peak12_alpha_1p0",
                "peak12_alpha_2p0",
            ):
                tariff = self.tariff_by_id[tariff_id]
                tier0 = self._synthetic_cell(
                    tariff, "TIER0_GIRO_ORIGINAL", "GIRO_ORIGINAL"
                )
                if tariff_id == "peak12_alpha_2p0":
                    tier0["metrics"]["charging_cost"] = None
                    tier0["metrics"]["continuous_charging_cost"] = None
                    tier0["metrics"]["terminal_surplus_total_kwh"] = None
                cells.extend([
                    tier0,
                    self._synthetic_cell(
                        tariff,
                        "TIER1_FIXED_GIRO_OPTIMIZED_CHARGING",
                        "FIXED_GIRO",
                    ),
                    self._synthetic_cell(
                        tariff,
                        "TIER2_GIRO40_AUGMENTED_ROUTE_CHARGING",
                        "GIRO40-AUGMENTED",
                        route_flexible=True,
                    ),
                ])
            experiment = {
                "schema": SCHEMA,
                "synthetic": True,
                "physics": PHYSICS,
                "tariff_manifest": str(TARIFF_MANIFEST),
                "tariff_manifest_sha256": sha256_file(TARIFF_MANIFEST),
                "giro40_duty_manifest": str(duty_manifest),
                "giro40_duty_manifest_sha256": sha256_file(duty_manifest),
                "cells": cells,
            }
            manifest = root / "experiment.json"
            manifest.write_text(json.dumps(experiment))
            first = root / "out1"
            second = root / "out2"
            build_evidence(manifest, first)
            build_evidence(manifest, second)
            required = {
                "tariff_manifest.csv", "giro40_duty_manifest.csv",
                "charging_blocks_long.csv",
                "tariff_response_summary.csv",
                "route_change_summary.csv",
                "fixed_duty_certificate_summary.csv",
                "cg_iteration_long.csv", "mip_checkpoint_long.csv",
                "artifact_inventory.csv", "data_dictionary.csv",
                "provenance.json", "gantt_three_tiers.png",
                "gantt_three_tiers.pdf", "gantt_plot.csv",
                "price_amplitude_response.png",
                "price_amplitude_response.pdf",
                "tariff_response_plot.csv",
                "negative_price_stress_plot.csv",
                "negative_price_stress.png",
                "negative_price_stress.pdf",
                "price_response_elasticity.csv",
                "primary_savings_summary.csv",
                "negative_price_stress_summary.csv",
                "SYNTHETIC_ONLY.txt",
            }
            self.assertTrue(required <= {path.name for path in first.iterdir()})
            first_files = {
                path.name for path in first.iterdir() if path.is_file()
            }
            second_files = {
                path.name for path in second.iterdir() if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            for name in first_files - {"provenance.json"}:
                self.assertEqual(
                    sha256_file(first / name), sha256_file(second / name),
                    name,
                )
            with (first / "tariff_response_plot.csv").open(
                newline=""
            ) as handle:
                primary_rows = list(csv.DictReader(handle))
            with (first / "negative_price_stress_plot.csv").open(
                newline=""
            ) as handle:
                stress_rows = list(csv.DictReader(handle))
            with (first / "price_response_elasticity.csv").open(
                newline=""
            ) as handle:
                elasticity_rows = list(csv.DictReader(handle))
            self.assertTrue(primary_rows)
            self.assertTrue(stress_rows)
            self.assertTrue(all(
                float(row["alpha"]) <= 1.0 for row in primary_rows
            ))
            self.assertTrue(all(
                float(row["alpha"]) == 2.0 for row in stress_rows
            ))
            self.assertTrue(all(
                float(row["alpha_right"]) <= 1.0
                and float(row["alpha_left"]) > 0.0
                for row in elasticity_rows
            ))
            reported_surplus = [
                float(row["terminal_surplus_total_kwh"])
                for row in stress_rows
                if row["terminal_surplus_total_kwh"] != ""
            ]
            self.assertTrue(reported_surplus)
            self.assertTrue(all(
                value > 0.0 for value in reported_surplus
            ))
            with (first / "tariff_response_summary.csv").open(
                newline=""
            ) as handle:
                summary_rows = list(csv.DictReader(handle))
            stress_summary = [
                row for row in summary_rows
                if row["analysis_role"] == "negative_price_stress"
            ]
            self.assertTrue(stress_summary)
            self.assertTrue(all(
                row["charging_only_savings_grid"] == ""
                and row["rerouting_increment_grid"] == ""
                and row["total_price_aware_savings_grid"] == ""
                for row in stress_summary
            ))
            with (first / "primary_savings_summary.csv").open(
                newline=""
            ) as handle:
                primary_savings = list(csv.DictReader(handle))
            self.assertTrue(primary_savings)
            self.assertTrue(all(
                row["analysis_role"] == "primary"
                and row["tariff_id"] != "peak12_alpha_2p0"
                for row in primary_savings
            ))
            self.assertIn(
                "NEGATIVE-PRICE STRESS (α=2)",
                (
                    REPO_ROOT / "src/build_tariff_response_evidence.py"
                ).read_text(),
            )
            corrupted = copy.deepcopy(cells[-1])
            corrupted["continuous_cost_pricing_certified"] = True
            with self.assertRaisesRegex(ValueError, "certificate"):
                _validate_cell(corrupted, self.tariff_by_id)
            terminal_mismatch = copy.deepcopy(cells[-1])
            terminal_mismatch["terminal_soc_policy"] = "terminal_soc_equal"
            with self.assertRaisesRegex(ValueError, "terminal SOC"):
                _validate_cell(terminal_mismatch, self.tariff_by_id)
            corrupted = copy.deepcopy(cells[-1])
            corrupted["terminal_soc_policy"] = "return_full"
            with self.assertRaisesRegex(ValueError, "terminal"):
                _validate_cell(corrupted, self.tariff_by_id)
            corrupted = copy.deepcopy(cells[-1])
            corrupted["treatment"] = "FIXED_GIRO"
            with self.assertRaisesRegex(ValueError, "fixed-duty label"):
                _validate_cell(corrupted, self.tariff_by_id)
            generic = copy.deepcopy(cells[-1])
            generic["tier"] = "TIER2_GIRO_AUGMENTED_ROUTE_CHARGING"
            generic["treatment"] = "GIRO-AUGMENTED"
            _validate_cell(generic, self.tariff_by_id)
            corrupted = copy.deepcopy(cells[-1])
            corrupted["routes"][0][
                "continuous_realized_charging_blocks"
            ] = None
            with self.assertRaisesRegex(ValueError, "lacks blocks"):
                _validate_cell(corrupted, self.tariff_by_id)

    def test_campaign_matrix_has_separate_k40_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, hashes = {}, {}
            for key in ("k5", "k8", "k40"):
                path = root / f"{key}.csv"
                scale = int(key[1:])
                path.write_text(
                    "Ordered_Trip_ID\n"
                    + "".join(f"{index}\n" for index in range(scale))
                )
                paths[key] = str(path)
                hashes[key] = sha256_file(path)
            frozen = root / "frozen.csv"
            with frozen.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "scale", "relative_path", "file_sha256",
                    "trip_count", "trip_set_sha256",
                ))
                writer.writeheader()
                for key in ("k5", "k8", "k40"):
                    scale = int(key[1:])
                    writer.writerow({
                        "scale": scale,
                        "relative_path": Path(paths[key]).name,
                        "file_sha256": hashes[key],
                        "trip_count": scale,
                        "trip_set_sha256": hashlib.sha256(json.dumps(
                            list(range(scale)),
                            sort_keys=True, separators=(",", ":"),
                        ).encode()).hexdigest(),
                    })
            identity = {
                "commit": "a" * 40,
                "detached": True,
                "tracked_clean": True,
                "reviewed_base":
                    "636dc0912f47e6ce85284fad3b36af30b4135887",
            }
            def fake_routes(_master, path):
                scale = int(Path(path).stem[1:])
                return [
                    {"trips": [index]} for index in range(scale)
                ]
            with (
                patch.object(
                    pilot, "giro_routes_for_instance",
                    side_effect=fake_routes,
                ),
                patch.object(
                    pilot, "FROZEN_K40_INSTANCE_SHA256", hashes["k40"]
                ),
                patch.object(
                    pilot.subprocess,
                    "run",
                    return_value=SimpleNamespace(
                        returncode=0,
                        stdout=json.dumps({
                            "python": "3.12.3",
                            "executable": str(Path(sys.executable).resolve()),
                            "executable_sha256": sha256_file(
                                Path(sys.executable).resolve()
                            ),
                            "numpy": "test", "pandas": "test",
                            "scipy": "test", "matplotlib": "test",
                            "gurobi": "test", "platform": "test",
                            "machine": "test", "pythonpath": None,
                            "numpy_build": None,
                        }),
                        stderr="",
                    ),
                ),
                patch.object(
                    pilot,
                    "build_preflight",
                    return_value={
                        "schema": "synthetic-unblocked-preflight",
                        "submission_blocked": False,
                    },
                ),
            ):
                plan = build_plan(
                    campaign="tariff-pilot-test",
                    instance_paths=paths,
                    instance_hashes=hashes,
                    tariff_manifest=TARIFF_MANIFEST,
                    identity=identity,
                    reservation_root=root / "reservations",
                    python_path=Path(sys.executable),
                    results_root=root / "results",
                    frozen_input_manifest=frozen,
                    frozen_input_manifest_sha256=sha256_file(frozen),
                )
                renamed_plan = build_plan(
                    campaign="tariff-pilot-renamed",
                    instance_paths=paths,
                    instance_hashes=hashes,
                    tariff_manifest=TARIFF_MANIFEST,
                    identity=identity,
                    reservation_root=root / "reservations",
                    python_path=Path(sys.executable),
                    results_root=root / "other-results",
                    frozen_input_manifest=frozen,
                    frozen_input_manifest_sha256=sha256_file(frozen),
                )
            self.assertEqual(plan["main_submission_job_count"], 111)
            self.assertEqual(plan["k40_preparation_job_count"], 33)
            self.assertFalse(plan["k40_mip_submission_allowed"])
            self.assertFalse(any(
                job["phase"] == "MIP" and job["scale"] == 40
                for job in plan["jobs"]
            ))
            stress_jobs = [
                job for job in plan["jobs"]
                if job["tariff_id"] == "peak12_alpha_2p0"
            ]
            self.assertEqual(len(stress_jobs), 13)
            self.assertTrue(all(
                job["analysis_role"] == "negative_price_stress"
                and str(job["primary_response_eligible"]).lower() == "false"
                for job in stress_jobs
            ))
            self.assertTrue(all(
                len(job["job_name"]) <= 15 for job in plan["jobs"]
            ))
            self.assertEqual(
                {
                    job["job_key"]: job["execution_digest"]
                    for job in plan["jobs"]
                },
                {
                    job["job_key"]: job["execution_digest"]
                    for job in renamed_plan["jobs"]
                },
            )
            campaign_root = Path(plan["campaign_root"])
            campaign_root.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                submit(plan, "f" * 64, k40_preparation=False)
            worker = (
                REPO_ROOT / "src/submit_tariff_response_pilot.sub"
            ).read_text()
            self.assertIn("EVSP_MIP_EXPECTED_RESULT_SHA256", worker)
            self.assertIn("EVSP_MIP_EXPECTED_JOURNAL_SHA256", worker)
            self.assertIn("--strict-tariff-coverage", worker)
            self.assertNotIn(
                "command+=(--initial-partition-routes", worker
            )
            runner = (
                REPO_ROOT / "src/run_exact_pool_mip.py"
            ).read_text()
            self.assertNotIn(
                'data_dir / Path(str(status["prices_csv"])).name',
                runner,
            )

    def test_fixed_duty_preflight_blocks_frozen_tariff_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for scale in (5, 8):
                for tariff_index in range(11):
                    seed_key = f"seed-k{scale}-{tariff_index}"
                    jobs.extend([
                        {
                            "job_key": seed_key,
                            "phase": "SEED",
                            "treatment": "GIRO-AUGMENTED",
                        },
                        {
                            "job_key": f"cg-{seed_key}",
                            "phase": "CG",
                            "treatment": "GIRO-AUGMENTED",
                        },
                        {
                            "job_key": f"mip-{seed_key}",
                            "phase": "MIP",
                            "treatment": "GIRO-AUGMENTED",
                        },
                    ])
            jobs.append({
                "job_key": "fixed-full",
                "phase": "FIXED_FULL",
                "treatment": "FIXED",
            })
            plan = {
                "physics": PHYSICS,
                "instances": {
                    "k5": {
                        "sha256":
                            "6ffea0b8cd3a9d15846946f6828705dd3431b7bafc69bd572ca30ed4530d5cb8",
                        "duty_count": 5,
                    },
                    "k8": {
                        "sha256":
                            "0d368920af0c5b14e0907b85977a9f72163a0cea6431c206f992e89aa31eb27f",
                        "duty_count": 8,
                    },
                    "k40": {
                        "sha256":
                            "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd",
                        "duty_count": 40,
                    },
                },
                "jobs": jobs,
                "campaign_root": str(root / "main"),
                "k40_campaign_root": str(root / "k40"),
            }
            preflight = build_fixed_duty_preflight(plan)
            by_key = {
                row["instance_key"]: row
                for row in preflight["instances"]
            }
            self.assertTrue(preflight["submission_blocked"])
            self.assertEqual(
                by_key["k5"]["primary_grid_representable_duty_count"], 1
            )
            self.assertEqual(
                by_key["k8"]["primary_grid_representable_duty_count"], 1
            )
            self.assertEqual(
                by_key["k40"]["primary_grid_representable_duty_count"], 9
            )
            self.assertEqual(
                by_key["k5"]["primary_grid_nonrepresentable_duty_count"], 4
            )
            self.assertEqual(
                by_key["k8"]["primary_grid_nonrepresentable_duty_count"], 7
            )
            self.assertEqual(
                len([
                    key for key in preflight["affected_job_keys"]
                    if key.startswith("seed-")
                ]),
                22,
            )
            plan["fixed_duty_submission_preflight"] = preflight
            plan["submission_blocked"] = True
            with (
                patch.object(
                    pilot,
                    "checkout_identity",
                    side_effect=AssertionError(
                        "checkout must not run for blocked preflight"
                    ),
                ),
                self.assertRaisesRegex(ValueError, "preflight"),
            ):
                pilot._submit_locked(
                    plan, "a" * 64, k40_preparation=False
                )

    def test_real_campaign_assembler_rejects_incomplete_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "jobs": [{
                    "job_key": "fixed",
                    "separate_k40_gate": False,
                    "submission_scope": MAIN_SUBMISSION_SCOPE,
                }]
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            (root / "approved-plan.json").write_bytes(raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": hashlib.sha256(raw).hexdigest(),
                "submission_scope": MAIN_SUBMISSION_SCOPE,
                "submitted_jobs": [],
            }))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                assemble(
                    root,
                    root / "manifest.json",
                    root / "evidence",
                )

    def test_gate_reconciliation_requires_completed_accounting_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {
                "job_key": "fixed",
                "job_name": "TF40Ffl",
                "partition": "default_partition",
                "execution_digest": "1" * 64,
                "submission_scope": MAIN_SUBMISSION_SCOPE,
                "dependency_key": None,
                "separate_k40_gate": False,
            }
            plan = {
                "schema": "test",
                "scheduler_identity": {"user": "nathan"},
                "jobs": [job],
            }
            plan_raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            (root / "approved-plan.json").write_bytes(plan_raw)
            spec = tariff_gate_spec(
                plan, plan_sha, MAIN_SUBMISSION_SCOPE, "12345"
            )
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "submission_scope": "main_k5_k8_pilot",
                "submitted_jobs": [
                    submitted_child_row(plan, job, "12345", "22345")
                ],
                "gate_state": "released",
                "gate_job_id": "12345",
                "gate_spec": spec,
            }))
            scheduler = SyntheticScheduler(
                live=[scheduler_result()],
                controller=[
                    scheduler_result(1, stderr="Invalid job id specified")
                ],
                accounting=[accounting_gate_row(spec)],
            )
            payload = reconcile(
                root, plan_sha, runner=scheduler, sleeper=lambda _value: None
            )
            self.assertEqual(
                payload["gate_state"], "completed_verified"
            )
            self.assertTrue(payload["submitted"])
            verified_gate_evidence(payload, spec)

    def test_gate_intent_is_discovered_after_accepted_before_record_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {
                "job_key": "fixed",
                "job_name": "TF40Ffl",
                "partition": "default_partition",
                "execution_digest": "1" * 64,
                "submission_scope": MAIN_SUBMISSION_SCOPE,
                "dependency_key": None,
                "separate_k40_gate": False,
            }
            plan = {
                "schema": "test",
                "scheduler_identity": {"user": "nathan"},
                "jobs": [job],
            }
            plan_raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            (root / "approved-plan.json").write_bytes(plan_raw)
            intent = tariff_gate_spec(
                plan, plan_sha, MAIN_SUBMISSION_SCOPE
            )
            spec = tariff_gate_spec(
                plan, plan_sha, MAIN_SUBMISSION_SCOPE, "12345"
            )
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "submission_scope": "main_k5_k8_pilot",
                "submitted": False,
                "submitted_jobs": [
                    submitted_child_row(plan, job, "12345", "22345")
                ],
                "gate_state": "submission_intent",
                "gate_submission_intent": intent,
            }))
            scheduler = SyntheticScheduler(
                live=[
                    live_gate_row(spec),
                    live_gate_row(spec),
                    live_gate_row(spec),
                    live_gate_row(
                        spec, state="RUNNING", reason="None"
                    ),
                ],
                release=[scheduler_result()],
            )
            payload = reconcile(
                root,
                plan_sha,
                runner=scheduler,
                sleeper=lambda _value: None,
            )
            self.assertEqual(payload["gate_job_id"], "12345")
            self.assertEqual(payload["gate_spec"], spec)
            self.assertEqual(
                payload["gate_state"], "released_verified"
            )
            self.assertTrue(payload["submitted"])
            self.assertEqual(sum(
                Path(command[0]).name == "scontrol"
                and command[1] == "release"
                for command in scheduler.commands
            ), 1)

    def test_child_intent_is_discovered_and_dependency_receipt_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {
                "job_key": "fixed",
                "job_name": "TF40Ffl",
                "partition": "default_partition",
                "execution_digest": "1" * 64,
                "submission_scope": MAIN_SUBMISSION_SCOPE,
                "dependency_key": None,
                "separate_k40_gate": False,
            }
            plan = {
                "scheduler_identity": {"user": "nathan"},
                "jobs": [job],
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            gate_spec = tariff_gate_spec(
                plan, plan_sha, MAIN_SUBMISSION_SCOPE, "12345"
            )
            child_intent = tariff_child_spec(
                plan, job, "afterok:12345"
            )
            child_spec = tariff_child_spec(
                plan, job, "afterok:12345", "22345"
            )
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "submission_scope": "main_k5_k8_pilot",
                "submitted": False,
                "submitted_jobs": [],
                "job_submission_intents": {"fixed": child_intent},
                "gate_state": "held_after_partial_submission",
                "gate_job_id": "12345",
                "gate_spec": gate_spec,
            }))
            scheduler = SyntheticScheduler(
                live=[
                    live_child_row(
                        child_spec, include_dependency=False
                    ),
                    live_child_row(
                        child_spec, include_dependency=True
                    ),
                    live_gate_row(gate_spec),
                    live_gate_row(gate_spec),
                    live_gate_row(
                        gate_spec, state="RUNNING", reason="None"
                    ),
                ],
                release=[scheduler_result()],
            )
            payload = reconcile(
                root,
                plan_sha,
                runner=scheduler,
                sleeper=lambda _value: None,
            )
            self.assertEqual(
                payload["job_submission_intents"], {}
            )
            self.assertEqual(
                payload["submitted_jobs"][0]["job_id"], "22345"
            )
            self.assertTrue(
                payload["submitted_jobs"][0][
                    "submission_receipt"
                ]["verified"]
            )
            self.assertTrue(payload["submitted"])

    def test_tariff_gate_scope_identity_is_disjoint_and_fail_closed(self):
        plan = {"scheduler_identity": {"user": "nathan"}}
        plan_sha = "a" * 64
        main = tariff_gate_spec(
            plan, plan_sha, MAIN_SUBMISSION_SCOPE, "12345"
        )
        k40 = tariff_gate_spec(
            plan, plan_sha, K40_SUBMISSION_SCOPE, "22345"
        )
        for field in ("job_name", "comment", "role", "submission_scope"):
            self.assertNotEqual(main[field], k40[field])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full_plan = {
                "scheduler_identity": {"user": "nathan"},
                "jobs": [],
            }
            raw = json.dumps(
                full_plan, sort_keys=True, separators=(",", ":")
            ).encode()
            observed_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            wrong = tariff_gate_spec(
                full_plan,
                observed_sha,
                K40_SUBMISSION_SCOPE,
                "22345",
            )
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": observed_sha,
                "submission_scope": MAIN_SUBMISSION_SCOPE,
                "submitted": False,
                "submitted_jobs": [],
                "gate_state": "held_verified",
                "gate_job_id": "22345",
                "gate_spec": wrong,
            }))
            scheduler = SyntheticScheduler()
            with self.assertRaisesRegex(ValueError, "specification"):
                reconcile(
                    root,
                    observed_sha,
                    runner=scheduler,
                    sleeper=lambda _value: None,
                )
            self.assertFalse(any(
                Path(command[0]).name == "scontrol"
                and len(command) > 1 and command[1] == "release"
                for command in scheduler.commands
            ))

    def test_ambiguous_main_receipt_never_adopts_k40_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "scheduler_identity": {"user": "nathan"},
                "jobs": [],
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            main_intent = tariff_gate_spec(
                plan, plan_sha, MAIN_SUBMISSION_SCOPE
            )
            k40_spec = tariff_gate_spec(
                plan, plan_sha, K40_SUBMISSION_SCOPE, "22345"
            )
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "submission_scope": MAIN_SUBMISSION_SCOPE,
                "submitted": False,
                "submitted_jobs": [],
                "gate_state": "submission_intent",
                "gate_submission_intent": main_intent,
            }))
            scheduler = SyntheticScheduler(
                live=[live_gate_row(k40_spec)] * 5
            )
            with self.assertRaisesRegex(RuntimeError, "ambiguous"):
                reconcile(
                    root,
                    plan_sha,
                    runner=scheduler,
                    sleeper=lambda _value: None,
                )
            persisted = json.loads(
                (root / "campaign.json").read_text()
            )
            self.assertEqual(
                persisted["gate_state"], "ambiguous_gate_receipt"
            )
            self.assertNotIn("gate_job_id", persisted)

    def test_main_and_k40_recovery_can_complete_simultaneously(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            main_job = {
                "job_key": "main-job",
                "job_name": "TM5Rfl",
                "partition": "default_partition",
                "execution_digest": "1" * 64,
                "dependency_key": None,
                "separate_k40_gate": False,
                "submission_scope": MAIN_SUBMISSION_SCOPE,
            }
            k40_job = {
                "job_key": "k40-job",
                "job_name": "TK40Rfl",
                "partition": "default_partition",
                "execution_digest": "2" * 64,
                "dependency_key": None,
                "separate_k40_gate": True,
                "submission_scope": K40_SUBMISSION_SCOPE,
            }
            plan = {
                "scheduler_identity": {"user": "nathan"},
                "jobs": [main_job, k40_job],
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(raw).hexdigest()
            cases = []
            for scope, job, gate_id, child_id in (
                (MAIN_SUBMISSION_SCOPE, main_job, "12345", "12346"),
                (K40_SUBMISSION_SCOPE, k40_job, "22345", "22346"),
            ):
                root = parent / scope
                root.mkdir()
                (root / "approved-plan.json").write_bytes(raw)
                gate_spec = tariff_gate_spec(
                    plan, plan_sha, scope, gate_id
                )
                (root / "campaign.json").write_text(json.dumps({
                    "approval_sha256": plan_sha,
                    "submission_scope": scope,
                    "submitted": False,
                    "submitted_jobs": [
                        submitted_child_row(
                            plan, job, gate_id, child_id
                        )
                    ],
                    "gate_state": "held_verified",
                    "gate_job_id": gate_id,
                    "gate_spec": gate_spec,
                }))
                scheduler = SyntheticScheduler(
                    live=[
                        live_gate_row(gate_spec),
                        live_gate_row(gate_spec),
                        live_gate_row(
                            gate_spec, state="RUNNING", reason="None"
                        ),
                    ],
                    release=[scheduler_result()],
                )
                cases.append((root, scheduler, gate_spec))

            def recover(case):
                root, scheduler, _spec = case
                return reconcile(
                    root,
                    plan_sha,
                    runner=scheduler,
                    sleeper=lambda _value: None,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(recover, cases))
            self.assertTrue(all(result["submitted"] for result in results))
            self.assertNotEqual(
                results[0]["gate_spec"]["comment"],
                results[1]["gate_spec"]["comment"],
            )

    def test_release_rc_zero_without_transition_fails_closed(self):
        spec = gate_fixture_spec()
        scheduler = SyntheticScheduler(
            live=[live_gate_row(spec)] * 3,
            release=[scheduler_result()],
        )
        with self.assertRaisesRegex(
            SlurmContractError, "postcondition was not observed"
        ):
            release_with_postcondition(
                spec,
                runner=scheduler,
                sleeper=lambda _value: None,
                command_attempts=1,
                verify_attempts=2,
            )

    def test_release_tolerates_stale_reads_and_nonzero_request(self):
        spec = gate_fixture_spec()
        for request_result in (
            scheduler_result(),
            scheduler_result(1, stderr="controller reported stale hold"),
        ):
            with self.subTest(returncode=request_result.returncode):
                scheduler = SyntheticScheduler(
                    live=[
                        live_gate_row(spec),
                        live_gate_row(spec),
                        live_gate_row(
                            spec, state="RUNNING", reason="None"
                        ),
                    ],
                    release=[request_result],
                )
                verification = release_with_postcondition(
                    spec,
                    runner=scheduler,
                    sleeper=lambda _value: None,
                )
                self.assertTrue(verification["verified"])
                self.assertEqual(
                    verification["observation"]["state"], "RUNNING"
                )

    def test_release_retries_only_after_bounded_held_window(self):
        spec = gate_fixture_spec()
        scheduler = SyntheticScheduler(
            live=[
                live_gate_row(spec),
                live_gate_row(spec),
                live_gate_row(spec),
                live_gate_row(spec, state="CONFIGURING", reason="None"),
            ],
            release=[scheduler_result(), scheduler_result()],
        )
        verification = release_with_postcondition(
            spec,
            runner=scheduler,
            sleeper=lambda _value: None,
            verify_attempts=2,
        )
        releases = [
            command for command in scheduler.commands
            if Path(command[0]).name == "scontrol"
            and command[1] == "release"
        ]
        self.assertEqual(len(releases), 2)
        self.assertEqual(verification["command_attempts"], 2)

    def test_scheduler_queries_fail_closed_then_transiently_recover(self):
        spec = gate_fixture_spec()
        failed = SyntheticScheduler(
            live=[scheduler_result(1, stderr="down")],
            controller=[scheduler_result(1, stderr="down")],
            accounting=[scheduler_result(1, stderr="down")],
        )
        with self.assertRaisesRegex(
            SlurmContractError, "could not be resolved"
        ):
            resolve_exact_job(spec, runner=failed)

        transient = SyntheticScheduler(
            live=[scheduler_result(1, stderr="transient")],
            controller=[scheduler_result(stdout=(
                f"JobId={spec['job_id']} UserId={spec['user']}(1000) "
                f"JobName={spec['job_name']} JobState=PENDING "
                f"Partition={spec['partition']} Reason=JobHeldUser "
                f"Comment={spec['comment']} ExitCode=0:0\n"
            ))],
        )
        observation = resolve_exact_job(spec, runner=transient)
        self.assertEqual(observation["source"], "scontrol")
        self.assertEqual(observation["reason"], "JobHeldUser")

    def test_nonterminal_accounting_lag_is_retried_boundedly(self):
        spec = gate_fixture_spec()
        scheduler = SyntheticScheduler(
            live=[
                scheduler_result(),
                live_gate_row(spec),
                live_gate_row(spec, state="RUNNING", reason="None"),
            ],
            controller=[
                scheduler_result(1, stderr="Invalid job id specified")
            ],
            accounting=[
                accounting_gate_row(
                    spec, state="PENDING", exit_code="0:0"
                )
            ],
            release=[scheduler_result()],
        )
        verification = release_with_postcondition(
            spec,
            runner=scheduler,
            sleeper=lambda _value: None,
            command_attempts=1,
            verify_attempts=2,
        )
        self.assertEqual(
            verification["observation"]["state"], "RUNNING"
        )
        self.assertEqual(verification["command_attempts"], 1)

    def test_identity_mismatch_before_and_after_release_is_rejected(self):
        spec = gate_fixture_spec()
        before = SyntheticScheduler(
            live=[live_gate_row(spec, comment="wrong")]
        )
        with self.assertRaisesRegex(
            SlurmContractError, "identity mismatch"
        ):
            release_with_postcondition(spec, runner=before)

        after = SyntheticScheduler(
            live=[
                live_gate_row(spec),
                live_gate_row(spec, comment="wrong"),
            ],
            release=[scheduler_result()],
        )
        with self.assertRaisesRegex(
            SlurmContractError, "identity mismatch"
        ):
            release_with_postcondition(
                spec, runner=after, sleeper=lambda _value: None
            )

    def test_release_state_and_dependency_classification(self):
        spec = gate_fixture_spec()
        for state, reason in (
            ("CONFIGURING", "None"),
            ("RUNNING", "None"),
            ("COMPLETING", "None"),
        ):
            scheduler = SyntheticScheduler(
                live=[live_gate_row(spec, state=state, reason=reason)]
            )
            result = release_with_postcondition(spec, runner=scheduler)
            self.assertEqual(result["observation"]["state"], state)

        dependency = SyntheticScheduler(
            live=[live_gate_row(spec, reason="Dependency")]
        )
        with self.assertRaisesRegex(
            SlurmContractError, "valid release precondition"
        ):
            release_with_postcondition(spec, runner=dependency)
        valid_dependency = SyntheticScheduler(
            live=[live_gate_row(spec, reason="Dependency")]
        )
        result = release_with_postcondition(
            spec, runner=valid_dependency, dependency_is_valid=True
        )
        self.assertTrue(result["verified"])
        never = SyntheticScheduler(
            live=[
                live_gate_row(spec, reason="DependencyNeverSatisfied")
            ]
        )
        with self.assertRaises(SlurmContractError):
            release_with_postcondition(
                spec, runner=never, dependency_is_valid=True
            )

    def test_terminal_exit_code_and_held_receipt_contracts(self):
        spec = gate_fixture_spec()
        success = SyntheticScheduler(
            live=[scheduler_result()],
            controller=[scheduler_result(1, stderr="Invalid job id")],
            accounting=[accounting_gate_row(spec)],
        )
        result = release_with_postcondition(spec, runner=success)
        self.assertEqual(result["observation"]["exit_code"], "0:0")

        for state, exit_code in (("FAILED", "1:0"), ("COMPLETED", "1:0")):
            with self.subTest(state=state):
                failed = SyntheticScheduler(
                    live=[scheduler_result()],
                    controller=[
                        scheduler_result(1, stderr="Invalid job id")
                    ],
                    accounting=[
                        accounting_gate_row(
                            spec, state=state, exit_code=exit_code
                        )
                    ],
                )
                with self.assertRaises(SlurmContractError):
                    release_with_postcondition(spec, runner=failed)

        missing = SyntheticScheduler(
            live=[scheduler_result()],
            controller=[scheduler_result(1, stderr="Invalid job id")],
            accounting=[
                accounting_gate_row(spec, state="FAILED", exit_code="")
            ],
        )
        with self.assertRaisesRegex(SlurmContractError, "exit code"):
            resolve_exact_job(spec, runner=missing)

        held = SyntheticScheduler(live=[live_gate_row(spec)])
        receipt = verify_held_receipt(
            spec, runner=held, sleeper=lambda _value: None
        )
        self.assertTrue(receipt["verified"])

    def test_cached_released_state_is_reobserved_and_failure_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "scheduler_identity": {"user": "nathan"},
                "jobs": [{
                    "job_key": "fixed",
                    "separate_k40_gate": False,
                }],
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            plan_sha = hashlib.sha256(raw).hexdigest()
            (root / "approved-plan.json").write_bytes(raw)
            spec = tariff_gate_spec(
                plan, plan_sha, MAIN_SUBMISSION_SCOPE, "12345"
            )
            manifest_path = root / "campaign.json"
            manifest_path.write_text(json.dumps({
                "approval_sha256": plan_sha,
                "submission_scope": "main_k5_k8_pilot",
                "submitted": True,
                "submitted_jobs": [],
                "gate_state": "released",
                "gate_job_id": "12345",
                "gate_spec": spec,
            }))
            scheduler = SyntheticScheduler(
                live=[
                    scheduler_result(),
                    scheduler_result(),
                ],
                controller=[
                    scheduler_result(1, stderr="Invalid job id"),
                    scheduler_result(1, stderr="Invalid job id"),
                ],
                accounting=[
                    accounting_gate_row(
                        spec, state="FAILED", exit_code="1:0"
                    ),
                    accounting_gate_row(
                        spec, state="FAILED", exit_code="1:0"
                    ),
                ],
            )
            with self.assertRaisesRegex(ValueError, "terminal"):
                reconcile(
                    root, plan_sha, runner=scheduler,
                    sleeper=lambda _value: None,
                )
            persisted = json.loads(manifest_path.read_text())
            self.assertEqual(persisted["gate_state"], "terminal_failed")
            self.assertFalse(persisted["submitted"])
            self.assertEqual(
                persisted["gate_terminal_failure"]["exit_code"], "1:0"
            )

    def test_legacy_released_evidence_is_readable_but_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan_raw = b'{"jobs":[]}'
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            (root / "approved-plan.json").write_bytes(plan_raw)
            manifest_path = root / "campaign.json"
            manifest_path.write_text(json.dumps({
                "approval_sha256": plan_sha,
                "gate_state": "released",
                "gate_job_id": "12345",
            }))
            with self.assertRaisesRegex(ValueError, "legacy"):
                reconcile(root, plan_sha)
            persisted = json.loads(manifest_path.read_text())
            self.assertEqual(persisted["legacy_gate_state"], "released")
            self.assertEqual(persisted["gate_state"], "legacy_unverified")
            self.assertFalse(
                persisted["gate_release_verification"]["verified"]
            )

    def test_archive_rejects_symlinks_and_swapped_reservations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.write_text("bytes")
            linked = root / "linked"
            linked.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "regular file"):
                archive_sha(linked)
            selected = {
                "a": {"job_key": "a", "execution_digest": "a" * 64},
                "b": {"job_key": "b", "execution_digest": "b" * 64},
            }
            files = [
                root / f"{selected[key]['execution_digest']}.json"
                for key in ("a", "b")
            ]
            for path, wrong in zip(files, ("b", "a")):
                path.write_text(json.dumps({
                    "schema":
                        "evsp-dr-tariff-response-reservation-v1",
                    "plan_sha256": "p" * 64,
                    "job_key": wrong,
                    "execution_digest":
                        selected[wrong]["execution_digest"],
                    "submission_scope": MAIN_SUBMISSION_SCOPE,
                }))
            with self.assertRaisesRegex(ValueError, "content"):
                validate_reservations(
                    files,
                    [str(path) for path in files],
                    selected,
                    "p" * 64,
                    MAIN_SUBMISSION_SCOPE,
                )

    def test_tariff_reservations_are_crash_adoptable_and_cross_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {
                "job_key": "fixed",
                "execution_digest": "a" * 64,
                "submission_scope": MAIN_SUBMISSION_SCOPE,
            }
            plan = {
                "campaign": "campaign-a",
                "reservation_root": str(root / "reservations"),
            }
            paths, transaction = pilot._reserve(
                plan, "1" * 64, [job], MAIN_SUBMISSION_SCOPE
            )
            adopted, adopted_transaction = pilot._reserve(
                plan, "1" * 64, [job], MAIN_SUBMISSION_SCOPE
            )
            self.assertEqual(paths, adopted)
            self.assertEqual(transaction, adopted_transaction)
            k40_job = {
                **job, "submission_scope": K40_SUBMISSION_SCOPE,
            }
            k40_paths, k40_transaction = pilot._reserve(
                plan, "1" * 64, [k40_job], K40_SUBMISSION_SCOPE
            )
            self.assertNotEqual(paths, k40_paths)
            self.assertNotEqual(transaction, k40_transaction)
            conflicting = {
                **plan, "campaign": "campaign-b",
            }
            with self.assertRaisesRegex(ValueError, "conflict"):
                pilot._reserve(
                    conflicting, "2" * 64, [job],
                    MAIN_SUBMISSION_SCOPE,
                )

    def test_worker_completion_cannot_be_swapped_between_same_plan_jobs(self):
        job = {
            "job_key": "job-a",
            "execution_digest": "a" * 64,
            "phase": "CG",
            "treatment": "RAW",
            "analysis_role": "primary",
            "scale": 5,
            "tariff_id": "flat",
            "instance": {"sha256": "b" * 64},
            "tariff_sha256": "c" * 64,
            "submission_scope": MAIN_SUBMISSION_SCOPE,
        }
        completion = {
            "schema": COMPLETION_SCHEMA,
            "job_key": "job-b",
            "execution_digest": "d" * 64,
            "phase": "CG",
            "treatment": "RAW",
            "analysis_role": "primary",
            "scale": 5,
            "tariff_id": "flat",
            "plan_sha256": "e" * 64,
            "instance_sha256": "b" * 64,
            "tariff_sha256": "c" * 64,
            "submission_scope": MAIN_SUBMISSION_SCOPE,
            "slurm_job_id": "123",
            "artifact_sha256": {"/tmp/result": "f" * 64},
        }
        with self.assertRaisesRegex(ValueError, "job-a"):
            validate_completion_identity(
                completion,
                job,
                "e" * 64,
                expected_slurm_job_id="123",
                expected_artifact_paths={"/tmp/result"},
            )
        incomplete = {
            **completion,
            "job_key": "job-a",
            "execution_digest": "a" * 64,
        }
        with self.assertRaisesRegex(ValueError, "artifact_set"):
            validate_completion_identity(
                incomplete,
                job,
                "e" * 64,
                expected_slurm_job_id="123",
                expected_artifact_paths={
                    "/tmp/result", "/tmp/result.iters.csv",
                },
            )


if __name__ == "__main__":
    unittest.main()
