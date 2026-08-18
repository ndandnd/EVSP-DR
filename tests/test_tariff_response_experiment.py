import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
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
    build_plan,
    submit,
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


TARIFF_MANIFEST = REPO_ROOT / "data/tariff_response/tariff_manifest.csv"


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
        self.assertTrue(all(
            int(row["coverage_end_hour"]) == 26 for row in self.tariffs
        ))

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
                cells.extend([
                    self._synthetic_cell(
                        tariff, "TIER0_GIRO_ORIGINAL", "GIRO_ORIGINAL"
                    ),
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
                "SYNTHETIC_ONLY.txt",
            }
            self.assertTrue(required <= {path.name for path in first.iterdir()})
            for name in required - {"provenance.json"}:
                self.assertEqual(
                    sha256_file(first / name), sha256_file(second / name),
                    name,
                )
            corrupted = copy.deepcopy(cells[-1])
            corrupted["continuous_cost_pricing_certified"] = True
            with self.assertRaisesRegex(ValueError, "certificate"):
                _validate_cell(corrupted, self.tariff_by_id)
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
                path.write_text(f"{key}\n")
                paths[key] = str(path)
                hashes[key] = sha256_file(path)
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
                )
            self.assertEqual(plan["main_submission_job_count"], 111)
            self.assertEqual(plan["k40_preparation_job_count"], 33)
            self.assertFalse(plan["k40_mip_submission_allowed"])
            self.assertFalse(any(
                job["phase"] == "MIP" and job["scale"] == 40
                for job in plan["jobs"]
            ))
            self.assertTrue(all(
                len(job["job_name"]) <= 15 for job in plan["jobs"]
            ))
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

    def test_real_campaign_assembler_rejects_incomplete_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {
                "jobs": [{
                    "job_key": "fixed",
                    "separate_k40_gate": False,
                }]
            }
            raw = json.dumps(
                plan, sort_keys=True, separators=(",", ":")
            ).encode()
            (root / "approved-plan.json").write_bytes(raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": hashlib.sha256(raw).hexdigest(),
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
            plan_raw = b'{"schema":"test"}'
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            (root / "approved-plan.json").write_bytes(plan_raw)
            (root / "campaign.json").write_text(json.dumps({
                "approval_sha256": plan_sha,
                "gate_state": "release_attempting",
                "gate_job_id": "12345",
            }))
            with patch(
                "reconcile_tariff_response_gate.subprocess.run",
                return_value=SimpleNamespace(
                    returncode=0,
                    stdout="12345|COMPLETED|\n",
                    stderr="",
                ),
            ):
                payload = reconcile(root, plan_sha)
            self.assertEqual(
                payload["gate_state"], "released_reconciled"
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
                }))
            with self.assertRaisesRegex(ValueError, "content"):
                validate_reservations(
                    files,
                    [str(path) for path in files],
                    selected,
                    "p" * 64,
                )


if __name__ == "__main__":
    unittest.main()
