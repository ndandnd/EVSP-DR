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

from audit_giro_known_columns import DEPOT, HORIZON_MIN  # noqa: E402
from audit_expanded_pool_physical import audit_pools  # noqa: E402
from expanded_path_realization import (  # noqa: E402
    EVENT_REALIZATION_SCHEMA,
    blocks_from_continuous_stops,
    realize_expanded_path,
    realized_costs,
    validate_continuous_charging_blocks,
)
from run_exact_pool_mip import (  # noqa: E402
    prepare_strict_partition_pool,
    validate_injected_route,
)
from utils_v2 import calculate_truck_route_cost_accurate  # noqa: E402


def problem(trip_energy, edges):
    trips = list(range(len(trip_energy)))
    adjacency = {}
    for source, target in edges:
        adjacency.setdefault(source, []).append(
            (target, 0.0, 0.0, "test")
        )
    return SimpleNamespace(
        trips=trips,
        trip_energy=dict(enumerate(trip_energy)),
        start_min={trip: trip * 10.0 for trip in trips},
        end_min={trip: trip * 10.0 + 5.0 for trip in trips},
        adjacency=adjacency,
    )


class ExpandedPathRealizationTests(unittest.TestCase):
    def test_accumulated_floor_residual_repairs_large_overcharge(self):
        station = "3127L_0"
        route_nodes = [DEPOT, 0, 1, 2, station, 3, DEPOT]
        p = problem(
            [0.1, 0.1, 0.1, 0.1],
            list(zip(route_nodes, route_nodes[1:])),
        )
        p.start_min[3] = 40.0
        p.end_min[3] = 45.0
        record = {
            "trips": [0, 1, 2, 3],
            "route_nodes": route_nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [30],
                "cet": [40],
                "kwh": [45.0],
            },
            "cost": 100035.0,
        }
        reason = validate_injected_route(
            p, record, 300.0, 300.0, 0.0, HORIZON_MIN
        )
        self.assertIn("raises SOC to 344.7", reason)
        realized, detail = realize_expanded_path(
            p, record, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        self.assertEqual(
            detail["classification"], "deterministically_repairable"
        )
        self.assertAlmostEqual(
            realized["charging_stops"]["kwh"][0], 0.3, places=6
        )
        self.assertIsNone(validate_injected_route(
            p, realized, 300.0, 300.0, 0.0, HORIZON_MIN
        ))
        self.assertEqual(realized["trips"], record["trips"])
        self.assertEqual(realized["route_nodes"], record["route_nodes"])
        self.assertEqual(
            realized["charging_stops"]["cst"],
            record["charging_stops"]["cst"],
        )

    def test_sub_kwh_capacity_boundary_is_realized_exactly(self):
        station = "2190L_0"
        route_nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem(
            [14.5, 0.1],
            list(zip(route_nodes, route_nodes[1:])),
        )
        p.start_min[1] = 20.0
        p.end_min[1] = 25.0
        record = {
            "trips": [0, 1],
            "route_nodes": route_nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [10],
                "cet": [20],
                "kwh": [15.0],
            },
            "cost": 100020.0,
        }
        self.assertIn("300.5", validate_injected_route(
            p, record, 300, 300, 0, HORIZON_MIN
        ))
        realized, _detail = realize_expanded_path(
            p, record, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        self.assertAlmostEqual(
            realized["charging_stops"]["kwh"][0], 14.5, places=9
        )
        self.assertIsNone(validate_injected_route(
            p, realized, 300, 300, 0, HORIZON_MIN
        ))
        realized["expanded_grid_charging_stops"] = record[
            "charging_stops"
        ]
        replayed, replay_detail = realize_expanded_path(
            p, realized, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        self.assertEqual(
            replayed["charging_stops"], realized["charging_stops"]
        )
        self.assertEqual(
            replay_detail["mapping"]["recorded_total_kwh"], 15.0
        )

    def test_arrival_grace_does_not_create_charging_power(self):
        station = "S"
        p = SimpleNamespace(
            trips=[0, 1],
            trip_energy={0: 0.0, 1: 0.0},
            start_min={0: 0.0, 1: 20.0},
            end_min={0: 0.0, 1: 25.0},
            adjacency={
                DEPOT: [(0, 0.0, 0.0, "test")],
                0: [(station, 11.0, 0.0, "test")],
                station: [(1, 0.0, 0.0, "test")],
                1: [(DEPOT, 0.0, 0.0, "test")],
            },
        )
        record = {
            "trips": [0, 1],
            "route_nodes": [DEPOT, 0, station, 1, DEPOT],
            "charging_stops": {
                "stations": [station],
                "cst": [10],
                "cet": [20],
                "kwh": [50.0],
            },
        }
        self.assertIn("exceeds 300 kW", validate_injected_route(
            p, record, 300, 300, 0, HORIZON_MIN
        ))
        with self.assertRaisesRegex(ValueError, "after priced"):
            blocks_from_continuous_stops(
                {
                    "charging_stops": {
                        "stations": [station],
                        "cst": [10], "cet": [20], "kwh": [45.0],
                    },
                },
                station_prices={"S": {0: 0.1}},
                charge_kw=300,
                earliest_start_by_stop=[11.0],
            )
        non_grid = {
            **record,
            "charging_stops": {
                "stations": [station],
                "cst": [11],
                "cet": [20],
                "kwh": [45.0],
            },
            "cost": 100050.0,
        }
        realized, detail = realize_expanded_path(
            p, non_grid, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        self.assertIsNone(realized)
        self.assertIn("non-grid charging window", detail["reason"])

    def test_event_contract_accepts_irregular_window_with_full_validation(self):
        station = "S"
        nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem([14.5, 0.1], list(zip(nodes, nodes[1:])))
        record = {
            "trips": [0, 1],
            "route_nodes": nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [5.5], "cet": [8.5], "kwh": [15.0],
            },
            "cost": 100006.65,
        }
        rejected, detail = realize_expanded_path(
            p, record, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        self.assertIsNone(rejected)
        self.assertIn("non-grid charging window", detail["reason"])

        realized, detail = realize_expanded_path(
            p, record, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10, time_model="event",
        )
        self.assertEqual(detail["mapping"]["schema"], EVENT_REALIZATION_SCHEMA)
        self.assertTrue(
            detail["mapping"]["irregular_charging_windows_accepted"]
        )
        self.assertAlmostEqual(
            realized["charging_stops"]["kwh"][0], 14.5, places=9,
        )
        self.assertIsNone(validate_injected_route(
            p, realized, 300, 300, 0, HORIZON_MIN,
            arrival_grace_min=0.0,
        ))
        costs = realized_costs(
            realized,
            detail["mapping"],
            station_prices={"S": {0: 0.1}},
        )
        self.assertEqual(
            sum(
                block["expanded_grid_kwh"]
                for block in costs["continuous_realized_charging_blocks"]
            ),
            15.0,
        )
        self.assertEqual(
            sum(
                block["realized_kwh"]
                for block in costs["continuous_realized_charging_blocks"]
            ),
            14.5,
        )

        too_short = {
            **record,
            "charging_stops": {
                "stations": [station],
                "cst": [5.5], "cet": [8.4], "kwh": [15.0],
            },
        }
        rejected, detail = realize_expanded_path(
            p, too_short, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10, time_model="event",
        )
        self.assertIsNone(rejected)
        self.assertIn("charger power", detail["reason"])

    def test_emitted_windows_must_match_expanded_grid_path(self):
        station = "S"
        nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem([14.5, 0.1], list(zip(nodes, nodes[1:])))
        record = {
            "trips": [0, 1],
            "route_nodes": nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [20], "cet": [30], "kwh": [14.5],
            },
            "expanded_grid_charging_stops": {
                "stations": [station],
                "cst": [10], "cet": [20], "kwh": [15.0],
            },
            "cost": 100006.5,
        }
        realized, detail = realize_expanded_path(
            p, record, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        self.assertIsNone(realized)
        self.assertIn("emitted cst differs", detail["reason"])

    def test_multiblock_tariff_schedule_reproduces_realized_cost(self):
        station = "PARX_1"
        nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem([70.0, 0.1], list(zip(nodes, nodes[1:])))
        p.start_min[1] = 70.0
        p.end_min[1] = 75.0
        record = {
            "trips": [0, 1],
            "route_nodes": nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [50], "cet": [70], "kwh": [75.0],
            },
            "cost": 100024.5,
        }
        realized, detail = realize_expanded_path(
            p, record, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        prices = {"PARX": {0: 0.1, 1: 0.5}}
        costs = realized_costs(
            realized, detail["mapping"], station_prices=prices
        )
        blocks = costs["continuous_realized_charging_blocks"]
        self.assertEqual(len(blocks), 2)
        self.assertEqual(
            [block["tariff_hour"] for block in blocks], [0, 1]
        )
        self.assertEqual(
            [block["realized_kwh"] for block in blocks], [40.0, 30.0]
        )
        self.assertAlmostEqual(
            costs["continuous_realized_cost"], 100024.0
        )
        aggregate_only = calculate_truck_route_cost_accurate(
            realized,
            100000.0,
            prices["PARX"],
            charge_rate_kw=300,
            station_hourly_prices=prices,
            charge_start_cost=5.0,
        )
        self.assertAlmostEqual(aggregate_only, 100020.0)
        self.assertNotEqual(
            aggregate_only, costs["continuous_realized_cost"]
        )
        validation = validate_continuous_charging_blocks(
            realized,
            blocks,
            station_prices=prices,
            charge_kw=300,
            expected_continuous_cost=costs[
                "continuous_realized_cost"
            ],
        )
        self.assertEqual(
            validation["block_schedule_sha256"],
            costs["continuous_realized_charging_blocks_sha256"],
        )
        overlapping = json.loads(json.dumps(blocks))
        overlapping[1]["start_min"] = 55.0
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_continuous_charging_blocks(
                realized,
                overlapping,
                station_prices=prices,
                charge_kw=300,
            )
        wrong_grid = json.loads(json.dumps(blocks))
        wrong_grid[0]["expanded_grid_kwh"] -= 1.0
        with self.assertRaisesRegex(ValueError, "grid kWh"):
            validate_continuous_charging_blocks(
                realized,
                wrong_grid,
                station_prices=prices,
                charge_kw=300,
            )

    def test_station_entry_uses_continuous_prefloor_reserve(self):
        station = "S"
        nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem([10.1, 0.1], list(zip(nodes, nodes[1:])))
        p.start_min[1] = 20.0
        p.end_min[1] = 25.0
        record = {
            "trips": [0, 1],
            "route_nodes": nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [10], "cet": [20], "kwh": [10.0],
            },
            "cost": 100006.0,
        }
        realized, detail = realize_expanded_path(
            p, record, g_kwh=30, charge_kw=60, reserve_kwh=15,
            soc_step=10, block_min=10,
        )
        self.assertIsNotNone(realized)
        self.assertEqual(
            detail["classification"], "deterministically_repairable"
        )
        self.assertIsNone(validate_injected_route(
            p, realized, 30, 60, 15, HORIZON_MIN
        ))

    def test_reserve_power_window_and_hash_determinism(self):
        station = "S"
        route_nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem(
            [260.0, 50.0],
            list(zip(route_nodes, route_nodes[1:])),
        )
        p.start_min[1] = 20.0
        p.end_min[1] = 25.0
        record = {
            "trips": [0, 1],
            "route_nodes": route_nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [10],
                "cet": [20],
                "kwh": [45.0],
            },
            "cost": 100050.0,
        }
        rejected, detail = realize_expanded_path(
            p, record, g_kwh=300, charge_kw=300, reserve_kwh=50,
            soc_step=15, block_min=10,
        )
        self.assertIsNone(rejected)
        self.assertIn("reserve", detail["reason"])
        rejected, detail = realize_expanded_path(
            p, record, g_kwh=300, charge_kw=30, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        self.assertIsNone(rejected)
        self.assertIn("recorded charge", detail["reason"])

        boundary = {
            "trips": [0, 1],
            "route_nodes": route_nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [10],
                "cet": [20],
                "kwh": [30.0],
            },
            "cost": 100050.0,
        }
        p.trip_energy = {0: 20.0, 1: 10.0}
        first, first_detail = realize_expanded_path(
            p, boundary, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        second, second_detail = realize_expanded_path(
            p, boundary, g_kwh=300, charge_kw=300, reserve_kwh=0,
            soc_step=15, block_min=10,
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first_detail["mapping"]["mapping_sha256"],
            second_detail["mapping"]["mapping_sha256"],
        )

    def test_strict_pool_gate_repairs_or_rejects_every_column(self):
        station = "PARX_1"
        route_nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem(
            [14.5, 0.1],
            list(zip(route_nodes, route_nodes[1:])),
        )
        p.start_min[1] = 20.0
        p.end_min[1] = 25.0
        route = {
            "trips": [0, 1],
            "route_nodes": route_nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [10],
                "cet": [20],
                "kwh": [15.0],
            },
            "cost": 100006.5,
        }
        status = {
            "csv": "instance.csv",
            "prices_csv": "prices.csv",
            "g_kwh": 300,
            "charge_kw": 300,
            "min_soc_frac": 0,
            "soc_step": 15,
            "block_min": 10,
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "instance.csv").write_text("instance\n")
            (data_dir / "prices.csv").write_text("prices\n")
            (data_dir / "Ref_dict.csv").write_text("refs\n")
            (data_dir / "par_ref_dhd.csv").write_text("deadhead\n")
            status["provenance"] = {
                "instance_sha256": hashlib.sha256(
                    (data_dir / "instance.csv").read_bytes()
                ).hexdigest(),
                "prices_sha256": hashlib.sha256(
                    (data_dir / "prices.csv").read_bytes()
                ).hexdigest(),
            }
            with (
                patch(
                    "audit_giro_known_columns.build_problem",
                    return_value=p,
                ),
                patch(
                    "utils_v2.load_station_hourly_prices",
                    return_value={"PARX": {0: 0.1}},
                ),
            ):
                prepared, audit = prepare_strict_partition_pool(
                    status, [route], data_dir=data_dir
                )
                second, second_audit = prepare_strict_partition_pool(
                    status, [route], data_dir=data_dir
                )
                tampered = json.loads(json.dumps(prepared[0]))
                tampered["continuous_realized_charging_blocks"][0][
                    "realized_kwh"
                ] += 1.0
                rejected, rejected_audit = prepare_strict_partition_pool(
                    status, [tampered], data_dir=data_dir
                )
            self.assertEqual(len(prepared), 1)
            self.assertEqual(audit["deterministically_repaired"], 1)
            self.assertEqual(audit["rejected_columns"], 0)
            self.assertEqual(
                len(audit["mip_ordered_pool_sha256"]), 64
            )
            self.assertEqual(
                audit["repaired_set_sha256"],
                second_audit["repaired_set_sha256"],
            )
            self.assertEqual(prepared, second)
            self.assertEqual(
                prepared[0]["cost"], route["cost"]
            )
            self.assertTrue(
                prepared[0]["continuous_realized_charging_blocks"]
            )
            self.assertEqual(
                len(prepared[0]["physical_realization"][
                    "continuous_realized_charging_blocks_sha256"
                ]),
                64,
            )
            self.assertLess(
                prepared[0]["continuous_realized_cost"],
                prepared[0]["expanded_grid_cost"],
            )
            self.assertEqual(rejected, [])
            self.assertEqual(rejected_audit["rejected_columns"], 1)

    def test_invalid_cheapest_duplicate_cannot_shadow_valid_column(self):
        station = "PARX_1"
        nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem([14.5, 0.1], list(zip(nodes, nodes[1:])))
        p.start_min[1] = 20.0
        p.end_min[1] = 25.0
        valid = {
            "trips": [0, 1],
            "route_nodes": nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [10],
                "cet": [20],
                "kwh": [15.0],
            },
            "cost": 100006.5,
        }
        invalid = {
            **valid,
            "route_nodes": [DEPOT, 0, "BAD", 1, DEPOT],
            "cost": 1.0,
        }
        status = {
            "csv": "instance.csv",
            "prices_csv": "prices.csv",
            "g_kwh": 300,
            "charge_kw": 300,
            "min_soc_frac": 0,
            "soc_step": 15,
            "block_min": 10,
        }
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            (data_dir / "instance.csv").write_text("instance\n")
            (data_dir / "prices.csv").write_text("prices\n")
            (data_dir / "Ref_dict.csv").write_text("refs\n")
            (data_dir / "par_ref_dhd.csv").write_text("deadhead\n")
            status["provenance"] = {
                "instance_sha256": hashlib.sha256(
                    (data_dir / "instance.csv").read_bytes()
                ).hexdigest(),
                "prices_sha256": hashlib.sha256(
                    (data_dir / "prices.csv").read_bytes()
                ).hexdigest(),
            }
            with (
                patch(
                    "audit_giro_known_columns.build_problem",
                    return_value=p,
                ),
                patch(
                    "utils_v2.load_station_hourly_prices",
                    return_value={"PARX": {0: 0.1}},
                ),
            ):
                prepared, audit = prepare_strict_partition_pool(
                    status, [invalid, valid], data_dir=data_dir
                )
            self.assertEqual(len(prepared), 1)
            self.assertEqual(prepared[0]["cost"], valid["cost"])
            self.assertEqual(audit["rejected_columns"], 1)
            self.assertEqual(audit["mip_unique_accepted_columns"], 1)

    def test_bounded_pool_audit_publishes_machine_outputs(self):
        station = "PARX_1"
        route_nodes = [DEPOT, 0, station, 1, DEPOT]
        p = problem(
            [14.5, 0.1],
            list(zip(route_nodes, route_nodes[1:])),
        )
        p.start_min[1] = 20.0
        p.end_min[1] = 25.0
        route = {
            "trips": [0, 1],
            "route_nodes": route_nodes,
            "charging_stops": {
                "stations": [station],
                "cst": [10],
                "cet": [20],
                "kwh": [15.0],
            },
            "cost": 100006.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = root / "input/cell"
            cell.mkdir(parents=True)
            data = cell / "data"
            data.mkdir()
            (data / "instance.csv").write_text("synthetic instance\n")
            (data / "prices.csv").write_text("synthetic prices\n")
            (root / "Ref_dict.csv").write_text("synthetic refs\n")
            (root / "par_ref_dhd.csv").write_text(
                "synthetic deadhead\n"
            )
            status_path = cell / "pool.snapshot.json"
            journal = Path(str(status_path) + ".columns.jsonl")
            journal.write_text(json.dumps(route) + "\n")
            status_path.write_text(json.dumps({
                "csv": "instance.csv",
                "prices_csv": "prices.csv",
                "g_kwh": 300,
                "charge_kw": 300,
                "min_soc_frac": 0,
                "soc_step": 15,
                "block_min": 10,
                "trip_ids": [0, 1],
                "columns_journal": str(journal),
                "provenance": {
                    "instance_sha256": hashlib.sha256(
                        (data / "instance.csv").read_bytes()
                    ).hexdigest(),
                    "prices_sha256": hashlib.sha256(
                        (data / "prices.csv").read_bytes()
                    ).hexdigest(),
                },
            }))
            progress = root / "progress/cell"
            progress.mkdir(parents=True)
            (progress / "latest.json").write_text(json.dumps({
                "schema": "evsp-dr-mip-convergence-v1",
                "kind": "latest",
                "incumbent": {
                    "selected_route_indices": [0],
                    "route_vector_sha256": hashlib.sha256(b"[0]").hexdigest(),
                },
                "metadata": {
                    "source_result_sha256": hashlib.sha256(
                        status_path.read_bytes()
                    ).hexdigest(),
                    "source_journal_sha256": hashlib.sha256(
                        journal.read_bytes()
                    ).hexdigest(),
                },
            }))
            output = root / "audit"
            second_output = root / "audit-second"
            with (
                patch(
                    "audit_expanded_pool_physical.build_problem",
                    return_value=p,
                ),
                patch(
                    "audit_expanded_pool_physical."
                    "load_station_hourly_prices",
                    return_value={"PARX": {0: 0.1}},
                ),
            ):
                report = audit_pools(
                    [status_path],
                    output_dir=output,
                    reference_data_dir=root,
                    campaign_root=root,
                    route_detail="selected",
                )
                audit_pools(
                    [status_path],
                    output_dir=second_output,
                    reference_data_dir=root,
                    campaign_root=root,
                    route_detail="selected",
                )
            self.assertEqual(
                report["pools"][0]["counts"],
                {"deterministically_repairable": 1},
            )
            self.assertTrue((output / "route_audit.csv").is_file())
            self.assertTrue((output / "pool_summary.csv").is_file())
            self.assertTrue((output / "ROOT_CAUSE.md").is_file())
            self.assertTrue((output / "completion.json").is_file())
            for path in output.iterdir():
                self.assertEqual(
                    path.read_bytes(),
                    (second_output / path.name).read_bytes(),
                    path.name,
                )


if __name__ == "__main__":
    unittest.main()
