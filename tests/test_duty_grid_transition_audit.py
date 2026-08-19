import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_fixed_duty_grid_transitions import (  # noqa: E402
    MODES,
    _classify_counterfactuals,
    audit_duty,
    evaluate_counterfactual_transition,
    publish as publish_oracle,
    validate as validate_oracle,
)
import audit_fixed_duty_grid_transitions as transition_oracle  # noqa: E402
from audit_giro_known_columns import HORIZON_MIN, STATIONS, build_problem  # noqa: E402
from audit_scale_ladder_known_membership import _prices  # noqa: E402
from build_scale_ladder_membership_v2 import (  # noqa: E402
    DEFAULT_OUTPUT as V2_OUTPUT,
    publish as publish_v2,
    validate as validate_v2,
)
import build_scale_ladder_membership_v2 as membership_v2  # noqa: E402
from fixed_duty_expanded_optimizer import (  # noqa: E402
    _arc_groups,
    evaluate_fixed_duty_transition,
    optimize_fixed_duty,
)
from tariff_response_core import giro_routes_for_instance  # noqa: E402
from utils_v2 import base_station_name  # noqa: E402


K5 = (
    REPO_ROOT
    / "data/scale_ladder/instances/Practice_Custom_DutyUnion_k05_r1.csv"
)
K5_SHA = "fc10ac0707becb960364e76b8c1e1c414d5d5639cbc3b7dadaf67a77e03f5322"
K40 = (
    REPO_ROOT
    / "data/tariff_response/frozen_instances/"
    "Practice_Custom_DutyUnion_k40_r2.csv"
)
K40_SHA = "3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd"
EXPECTED_TRANSITIONS = (
    ((15.0, 10), (46, 53), (106, 119)),
    ((5.0, 10), (46, 53), (106, 119)),
    ((2.5, 10), (53, 59), (119, 132)),
    ((1.0, 10), (53, 59), (119, 132)),
    ((1.0, 5), (73, 77), (158, 167)),
)


def _problem(*, deadline=20.0, successor_energy=20.0, adjacency=None):
    return SimpleNamespace(
        trips=(0, 1),
        trip_energy={0: 0.0, 1: float(successor_energy)},
        start_min={0: 0.0, 1: float(deadline)},
        end_min={0: 0.0, 1: float(deadline) + 1.0},
        adjacency=adjacency or {},
    )


def _adjacency(*, direct=False, station_options=()):
    rows = {0: [], 1: []}
    if direct:
        rows[0].append((1, 0.0, 0.0, "trip_trip"))
    for station, inbound_time, inbound_kwh, outbound_time, (
        outbound_kwh
    ) in station_options:
        rows[0].append(
            (station, inbound_time, inbound_kwh, "trip_station")
        )
        rows.setdefault(station, []).append(
            (1, outbound_time, outbound_kwh, "station_trip")
        )
    return rows


class DutyGridTransitionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = build_problem(
            K5.parent,
            K5.name,
            max_station_to_trip_wait_min=HORIZON_MIN,
            reference_data_dir=REPO_ROOT / "data",
        )
        cls.routes = giro_routes_for_instance(
            REPO_ROOT / "data/Par_VehicleDetails_Updated.csv", K5
        )
        cls.prices = _prices()

    def test_actual_duty_13411_all_five_grid_outcomes(self):
        payload, candidates, frontiers, counterfactuals = audit_duty(
            instance_path=K5,
            expected_instance_sha256=K5_SHA,
            duty_id="13411",
            grids=[item[0] for item in EXPECTED_TRANSITIONS],
            cell_id="k05_s1",
            comparison_instance=K40,
            comparison_instance_sha256=K40_SHA,
        )
        observed = tuple((
            (row["soc_step"], row["block_min"]),
            tuple(row["failed_local_transition"]),
            tuple(row["failed_ordered_transition"]),
        ) for row in payload["grid_results"])
        self.assertEqual(observed, EXPECTED_TRANSITIONS)
        self.assertTrue(all(
            row["certificate_certified"] is False
            and row["feasible"] is False
            for row in payload["grid_results"]
        ))
        self.assertTrue(
            payload["comparison_instance"]["same_ordered_duty"]
        )
        self.assertEqual(
            payload["ordered_trip_sequence"],
            payload["comparison_instance"]["ordered_trip_sequence"],
        )
        self.assertEqual(
            payload["continuous_witness"]["physical_validation"],
            "validated_injected_route",
        )
        self.assertTrue(candidates)
        self.assertTrue(frontiers)
        self.assertEqual(len(counterfactuals), 20)
        self.assertEqual(
            [row["cause_classification"] for row in payload["grid_results"]],
            [
                "interaction", "interaction",
                "unresolved", "unresolved", "unresolved",
            ],
        )

    def test_trace_disabled_preserves_optimizer_semantics(self):
        for duty_id in ("13411", "13412"):
            route = next(
                row for row in self.routes
                if str(row["duty_id"]) == duty_id
            )
            arguments = {
                "g_kwh": 300.0,
                "charge_kw": 300.0,
                "reserve_kwh": 0.0,
                "soc_step": 15.0,
                "block_min": 10,
                "tariff_id": "historical_flat",
                "tariff_sha256":
                    "1f51f2e1f6ca303838ebaaf6272a28ff2d6bbee97146cb04d330e10f191f8200",
                "instance_sha256": K5_SHA,
                "allow_diagnostic_grid": True,
            }
            plain = optimize_fixed_duty(
                self.problem,
                route["trips"],
                self.prices,
                trace=False,
                **arguments,
            )
            traced = optimize_fixed_duty(
                self.problem,
                route["trips"],
                self.prices,
                trace=True,
                **arguments,
            )

            def normalized(value):
                value = copy.deepcopy(value)
                value.pop("runtime_s", None)
                value.pop("diagnostic_trace", None)
                certificate = value.get("certificate") or {}
                certificate.pop("implementation_sha256", None)
                certificate.pop("certificate_sha256", None)
                route_value = value.get("route") or {}
                route_value.pop("fixed_duty_certificate_sha256", None)
                return value

            self.assertEqual(normalized(plain), normalized(traced))
            if plain["feasible"]:
                self.assertEqual(
                    plain["route"]["route_nodes"],
                    traced["route"]["route_nodes"],
                )
                self.assertEqual(
                    plain["expanded_grid_objective"],
                    traced["expanded_grid_objective"],
                )
                self.assertEqual(
                    plain["physical_replay_status"], "validated"
                )

    def test_counterfactual_cause_classes_and_missing_arc(self):
        station = STATIONS[0]
        prices = {
            base_station_name(value): {
                hour: 1.0 for hour in range(27)
            }
            for value in STATIONS
        }
        cases = [
            (
                "block alignment",
                _problem(
                    deadline=4.0,
                    successor_energy=20.0,
                    adjacency=_adjacency(
                        direct=True,
                        station_options=((station, 0.0, 0.0, 0.0, 0.0),),
                    ),
                ),
                10.0,
                10.0,
            ),
            (
                "accumulated SOC flooring",
                _problem(
                    deadline=20.0,
                    successor_energy=20.0,
                    adjacency=_adjacency(direct=True),
                ),
                5.0,
                25.0,
            ),
            (
                "interaction",
                _problem(
                    deadline=4.0,
                    successor_energy=20.0,
                    adjacency=_adjacency(
                        direct=True,
                        station_options=((station, 0.0, 0.0, 0.0, 0.0),),
                    ),
                ),
                5.0,
                25.0,
            ),
        ]
        for expected, problem, production_soc, continuous_soc in cases:
            arcs = _arc_groups(problem)
            production_candidates, _trace = (
                evaluate_fixed_duty_transition(
                    problem,
                    arcs,
                    trip=0,
                    successor=1,
                    final_gap=False,
                    level=int(production_soc),
                    base_cost=100000.0,
                    actions=(),
                    grid=[float(index) for index in range(301)],
                    soc_step=1.0,
                    block_min=5,
                    g_kwh=300.0,
                    charge_kw=300.0,
                    reserve_kwh=0.0,
                    station_prices=prices,
                    n_blocks=24 * 60 // 5,
                    include_trace=True,
                )
            )
            by_mode = {
                mode: (
                    {"feasible": bool(production_candidates)}
                    if mode == MODES[0]
                    else evaluate_counterfactual_transition(
                        problem,
                        arcs,
                        trip=0,
                        successor=1,
                        soc_step=1.0,
                        block_min=5,
                        production_soc_after_trip=production_soc,
                        no_floor_prefix_soc_after_trip=continuous_soc,
                        mode=mode,
                    )
                )
                for mode in MODES
            }
            self.assertEqual(
                _classify_counterfactuals(by_mode, True), expected
            )
        self.assertEqual(
            _classify_counterfactuals(
                {mode: {"feasible": False} for mode in MODES},
                False,
            ),
            "graph/reference defect",
        )
        self.assertTrue(all(prices.values()))

    def test_production_trace_has_alternative_station_and_delayed_charge(self):
        bad_station, good_station = STATIONS[:2]
        problem = _problem(
            deadline=30.0,
            successor_energy=20.0,
            adjacency=_adjacency(
                station_options=(
                    (bad_station, 0.0, 0.0, 29.0, 0.0),
                    (good_station, 0.0, 0.0, 0.0, 0.0),
                )
            ),
        )
        arcs = _arc_groups(problem)
        prices = {
            base_station_name(value): {
                hour: 1.0 for hour in range(27)
            }
            for value in STATIONS
        }
        candidates, trace = evaluate_fixed_duty_transition(
            problem,
            arcs,
            trip=0,
            successor=1,
            final_gap=False,
            level=1,
            base_cost=100000.0,
            actions=(),
            grid=[float(index) for index in range(301)],
            soc_step=1.0,
            block_min=5,
            g_kwh=300.0,
            charge_kw=300.0,
            reserve_kwh=0.0,
            station_prices=prices,
            n_blocks=24 * 60 // 5,
            include_trace=True,
        )
        self.assertTrue(any(
            row["station"] == bad_station
            and row["accepted"] is False
            for row in trace
        ))
        self.assertTrue(any(
            candidate["action"].get("station") == good_station
            for candidate in candidates
        ))
        self.assertTrue(any(
            row.get("station") == good_station
            and row.get("delayed_charging") is True
            and row.get("accepted") is True
            for row in trace
        ))

    def test_oracle_no_clobber_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "oracle"
            arguments = dict(
                instance_path=K5,
                expected_instance_sha256=K5_SHA,
                duty_id="13411",
                grids=[item[0] for item in EXPECTED_TRANSITIONS],
                cell_id="k05_s1",
                comparison_instance=K40,
                comparison_instance_sha256=K40_SHA,
            )
            expected = transition_oracle.audit_duty(**arguments)

            def expected_copy(**_kwargs):
                return copy.deepcopy(expected)

            with patch.object(
                transition_oracle,
                "audit_duty",
                side_effect=expected_copy,
            ):
                publish_oracle(
                    output,
                    **arguments,
                )
                validate_oracle(output)
                for name in (
                    "oracle.json", "transition_candidates.csv",
                    "frontier_states.csv", "counterfactuals.csv",
                    "README.md",
                ):
                    self.assertEqual(
                        (output / name).read_bytes(),
                        (
                            REPO_ROOT
                            / "analysis/"
                            "duty_13411_grid_transition_oracle_20260819"
                            / name
                        ).read_bytes(),
                        name,
                    )
                with self.assertRaises(FileExistsError):
                    publish_oracle(output, **arguments)
                oracle_path = output / "oracle.json"
                original_oracle = oracle_path.read_bytes()
                changed = json.loads(oracle_path.read_text())
                changed["grid_results"][0][
                    "cause_classification"
                ] = "unresolved"
                oracle_path.write_text(json.dumps(
                    changed, indent=2, sort_keys=True
                ) + "\n")
                with self.assertRaisesRegex(ValueError, "summary"):
                    validate_oracle(output)
                oracle_path.write_bytes(original_oracle)
                readme = output / "README.md"
                readme.write_text(readme.read_text() + "tampered\n")
                with self.assertRaisesRegex(ValueError, "README"):
                    validate_oracle(output)

    def test_v2_tracked_artifacts_validate_and_are_deterministic(self):
        expected = membership_v2.build_payload()

        def expected_copy():
            return copy.deepcopy(expected)

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "membership-v2"
            with patch.object(
                membership_v2,
                "build_payload",
                side_effect=expected_copy,
            ):
                payload = validate_v2(V2_OUTPUT)
                self.assertTrue(payload["v1_parity_verified"])
                publish_v2(output)
                for name in (
                    "membership_summary.json",
                    "duty_summary.csv",
                    "duty_grid_outcome_long.csv",
                    "README.md",
                ):
                    self.assertEqual(
                        (output / name).read_bytes(),
                        (V2_OUTPUT / name).read_bytes(),
                        name,
                    )
                with self.assertRaises(FileExistsError):
                    publish_v2(output)
                grid_path = output / "duty_grid_outcome_long.csv"
                with grid_path.open(newline="") as handle:
                    first = next(csv.DictReader(handle))
                mapping = json.loads(
                    first["local_to_ordered_trip_mapping_json"]
                )
                self.assertEqual(
                    int(first["trip_count"]), len(mapping)
                )
                self.assertGreaterEqual(
                    int(first["instance_trip_count"]),
                    int(first["trip_count"]),
                )
                self.assertEqual(
                    first["schema"],
                    "evsp-dr-scale-ladder-duty-grid-outcome-v2",
                )
                self.assertEqual(
                    first["trip_identity_schema"],
                    "evsp-dr-trip-identity-v1",
                )
                original_grid = grid_path.read_bytes()
                with grid_path.open("a") as handle:
                    handle.write("tampered\n")
                with self.assertRaisesRegex(ValueError, "table"):
                    validate_v2(output)
                grid_path.write_bytes(original_grid)
                summary = output / "membership_summary.json"
                original_summary = summary.read_bytes()
                changed = json.loads(summary.read_text())
                changed["tariff_sha256"] = "0" * 64
                summary.write_text(json.dumps(
                    changed, indent=2, sort_keys=True
                ) + "\n")
                with self.assertRaisesRegex(ValueError, "summary"):
                    validate_v2(output)
                summary.write_bytes(original_summary)
                readme = output / "README.md"
                readme.write_text(readme.read_text() + "tampered\n")
                with self.assertRaisesRegex(ValueError, "README"):
                    validate_v2(output)
            substituted = Path(tmp) / "substituted-v1.json"
            substituted.write_bytes(
                membership_v2.V1_PATH.read_bytes() + b" "
            )
            with self.assertRaisesRegex(ValueError, "fixed v1"):
                membership_v2.build_payload(v1_path=substituted)


if __name__ == "__main__":
    unittest.main()
