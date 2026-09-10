import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "terminal_energy_fair_pilot",
    ROOT / "scripts/event_uniform_envelope/terminal_energy_fair_pilot.py",
)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


def route(trip, terminal_kwh, variable_cost):
    return {
        "trips": [trip], "route_nodes": ["DEPOT", trip, "DEPOT"],
        "charging_stops": {"stations": [], "cst": [], "cet": [], "kwh": []},
        "expanded_grid_charging_stops": {
            "stations": [], "cst": [], "cet": [], "kwh": [],
        },
        "cost": PILOT.BUS_COST + variable_cost,
        "continuous_realized_cost": PILOT.BUS_COST + variable_cost,
        "continuous_realization": {
            "expanded_grid_terminal_soc_kwh": terminal_kwh,
            "continuous_terminal_soc_kwh": terminal_kwh,
        },
    }


class TerminalEnergyFairPilotTests(unittest.TestCase):
    def test_replay_reconstructs_legacy_terminal_metadata(self):
        legacy = route(0, 0.0, 0.0)
        legacy.pop("continuous_realization")
        with self.assertRaisesRegex(ValueError, "missing grid terminal"):
            PILOT.terminal(legacy, "grid")
        mapping = {
            "expanded_grid_terminal_soc_kwh": 12.5,
            "continuous_terminal_soc_kwh": 14.25,
        }
        replayed = dict(legacy)
        replayed["continuous_realization"] = dict(mapping)
        costs = {
            "recomputed_expanded_grid_cost": legacy["cost"],
            "continuous_realized_cost": legacy["cost"],
            "continuous_realized_charging_blocks": [],
        }
        with patch(
            "expanded_path_realization.realize_expanded_path",
            return_value=(replayed, {"mapping": mapping}),
        ), patch(
            "expanded_path_realization.realized_costs",
            return_value=costs,
        ), patch(
            "run_exact_pool_mip.validate_injected_route",
            return_value=None,
        ):
            result = PILOT.replay_pool_routes(object(), [legacy], {})

        self.assertEqual(len(result), 1)
        normalized = result[0]
        self.assertEqual(
            PILOT.terminal(normalized, "grid"),
            mapping["expanded_grid_terminal_soc_kwh"],
        )
        self.assertEqual(
            PILOT.terminal(normalized, "continuous"),
            mapping["continuous_terminal_soc_kwh"],
        )
        self.assertEqual(normalized["continuous_realized_cost"], legacy["cost"])
        self.assertEqual(normalized["continuous_realized_charging_blocks"], [])

    def test_fixed_and_joint_use_same_aggregate_terminal_row(self):
        frontiers = []
        for duty in range(5):
            frontiers.append({
                "duty_id": duty,
                "routes": [route(duty, 0.0, 0.0), route(duty, 60.0, 6.0)],
            })
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixed, _fixed_detail = PILOT.solve_fixed(
                frontiers, PILOT.TARGET_KWH, directory / "fixed.log",
            )
            self.assertEqual(len(fixed), 5)
            self.assertGreaterEqual(
                PILOT.summarize_routes(fixed)[
                    "expanded_grid_terminal_energy_kwh"
                ],
                PILOT.TARGET_KWH,
            )
            pool = PILOT.pareto_pool(
                [value for duty in frontiers for value in duty["routes"]]
            )
            joint, detail = PILOT.solve_joint(
                pool, list(range(5)), fixed, PILOT.TARGET_KWH,
                directory / "joint.log", total_seconds=20,
                stage1_seconds=10,
            )
            audit = PILOT.summarize_routes(joint)
            self.assertEqual(audit["fleet"], 5)
            self.assertGreaterEqual(
                audit["expanded_grid_terminal_energy_kwh"],
                PILOT.TARGET_KWH,
            )
            self.assertLessEqual(audit["fleet"], detail["stage1"]["buses"])
            self.assertTrue(detail["stage1"]["incumbent_validated"])

    def test_pareto_pool_keeps_cost_terminal_tradeoff(self):
        values = [
            route(1, 0.0, 1.0), route(1, 60.0, 3.0),
            route(1, 50.0, 4.0), route(1, 60.0, 5.0),
        ]
        retained = PILOT.pareto_pool(values)
        observed = sorted(
            (PILOT.terminal(value), PILOT.variable_cost(value))
            for value in retained
        )
        self.assertEqual(observed, [(0.0, 1.0), (60.0, 3.0)])

    def test_validated_fixed_start_survives_pareto_dominance(self):
        fixed = route(1, 50.0, 5.0)
        dominating = route(1, 60.0, 3.0)
        pareto = PILOT.pareto_pool([fixed, dominating])
        self.assertNotIn(PILOT.route_signature(fixed), {
            PILOT.route_signature(value) for value in pareto
        })
        preserved = PILOT.preserve_start_routes(pareto, [fixed])
        self.assertIn(PILOT.route_signature(fixed), {
            PILOT.route_signature(value) for value in preserved
        })


if __name__ == "__main__":
    unittest.main()
