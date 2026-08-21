import sys
import unittest
from dataclasses import replace
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audit_giro_known_columns import (  # noqa: E402
    DEPOT,
    HORIZON_MIN,
    STATIONS,
    build_problem,
)
from config import CHARGING_STATIONS  # noqa: E402
from event_pricer_network import EventExpandedNetwork  # noqa: E402
from fixed_duty_continuous_optimizer import (  # noqa: E402
    optimize_fixed_duty_continuous,
)
from tariff_response_core import giro_routes_for_instance  # noqa: E402
from utils_v2 import load_station_hourly_prices  # noqa: E402


def restricted_problem(problem, trips):
    trip_set = set(trips)
    station_set = set(STATIONS)
    adjacency = {}
    for source, arcs in problem.adjacency.items():
        if source != DEPOT and source not in trip_set and source not in station_set:
            continue
        retained = [
            arc for arc in arcs
            if (
                arc[0] == DEPOT
                or arc[0] in trip_set
                or arc[0] in station_set
            )
        ]
        if retained:
            adjacency[source] = retained
    return replace(
        problem,
        trips=tuple(trips),
        adjacency=adjacency,
        start_min={trip:problem.start_min[trip] for trip in trips},
        end_min={trip:problem.end_min[trip] for trip in trips},
        trip_energy={trip:problem.trip_energy[trip] for trip in trips},
    )


class EventPricerTargetGates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prices = load_station_hourly_prices(
            REPO/"data/hourly_prices_flat.csv",
            CHARGING_STATIONS,
        )

    def event_record(self, relative_path, duty_id):
        path=REPO/relative_path
        problem=build_problem(
            path.parent,path.name,
            max_station_to_trip_wait_min=HORIZON_MIN,
            reference_data_dir=REPO/"data",
        )
        route=next(
            route for route in giro_routes_for_instance(
                REPO/"data/Par_VehicleDetails_Updated.csv",path
            )
            if route["duty_id"]==duty_id
        )
        restricted=restricted_problem(problem,route["trips"])
        network=EventExpandedNetwork(
            restricted,self.prices,
            soc_step=2.5,block_min=5,
            g_kwh=240.0,charge_kw=240.0,reserve_kwh=0.0,
        )
        record=network.fixed_sequence_record(route["trips"])
        continuous=optimize_fixed_duty_continuous(
            restricted,route["trips"],self.prices
        )
        self.assertTrue(continuous["feasible"])
        self.assertIsNotNone(record)
        self.assertEqual(record["trips"],route["trips"])
        return record

    def test_k02_s2_duty_13413_transition_14_to_16_is_representable(self):
        record=self.event_record(
            "data/scale_ladder/instances/"
            "Practice_Custom_DutyUnion_k02_r2.csv",
            "13413",
        )
        self.assertIn((14,16),list(zip(record["trips"],record["trips"][1:])))

    def test_duty_13411_all_previously_failed_transitions_are_representable(self):
        record=self.event_record(
            "data/scale_ladder/instances/"
            "Practice_Custom_DutyUnion_k05_r1.csv",
            "13411",
        )
        transitions=set(zip(record["trips"],record["trips"][1:]))
        for transition in ((46,53),(46,53),(53,59),(53,59),(73,77)):
            self.assertIn(transition,transitions)


if __name__=="__main__":
    unittest.main()
