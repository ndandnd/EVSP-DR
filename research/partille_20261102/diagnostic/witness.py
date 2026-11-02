"""Fixed-GIRO-sequence diagnostic on the original v15 graph physics, rechoosing stations and times."""
import argparse
import json
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

from run_capacity_terminal_cg import CapacityTerminalNetwork, DEFAULT_PLUGS, audit_energy, problem_kwargs
from audit_giro_known_columns import build_problem
from config import CHARGING_STATIONS
from run_fixed_duty_terminal import giro_sequences
from utils_v2 import load_station_hourly_prices
from capacity_event_pricer import record_occupancy


class FixedSequence(CapacityTerminalNetwork):
    # Retain full-instance transfer tables and event times. Only the service successor and depot
    # endpoints are restricted. This gives a subset of full-network paths under the same physics.
    def __init__(self, problem, prices, trips):
        self.following = dict(zip(trips, [*trips[1:], None]))
        self.first = trips[0]
        super().__init__(replace(problem, trips=tuple(trips)), prices, plugs=DEFAULT_PLUGS,
                         soc_step=2.5, block_min=5, g_kwh=240, charge_kw=240, station_kw={'PARX': 60},
                         reserve_kwh=0, strict_tariff_coverage=False)

    def _source_candidates(self):
        for edge in super()._source_candidates():
            if edge[2] == self.first:
                yield edge

    def _direct_candidates(self, trip, soc_exit, depart):
        for edge in super()._direct_candidates(trip, soc_exit, depart):
            if edge[2] == self.following[trip]:
                yield edge

    def _charge_candidates(self, trip, soc_exit, depart):
        successor = self.following[trip]
        outgoing, depot = self.station_trip, self.station_depot
        self.station_trip = {station: {successor: trips[successor]} for station, trips in outgoing.items()
                             if successor in trips}
        self.station_depot = depot if successor is None else {}
        try:
            yield from super()._charge_candidates(trip, soc_exit, depart)
        finally:
            self.station_trip, self.station_depot = outgoing, depot


def search(data, instance, prices, out):
    problem = build_problem(data, instance, **problem_kwargs())
    duties = giro_sequences(data, instance, problem)
    curves = load_station_hourly_prices(prices, CHARGING_STATIONS)
    nets = []
    out.mkdir(parents=True, exist_ok=False)
    for duty in duties:
        net = FixedSequence(problem, curves, duty['trips'])
        nets.append(net)
        print('graph built', len(nets), net.n_arcs, flush=True)
    randomizer = random.Random(0)
    order = list(range(len(nets)))
    seen = set()
    for attempt in range(150):
        occupancy = Counter()
        selection = []
        if attempt:
            randomizer.shuffle(order)
        for i in order:
            # Price exactly on this duty's subgraph. The penalty steers toward currently free
            # charger-minutes. Accept a new duty route only if its physical occupancy actually fits.
            penalty = {key: 10000.0 for key, count in occupancy.items()
                       if DEFAULT_PLUGS[key[0]] is not None and count >= DEFAULT_PLUGS[key[0]]}
            rows = nets[i].capacity_terminal_batch({}, penalty, limit=1)
            if not rows:
                print('fixed sequence has no reachable terminal', i, flush=True)
                return False
            record = audit_energy(nets[i], rows[0]['record'])
            assert record['trips'] == duties[i]['trips']
            record['duty_id'] = duties[i]['duty_id']
            occ = record_occupancy(record)
            record['capacity_occupancy'] = sorted([s, m, count] for (s, m), count in occ.items()
                                                  if DEFAULT_PLUGS[s] is not None)
            key = json.dumps([record['trips'], record['expanded_grid_charging_stops']], sort_keys=True)
            if key not in seen:
                with (out / 'candidates.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                seen.add(key)
            occupancy.update(occ)
            if any(DEFAULT_PLUGS[s] is not None and count > DEFAULT_PLUGS[s]
                   for (s, _), count in occupancy.items()):
                break
            selection.append(record)
        print('ordering', attempt, 'compatible', len(selection), '/', len(nets), flush=True)
        if len(selection) == len(nets):
            (out / 'selected_routes.json').write_text(json.dumps(selection, indent=2) + '\n')
            return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--prices', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--start-fee', type=float, required=True, choices=[5])
    args = parser.parse_args()
    raise SystemExit(0 if search(args.data, args.instance, args.prices, args.out) else 3)
