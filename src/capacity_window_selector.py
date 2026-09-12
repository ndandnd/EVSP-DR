"""Optional exact capacity-window prefix and iteration memo accelerator."""
import math
import sys
import time
from bisect import bisect_left
import event_pricer_network as ep
from utils_v2 import base_station_name

class CapacityWindowSelector:
    """One pricing-call snapshot; caches never survive dual/config changes.

    Bounded option storage and memo entries avoid growth with real-network keys.
    Prefix subtraction uses exact source row sums for near ties and the winner.
    """
    MAX_OPTIONS = 100_000
    MAX_MEMO = 4096
    MAX_PREFIX_ROWS = 1_000_000

    def __init__(self, network, duals, sites, grid, *, deadline=None, clock=time.perf_counter):
        self.events = network.events
        self.prices = network.prices
        self.station_charge_kw = network.station_charge_kw
        self.charge_kw = network.charge_kw
        self._charge_power = network._charge_power
        self.begin_iteration(duals, sites, grid, deadline=deadline, clock=clock)

    def begin_iteration(self, duals, sites, grid, *, deadline=None, clock=time.perf_counter):
        ep._check_pricing_deadline(deadline, clock)
        self._capacity_options = {}
        self._option_count = 0
        self._capacity_memo = {}
        self._capacity_duals = dict(duals)
        self._capacity_sites = None if sites is None else frozenset(sites)
        self._capacity_grid = int(grid)
        if self._capacity_grid <= 0:
            raise ValueError('capacity grid must be positive')
        self._capacity_prefix = {}
        self.stats = dict(calls=0, charge_calls=0, hits=0, options=0, interval_queries=0, tie_fallbacks=0, options_hits=0)
        by_station = {}
        for (station, minute), value in duals.items():
            ep._check_pricing_deadline(deadline, clock)
            if minute % self._capacity_grid == 0:
                by_station.setdefault(station, {})[int(minute // self._capacity_grid)] = float(value)
        prefix_rows = 0
        self._sparse = {}
        for station, rows in by_station.items():
            lower, upper = min(rows), max(rows) + 1
            if upper - lower > 100_000 or prefix_rows + upper - lower > self.MAX_PREFIX_ROWS:
                # Huge sparse row spans must not allocate a dense horizon.
                indices = sorted(rows)
                prefix = [0.0]
                for offset, index in enumerate(indices):
                    if offset % 256 == 0:
                        ep._check_pricing_deadline(deadline, clock)
                    prefix.append(prefix[-1] + rows[index])
                self._sparse[station] = indices, prefix
                continue
            prefix_rows += upper - lower
            prefix = [0.0]
            for i in range(lower, upper):
                if i % 256 == 0:
                    ep._check_pricing_deadline(deadline, clock)
                prefix.append(prefix[-1] + rows.get(i, 0.0))
            self._capacity_prefix[station] = lower, upper, prefix
        # Conservative floating error envelope for both prefix cancellation and
        # the original arbitrary frozenset sum; near ties use the exact oracle.
        total_abs = sum(abs(float(v)) for v in duals.values())
        self._capacity_error = 64 * sys.float_info.epsilon * max(1, len(duals)) * max(1., total_abs)

    def interval(self, station, start, end):
        self.stats['interval_queries'] += 1
        if self._capacity_sites is not None and station not in self._capacity_sites:
            return 0.0
        info = self._capacity_prefix.get(station)
        g = self._capacity_grid
        first, last = math.floor(start/g), math.ceil(end/g)
        # Exactly the source predicates, including TOL near both boundaries.
        while first < last and (first+1)*g <= start+ep.TOL:
            first += 1
        while last > first and (last-1)*g >= end-ep.TOL:
            last -= 1
        if info is None:
            indices, prefix = self._sparse.get(station, ((), (0.0,)))
            left, right = bisect_left(indices, first), bisect_left(indices, last)
            return prefix[right] - prefix[left]
        lower, upper, prefix = info
        first, last = max(lower, min(upper, first)), max(lower, min(upper, last))
        return prefix[last-lower]-prefix[first-lower] if last > first else 0.0

    def _options(self, action, deadline, clock):
        station = action['station']
        key = (station, float(action['arrival_min']), float(action['deadline_min']), float(action['kwh']), self._charge_power(station), self._capacity_grid)
        if key in self._capacity_options:
            self.stats['options_hits'] += 1
            return key, self._capacity_options[key]
        arrival, end, energy, power, grid = key[1:]
        duration = energy*60.0/power
        latest = end-duration
        if latest < arrival - ep.TOL:
            return key, ()
        candidates = {arrival, latest}
        for event in self.events[station]:
            ep._check_pricing_deadline(deadline, clock)
            if abs(event/60-round(event/60)) <= ep.TOL:
                candidates.add(float(event))
                candidates.add(float(event)-duration)
        for index in range(math.floor(arrival/grid), math.ceil(end/grid)+1):
            if index % 256 == 0:
                ep._check_pricing_deadline(deadline, clock)
            candidates.add(float(index*grid))
            candidates.add(float(index*grid)-duration)
        options = []
        for start in sorted(candidates):
            ep._check_pricing_deadline(deadline, clock)
            if start >= arrival-ep.TOL and start <= latest+ep.TOL:
                options.append((ep._window_cost(station, start, duration, energy, self.prices, power), start, start+duration))
        # Same source ordering; tie fallback therefore observes the same oracle.
        result = tuple(sorted(options))
        if self._option_count + len(result) <= self.MAX_OPTIONS:
            self._capacity_options[key] = result
            self._option_count += len(result)
        return key, result

    def adjust(self, cost, action, capacity_duals, capacity_sites, capacity_grid_min, *, deadline=None, clock=time.perf_counter):
        ep._check_pricing_deadline(deadline, clock)
        self.stats['calls'] += 1
        if action.get('kind') != 'charge' or not self._capacity_duals:
            return float(cost), action
        self.stats['charge_calls'] += 1
        key = (action['station'], float(action['arrival_min']), float(action['deadline_min']), float(action['kwh']), self._charge_power(action['station']), self._capacity_grid)
        if key in self._capacity_memo:
            self.stats['hits'] += 1
            adjusted, start, end = self._capacity_memo[key]
            return adjusted, {**action, 'cst': start, 'cet': end}
        key, options = self._options(action, deadline, clock)
        best = None
        station = base_station_name(action['station'])
        def exact(option):
            energy_cost, start, end = option
            rows = ep.conservative_capacity_rows({**action, 'cst':start, 'cet':end}, sites=self._capacity_sites, grid_min=self._capacity_grid)
            return (ep.CHARGE_START_COST+energy_cost-sum(float(self._capacity_duals.get(r,0.)) for r in rows), start, end)
        best_option = None
        for option in options:
            ep._check_pricing_deadline(deadline, clock)
            self.stats['options'] += 1
            energy_cost, start, end = option
            candidate = (ep.CHARGE_START_COST+energy_cost-self.interval(station,start,end),start,end)
            if best is not None and abs(candidate[0]-best[0]) <= self._capacity_error:
                self.stats['tie_fallbacks'] += 1
                candidate, best = exact(option), exact(best_option)
            if best is None or candidate < best:
                best, best_option = candidate, option
        if best is None:
            raise RuntimeError('stored event charge arc has no feasible window')
        best = exact(best_option) # returned cost exactly uses original row sum
        if len(self._capacity_memo) < self.MAX_MEMO:
            self._capacity_memo[key] = best
        adjusted, start, end = best
        return adjusted, {**action,'cst':start,'cet':end}

