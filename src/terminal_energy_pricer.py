"""Experimental exact event pricing with an aggregate return-energy dual.

Terminal alternatives are retained BEFORE the base graph collapses parallel
sink arcs. Internal arcs can still use the base graph's state/cost dominance.
No old graph cache is compatible with this class.
"""
import math

from event_pricer_network import EventExpandedNetwork
from config import BUS_COST_KX


class TerminalEnergyNetwork(EventExpandedNetwork):
    def __init__(self, *args, **kwargs):
        self.terminal_options = {}
        super().__init__(*args, **kwargs)

    def _add(self, source, target, cost, trip, action):
        if target == self.SINK:
            _, last_trip, level = self.node_meta[source]
            energy = (
                self.grid[action['exit_level']] - action['outbound_kwh']
                if action['kind'] == 'charge' else
                self.grid[level] - self.problem.trip_energy[last_trip]
                - action['deadhead_kwh']
            )
            options = self.terminal_options.setdefault(source, [])
            # For a common prefix, only cost/return-energy dominance is safe.
            if not any(c <= cost and e >= energy for c, e, a in options):
                options[:] = [(c, e, a) for c, e, a in options
                              if not (cost <= c and energy >= e)]
                options.append((float(cost), float(energy), dict(action)))
        super()._add(source, target, cost, trip, action)

    def terminal_batch(self, trip_duals, *, terminal_dual=0.0,
                       route_dual=0.0, objective='combined-cost', limit=30):
        if terminal_dual < -1e-8:
            raise ValueError('Return-energy >= row requires nonnegative dual')
        base = self.min_reduced_cost_route(
            trip_duals, route_dual=route_dual, objective=objective)
        if base is None:
            return []
        values, parent = base['_value'], base['_parent']
        candidates = []
        for source, options in self.terminal_options.items():
            if not math.isfinite(values[source]):
                continue
            for cost, energy, action in options:
                sink_cost = cost if objective in {'combined-cost', 'charging-cost'} else 0.0
                rc = float(values[source]) + sink_cost - terminal_dual * energy
                candidates.append((rc, source, energy, action))
        candidates.sort(key=lambda row: (row[0], row[1], row[2]))
        result, seen = [], set()
        for rc, source, energy, action in candidates:
            if result and (rc >= -1e-9 or len(result) >= limit):
                break
            actions = [action]
            node = source
            while node != 0:
                if self.arc_mode == 'explicit':
                    previous, edge = parent[node]
                else:
                    previous = int(parent[node])
                    edge = self._edge_action(previous, node)
                actions.append(edge)
                node = previous
            record = self._record(list(reversed(actions)))
            record['terminal_energy_kwh'] = energy
            key = (frozenset(record['trips']), energy, record['cost'])
            if key in seen:
                continue
            seen.add(key)
            objective_cost = (record['cost'] if objective == 'combined-cost'
                              else record['cost'] - BUS_COST_KX if objective == 'charging-cost'
                              else 1.0 if objective == 'fleet-only' else 0.0)
            recomputed = (objective_cost - sum(trip_duals.get(t, 0) for t in record['trips'])
                          - terminal_dual * energy - route_dual)
            if not math.isclose(rc, recomputed, abs_tol=2e-5, rel_tol=1e-10):
                raise RuntimeError(f'Pricing cost mismatch: {rc} != {recomputed}')
            result.append({'rc': rc, 'record': record})
        return result
