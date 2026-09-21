"""Standalone diving-with-pricing pilot (experimental; not production CG).

Question under test: can integer-complementary routes be generated *without*
any sequential/GIRO witness solution?  The witness study
(``outputs/independent_review_20260916/advisor_witness_columns_20260917``)
showed the missing routes are LP-suboptimal under the fresh pool's own
certified duals (reduced cost up to 53), so reduced-cost enrichment cannot
find them.  Fixing part of the solution changes the duals, which is the
mechanism this pilot exercises:

    solve a fractional restricted master over the frozen fresh pool
    -> tentatively fix one promising fractional route (lb = ub = 1)
    -> reprice with the resulting duals, adding whatever the pricer returns
    -> repeat; backtrack or restart deterministically on failure.

Scope and safety rules encoded here:

* **Never a global claim.**  Every LP solved is a *restricted* LP over a
  *dive node*.  A node retaining artificials is recorded as ``node_artificials_remain``.
  This is a heuristic backtrack, not an infeasibility proof: no sufficient
  big-M penalty bound has been established.
  The manifest carries ``global_certificate: null`` unconditionally.
* **Original columns are preserved.**  The source status/journal are opened
  read-only, hashed on entry and re-hashed on exit; the augmented journal
  starts with a byte-for-byte copy of the original.
* **No witness input.**  The loader refuses any status carrying witness,
  warm-start or inherited-pool provenance, and refuses input paths that look
  like warm/witness artifacts.
* **Exact physical replay.**  Every appended route is replayed through
  ``run_exact_pool_mip.validate_injected_route`` before it is written.
* **Fleet cap and fixings in the reduced cost.**  The master carries a
  ``sum x <= fleet_cap`` row; its dual ``mu <= 0`` is subtracted from every
  reduced cost.  See :class:`EventPricer` for why ``mu`` is applied here and
  not passed into the pricer's ``route_dual`` argument.

The final pool MIP is *not* run here: the augmented ``cg.json`` plus journal
are written in exactly the layout ``src/run_exact_pool_mip.py`` consumes, and
the campaign scripts invoke that trusted runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

DIVE_SCHEMA = "evsp-dr-diving-pricing-pilot-v1"
DEFAULT_RC_EPS = 1e-4
INTEGRALITY_TOL = 1e-6
ARTIFICIAL_TOL = 1e-6

# Provenance keys that mark a pool as warm-started, witness-augmented or
# otherwise derived from a known solution.  Any of them present and truthy is
# a hard refusal.
FORBIDDEN_STATUS_KEYS = (
    "witness_augmentation",
    "validated_seed_routes_sha256",
    "inherited_event_pool_status_sha256",
    "diving_augmentation",
)
FORBIDDEN_PATH_PATTERN = re.compile(
    r"(?:^|[/_.-])(?:warm|witness|giro|seed_routes|mip_warm)(?:[/_.-]|$)",
    re.IGNORECASE,
)

NODE_INTEGRAL = "node_integral"
NODE_FRACTIONAL = "node_fractional"
NODE_ARTIFICIAL_REMAINS = "node_artificials_remain"
NODE_UNCERTIFIED = "node_uncertified"


class DivePilotError(RuntimeError):
    """Refusal or unrecoverable inconsistency in the diving pilot."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def trip_set_sha256(keys) -> str:
    payload = json.dumps(
        [sorted(int(trip) for trip in key) for key in keys],
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def git_value(repo: Path, *args) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


class Budget:
    """One strict wall-clock budget shared by every stage of a dive run."""

    def __init__(self, limit_s: float | None):
        self.limit_s = None if limit_s is None else float(limit_s)
        self.started = time.time()

    def elapsed_s(self) -> float:
        return time.time() - self.started

    def remaining_s(self, reserve_s: float = 0.0) -> float:
        if self.limit_s is None:
            return math.inf
        return self.limit_s - self.elapsed_s() - reserve_s

    def expired(self, reserve_s: float = 0.0) -> bool:
        return self.remaining_s(reserve_s) <= 0.0


# --------------------------------------------------------------------------
# restricted master
# --------------------------------------------------------------------------

class DiveLPResult:
    __slots__ = (
        "objective", "trip_duals", "fleet_dual", "artificial_total",
        "artificial_values", "route_values", "reduced_costs", "runtime_s",
        "max_row_violation", "status",
    )

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))


class DiveMaster:
    """Covering restricted master with a fleet cap and fixable route bounds.

    Rows
        ``cover[t]: sum_{r : t in r} x_r + a_t >= 1`` for every trip ``t``
        ``fleet:    sum_r x_r <= fleet_cap``

    Objective
        ``min sum_r c_r x_r + M sum_t a_t`` with ``M = BIG_M_PENALTY``, the
        same artificial penalty the production restricted master uses, so the
        pricing objective stays the true combined cost.

    Unfixed route variables are ``x_r >= 0`` with **no** upper bound, exactly
    as in the production restricted master.  An explicit ``x_r <= 1`` is
    redundant here -- with nonnegative costs, clipping any ``x_r`` to one
    preserves ``Ax >= 1``, relaxes the fleet row and lowers the objective --
    but it is not harmless: a column sitting at an upper bound may carry a
    negative reduced cost at optimality, which destroys the pricing
    certificate.  (Observed directly: with ``ub = 1`` the fleet dual settled
    at -499,000 on a degenerate vertex and pricing could never certify.)

    A dive fixes a route with ``lb = ub = 1``.  Fixed variables are nonbasic
    at a bound, so the row duals still price every *free* column correctly.
    The LP is feasible at every node with at most ``fleet_cap`` fixings
    because the artificials are unbounded above.
    """

    def __init__(
        self, trip_ids, *, fleet_cap, artificial_penalty,
        seed=0, threads=1, log_file=None, feasibility_tolerance=1e-6,
    ):
        import gurobipy as gp

        self._gp = gp
        self.trip_ids = [int(trip) for trip in trip_ids]
        if len(set(self.trip_ids)) != len(self.trip_ids):
            raise DivePilotError("trip_ids repeats a trip")
        self.fleet_cap = int(fleet_cap)
        self.artificial_penalty = float(artificial_penalty)
        self.feasibility_tolerance = float(feasibility_tolerance)
        self.model = gp.Model("evsp_dive_master")
        self.model.Params.OutputFlag = 1 if log_file else 0
        self.model.Params.Threads = int(threads)
        self.model.Params.Method = 1
        self.model.Params.Seed = int(seed)
        if log_file is not None:
            log_path = Path(log_file).expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.Params.LogFile = str(log_path)

        self._cover = {}
        for trip in self.trip_ids:
            self._cover[trip] = self.model.addConstr(
                gp.LinExpr() >= 1.0, name=f"cover[{trip}]"
            )
        self._fleet = self.model.addConstr(
            gp.LinExpr() <= float(self.fleet_cap), name="fleet"
        )
        self._artificial = {}
        for trip in self.trip_ids:
            column = gp.Column()
            column.addTerms(1.0, self._cover[trip])
            self._artificial[trip] = self.model.addVar(
                lb=0.0, ub=gp.GRB.INFINITY, obj=self.artificial_penalty,
                column=column, name=f"artificial[{trip}]",
            )
        self.keys = []            # insertion-ordered frozenset keys
        self._var_list = []       # parallel to self.keys, for batched getAttr
        self._vars = {}           # key -> gurobi var
        self._costs = {}          # key -> cost
        self._trips = {}          # key -> ordered trip list
        self._artificial_list = [
            self._artificial[trip] for trip in self.trip_ids
        ]
        self._fixed = set()       # keys currently pinned to one
        self.model.update()

    # -- columns ---------------------------------------------------------
    def add_route(self, trips, cost) -> bool:
        """Add one column, or lower an existing column's cost.

        Returns ``True`` when the pool changed.
        """
        ordered = [int(trip) for trip in trips]
        if not ordered:
            raise DivePilotError("route contains no trips")
        if len(set(ordered)) != len(ordered):
            raise DivePilotError("route repeats a trip")
        unknown = [trip for trip in ordered if trip not in self._cover]
        if unknown:
            raise DivePilotError(f"route has unknown trips: {unknown[:8]}")
        cost = float(cost)
        if not math.isfinite(cost) or cost < 0.0:
            raise DivePilotError("route cost must be finite and nonnegative")
        key = frozenset(ordered)
        existing = self._vars.get(key)
        if existing is not None:
            if cost < self._costs[key] - 1e-9:
                existing.Obj = cost
                self._costs[key] = cost
                self._trips[key] = ordered
                self.model.update()
                return True
            return False
        column = self._gp.Column()
        for trip in ordered:
            column.addTerms(1.0, self._cover[trip])
        column.addTerms(1.0, self._fleet)
        variable = self.model.addVar(
            lb=0.0, ub=self._gp.GRB.INFINITY, obj=cost, column=column,
            name=f"route[{len(self.keys)}]",
        )
        self._vars[key] = variable
        self._var_list.append(variable)
        self._costs[key] = cost
        self._trips[key] = ordered
        self.keys.append(key)
        self.model.update()
        return True

    def add_routes(self, routes) -> int:
        added = 0
        for route in routes:
            if self.add_route(route["trips"], route["cost"]):
                added += 1
        return added

    def cost_of(self, key) -> float:
        return self._costs[key]

    def trips_of(self, key):
        return list(self._trips[key])

    # -- bounds ----------------------------------------------------------
    def set_fixed(self, fixed_keys) -> None:
        """Fix exactly ``fixed_keys`` to one; release every other column."""

        fixed = list(fixed_keys)
        if len(set(fixed)) != len(fixed):
            raise DivePilotError("duplicate route fixed in one dive path")
        if len(fixed) > self.fleet_cap:
            raise DivePilotError("more routes fixed than the fleet cap allows")
        wanted = set(fixed)
        free_upper = self._gp.GRB.INFINITY
        # Touch only the columns whose bounds actually change.  A full sweep
        # costs two attribute round trips per column per dive node, which at
        # 40,000 columns dominates the node.
        for key in self._fixed - wanted:
            variable = self._vars[key]
            variable.LB = 0.0
            variable.UB = free_upper
        for key in wanted - self._fixed:
            variable = self._vars[key]
            variable.LB = 1.0
            variable.UB = 1.0
        self._fixed = wanted
        self.model.update()

    # -- solve -----------------------------------------------------------
    def solve(self, time_limit_s=None) -> DiveLPResult:
        gp = self._gp
        if time_limit_s is None or not math.isfinite(time_limit_s):
            self.model.Params.TimeLimit = gp.GRB.INFINITY
        else:
            self.model.Params.TimeLimit = max(1.0, float(time_limit_s))
        started = time.perf_counter()
        self.model.optimize()
        runtime_s = time.perf_counter() - started
        if self.model.Status != gp.GRB.OPTIMAL:
            raise DivePilotError(
                f"dive restricted master did not reach OPTIMAL: "
                f"status={self.model.Status}"
            )
        # Batched attribute reads: a per-variable ``.X``/``.RC`` round trip is
        # a measurable cost at 40k columns.
        get = self.model.getAttr
        route_values = dict(zip(self.keys, get("X", self._var_list)))
        reduced_costs = dict(zip(self.keys, get("RC", self._var_list)))
        artificial_values = dict(
            zip(self.trip_ids, get("X", self._artificial_list))
        )
        trip_duals = dict(zip(
            self.trip_ids,
            get("Pi", [self._cover[trip] for trip in self.trip_ids]),
        ))
        fleet_dual = float(self._fleet.Pi)
        if not all(math.isfinite(value) for value in trip_duals.values()):
            raise DivePilotError("dive master returned non-finite trip duals")
        if not math.isfinite(fleet_dual):
            raise DivePilotError("dive master returned a non-finite fleet dual")
        # A `<= cap` row in a minimization has a nonpositive dual.  The pricing
        # shift below relies on it, so verify rather than assume.
        if fleet_dual > 1e-6:
            raise DivePilotError(
                f"fleet-cap dual must be nonpositive; got {fleet_dual}"
            )
        # Accumulate coverage over the *support* only.  Scanning every column
        # for every trip is O(trips x columns) -- 194 x 40,000 per LP on the
        # real pools -- while the support of a covering LP is a handful of
        # routes.
        coverage = {}
        fleet_use = 0.0
        for key, value in route_values.items():
            if value <= 0.0:   # nonbasic columns are exactly zero
                continue
            fleet_use += value
            for trip in key:
                coverage[trip] = coverage.get(trip, 0.0) + value
        max_row_violation = 0.0
        for trip in self.trip_ids:
            covered = artificial_values[trip] + coverage.get(trip, 0.0)
            max_row_violation = max(
                max_row_violation, max(0.0, 1.0 - covered)
            )
        max_row_violation = max(
            max_row_violation, max(0.0, fleet_use - self.fleet_cap)
        )
        if max_row_violation > self.feasibility_tolerance:
            raise DivePilotError(
                f"dive master returned a row-infeasible LP: "
                f"violation={max_row_violation}"
            )
        return DiveLPResult(
            objective=float(self.model.ObjVal),
            trip_duals=trip_duals,
            fleet_dual=fleet_dual,
            artificial_total=float(sum(artificial_values.values())),
            artificial_values=artificial_values,
            route_values=route_values,
            reduced_costs=reduced_costs,
            runtime_s=runtime_s,
            max_row_violation=max_row_violation,
            status="optimal",
        )

    def close(self) -> None:
        try:
            self.model.dispose()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------

class EventPricer:
    """Adapter over ``EventExpandedNetwork`` with a fleet-cap reduced cost.

    The network's ``route_dual`` argument is deliberately **not** used.  In
    ``_min_reduced_cost_route_lazy`` the non-fast branch applies
    ``arc_costs - BUS_COST_KX`` at the source arc for *any* objective other
    than ``artificial-elimination``/``fleet-only`` -- including
    ``combined-cost`` -- whereas the explicit-arc path subtracts
    ``BUS_COST_KX`` only for ``charging-cost``.  Passing a nonzero
    ``route_dual`` under ``--event-arc-mode lazy`` (the mode every campaign
    uses) would therefore silently shift every reduced cost by 1e5.  The
    campaign never triggers it because production has no fleet row.

    Because the fleet dual ``mu`` enters every column identically, shifting
    afterwards is exact: ``argmin_r (c_r - alpha.a_r - mu)`` equals
    ``argmin_r (c_r - alpha.a_r)``, and the node certificate is the network's
    own minimum minus ``mu``.
    """

    def __init__(self, network, *, columns_per_iter=30):
        self.network = network
        self.columns_per_iter = int(columns_per_iter)
        self.calls = 0
        self.total_s = 0.0

    def price(self, trip_duals, fleet_dual):
        started = time.perf_counter()
        batch = self.network.sink_predecessor_route_batch(
            trip_duals, limit=self.columns_per_iter,
            selection_mode="reduced_cost",
        )
        self.calls += 1
        self.total_s += time.perf_counter() - started
        if not batch:
            return math.inf, []
        candidates = []
        for route in batch:
            record = route.get("_event_record")
            if record is None:
                raise DivePilotError(
                    "diving pilot requires the event-time pricer "
                    "(--time-model event artifacts)"
                )
            trips = [int(trip) for trip in route["trips"]]
            cost = float(record["cost"])
            dual_sum = sum(float(trip_duals.get(trip, 0.0)) for trip in trips)
            path_rc = float(route["rc"])
            # The network's rc is exactly cost - alpha.a for this record.
            if not math.isclose(
                path_rc, cost - dual_sum, rel_tol=1e-9, abs_tol=1e-6
            ):
                raise DivePilotError(
                    "pricing reduced cost disagrees with the replayed record "
                    f"cost: rc={path_rc} cost-duals={cost - dual_sum}"
                )
            candidates.append({
                "trips": trips,
                "cost": cost,
                "rc_path": path_rc,
                "rc_true": path_rc - fleet_dual,
                "record": record,
            })
        node_min_rc = float(batch[0]["rc"]) - float(fleet_dual)
        return node_min_rc, candidates


# --------------------------------------------------------------------------
# source loading
# --------------------------------------------------------------------------

def _refuse_witness_paths(paths) -> None:
    for path in paths:
        if path is None:
            continue
        text = str(path)
        for part in Path(text).parts:
            if FORBIDDEN_PATH_PATTERN.search(part):
                raise DivePilotError(
                    f"refusing an input path that looks warm/witness-derived: "
                    f"{text}"
                )


def load_source_pool(result_path: Path):
    """Load the frozen fresh status and its journal, read-only.

    Returns ``(status, journal_path, routes, trips)`` where ``routes`` is a
    list of ``{"trips", "cost"}`` deduplicated to the cheapest record per trip
    incidence, matching ``run_exact_pool_mip.load_pool``.
    """

    from durable_io import read_jsonl_records
    from run_exact_pool_mip import resolve_pool_journal

    result_path = Path(result_path).expanduser().resolve()
    status = json.loads(result_path.read_text())
    for key in FORBIDDEN_STATUS_KEYS:
        if status.get(key):
            raise DivePilotError(
                f"refusing {result_path}: status carries {key!r}; the pilot "
                "must start from a fresh, un-augmented pool"
            )
    treatment = status.get("column_pool_treatment")
    if treatment not in (None, "RAW"):
        raise DivePilotError(
            f"refusing {result_path}: column_pool_treatment={treatment!r} is "
            "not RAW"
        )
    if status.get("initial_pool") not in (None, "singletons", "artificial"):
        raise DivePilotError(
            f"refusing {result_path}: unexpected initial_pool "
            f"{status.get('initial_pool')!r}"
        )
    journal_path = resolve_pool_journal(result_path, status).resolve()
    _refuse_witness_paths([result_path, journal_path])

    trips = [int(trip) for trip in status["trip_ids"]]
    allowed = set(trips)
    pool = {}
    for ordinal, record in enumerate(
        read_jsonl_records(journal_path, repair_trailing=False), start=1
    ):
        route_trips = record.get("trips")
        if not isinstance(route_trips, list) or not route_trips:
            raise DivePilotError(
                f"{journal_path} record {ordinal} has no nonempty trips list"
            )
        route_trips = [int(trip) for trip in route_trips]
        if len(set(route_trips)) != len(route_trips):
            raise DivePilotError(
                f"{journal_path} record {ordinal} repeats a trip"
            )
        if any(trip not in allowed for trip in route_trips):
            raise DivePilotError(
                f"{journal_path} record {ordinal} leaves the trip set"
            )
        cost = float(record["cost"])
        if not math.isfinite(cost):
            raise DivePilotError(
                f"{journal_path} record {ordinal} has a non-finite cost"
            )
        key = frozenset(route_trips)
        if key not in pool or cost < pool[key]["cost"] - 1e-9:
            pool[key] = {"trips": route_trips, "cost": cost}
    return status, journal_path, list(pool.values()), trips


# --------------------------------------------------------------------------
# the dive
# --------------------------------------------------------------------------

RESTART_RULES = ("max_value", "max_value_long_route", "max_value_cheap_seat")


def _near(value, target) -> bool:
    return abs(float(value) - float(target)) <= INTEGRALITY_TOL


def rank_candidates(lp, master, rule):
    """Deterministically order fractional routes as branching candidates."""

    rows = []
    for key, value in lp.route_values.items():
        if _near(value, 0.0) or _near(value, 1.0):
            continue
        trips = master.trips_of(key)
        cost = master.cost_of(key)
        sort_trips = tuple(sorted(trips))
        if rule == "max_value":
            order = (-value, -len(trips), cost, sort_trips)
        elif rule == "max_value_long_route":
            order = (-len(trips), -value, cost, sort_trips)
        elif rule == "max_value_cheap_seat":
            order = (cost / len(trips), -value, -len(trips), sort_trips)
        else:
            raise DivePilotError(f"unknown branching rule: {rule!r}")
        rows.append((order, key))
    rows.sort(key=lambda row: row[0])
    return [key for _order, key in rows]


class DivePilot:
    def __init__(self, *, master, pricer, trips, budget, journal, options):
        self.master = master
        self.pricer = pricer
        self.trips = list(trips)
        self.trip_set = set(self.trips)
        self.budget = budget
        self.journal = journal
        self.options = options
        self.node_counter = 0
        self.generated = []          # ordered new records
        self.generated_keys = set()
        self.integer_solution = None
        self.stop_reason = "not_started"
        self.node_outcomes = []

    # -- journaling ------------------------------------------------------
    def emit(self, kind, **payload):
        entry = {
            "kind": kind,
            "elapsed_s": round(self.budget.elapsed_s(), 6),
            **payload,
        }
        self.journal.write(json.dumps(entry, sort_keys=True,
                                      separators=(",", ":")) + "\n")
        self.journal.flush()
        return entry

    # -- one dive node ---------------------------------------------------
    def solve_node(self, fixed_keys):
        """Price a dive node to certification, the node cap or the budget."""

        self.node_counter += 1
        node_id = self.node_counter
        fixed_keys = list(fixed_keys)
        self.master.set_fixed(fixed_keys)
        pricing_iters = 0
        added_here = 0
        node_started = time.time()
        lp = None
        min_rc = None
        certified = False
        while True:
            if self.budget.expired(self.options.reserve_s):
                outcome = NODE_UNCERTIFIED
                stop = "wall_limit"
                break
            lp = self.master.solve(
                time_limit_s=self.budget.remaining_s(self.options.reserve_s)
            )
            if (time.time() - node_started) > self.options.node_time_s:
                outcome = NODE_UNCERTIFIED
                stop = "node_time_limit"
                break
            if pricing_iters >= self.options.max_pricing_iters:
                outcome = NODE_UNCERTIFIED
                stop = "node_pricing_iter_limit"
                break
            min_rc, candidates = self.pricer.price(
                lp.trip_duals, lp.fleet_dual
            )
            pricing_iters += 1
            improving = [
                candidate for candidate in candidates
                if candidate["rc_true"] < -self.options.rc_eps
            ]
            added = self.accept_columns(improving, node_id, len(fixed_keys),
                                        fixed_keys, lp.fleet_dual)
            added_here += added
            if min_rc >= -self.options.rc_eps:
                certified = True
                break
            if added == 0:
                # Every improving incidence is already present at this cost:
                # the duals sit on a degenerate vertex.  Treat the node as
                # exhausted rather than looping; the parent will backtrack.
                outcome = NODE_UNCERTIFIED
                stop = "degenerate_stall"
                break
        if certified:
            if lp.artificial_total > ARTIFICIAL_TOL:
                outcome = NODE_ARTIFICIAL_REMAINS
                stop = "penalized_pricing_closed_artificials_remain"
            elif self.is_integral(lp):
                outcome = NODE_INTEGRAL
                stop = "certified_integral"
            else:
                outcome = NODE_FRACTIONAL
                stop = "certified_fractional"
        record = {
            "node_id": node_id,
            "depth": len(fixed_keys),
            "fixed_trip_sets_sha256": trip_set_sha256(fixed_keys),
            "fixed_routes": [sorted(int(t) for t in key) for key in fixed_keys],
            "outcome": outcome,
            "node_stop": stop,
            "lp_objective": None if lp is None else lp.objective,
            "lp_route_weight": None if lp is None else sum(
                lp.route_values.values()
            ),
            "artificial_total": None if lp is None else lp.artificial_total,
            "fleet_dual": None if lp is None else lp.fleet_dual,
            "min_reduced_cost_with_fleet_dual": (
                None if min_rc is None or not math.isfinite(min_rc)
                else min_rc
            ),
            "pricing_iterations": pricing_iters,
            "columns_added": added_here,
            "node_wall_s": round(time.time() - node_started, 6),
            "scope": "dive node only; never a global certificate",
        }
        self.node_outcomes.append(record)
        self.emit("node", **record)
        return outcome, lp, record

    def is_integral(self, lp) -> bool:
        # Zero or one only.  A value above one cannot occur at an optimum of
        # this covering LP (clipping to one is always cheaper and feasible),
        # but treating it as fractional keeps the check honest if it ever does.
        return all(
            _near(value, 0.0) or _near(value, 1.0)
            for value in lp.route_values.values()
        )

    def accept_columns(self, candidates, node_id, depth, fixed_keys,
                       fleet_dual) -> int:
        added = 0
        for candidate in candidates:
            key = frozenset(candidate["trips"])
            changed = self.master.add_route(
                candidate["trips"], candidate["cost"]
            )
            if not changed:
                continue
            record = dict(candidate["record"])
            record["trips"] = [int(trip) for trip in record["trips"]]
            record.update({
                "cost": float(candidate["cost"]),
                "expanded_grid_cost": float(candidate["cost"]),
                "origin": "diving_pricing",
                "dive_node": node_id,
                "dive_depth": depth,
                "dive_fixed_trip_sets_sha256": trip_set_sha256(fixed_keys),
                "dive_rc_true_at_generation": float(candidate["rc_true"]),
                "dive_rc_path_at_generation": float(candidate["rc_path"]),
                "dive_fleet_dual_at_generation": float(fleet_dual),
                "cost_tariff_sha256": self.options.prices_sha256,
            })
            self.validate_record(record)
            if key not in self.generated_keys:
                self.generated_keys.add(key)
            self.generated.append(record)
            added += 1
        return added

    def validate_record(self, record) -> None:
        """Exact physical replay of one route before it may be appended."""

        from run_exact_pool_mip import validate_injected_route

        # ``physical_replay`` has no command-line switch: ``resolve_options``
        # always sets it True.  Only unit tests driving a stub pricer (whose
        # synthetic records have no model graph to replay against) turn it off.
        if not getattr(self.options, "physical_replay", True):
            return

        reason = validate_injected_route(
            self.options.problem, record,
            self.options.g_kwh, self.options.charge_kw,
            self.options.reserve_kwh, self.options.horizon_min,
            arrival_grace_min=0.0,
        )
        if reason is not None:
            raise DivePilotError(
                f"generated route failed physical replay: {reason}"
            )

    # -- the search ------------------------------------------------------
    def run(self):
        rule_index = 0
        restarts = 0
        while True:
            rule = RESTART_RULES[rule_index]
            self.emit("dive_start", rule=rule, restart=restarts)
            exhausted, max_depth = self.dive(rule)
            if self.integer_solution is not None:
                self.stop_reason = "integer_solution"
                return
            if self.budget.expired(self.options.reserve_s):
                self.stop_reason = "wall_limit"
                return
            if self.node_counter >= self.options.max_nodes:
                self.stop_reason = "node_limit"
                return
            if not exhausted:
                self.stop_reason = "dive_stopped"
                return
            if max_depth == 0:
                # The root itself could not be branched.  A different
                # branching rule ranks the same (empty) candidate list, so a
                # restart would repeat identical work.
                self.emit("restart_declined", reason="root_never_branched")
                self.stop_reason = "root_not_branchable"
                return
            rule_index += 1
            restarts += 1
            if rule_index >= len(RESTART_RULES) or (
                restarts > self.options.max_restarts
            ):
                self.stop_reason = "search_exhausted"
                return

    def dive(self, rule):
        """Run one deterministic dive.

        Returns ``(tree_exhausted, max_depth_reached)``.
        """

        levels = []   # [{"candidates": [...], "index": int}]
        fixed = []
        max_depth = 0
        while True:
            if self.budget.expired(self.options.reserve_s):
                return False, max_depth
            if self.node_counter >= self.options.max_nodes:
                return False, max_depth
            outcome, lp, _record = self.solve_node(fixed)
            if outcome == NODE_INTEGRAL:
                self.record_integer_solution(lp, fixed)
                return False, max_depth
            if self.covers_everything(fixed):
                self.record_integer_solution(lp, fixed, from_fixings=True)
                return False, max_depth
            # Surviving artificials trigger heuristic backtracking, not a proof
            # of infeasibility for the unpenalized problem. An uncertified node is a
            # heuristic column generator that ran out of its per-node
            # allowance: if its coverage is complete we may still branch on it,
            # and it is never called infeasible.
            retreat = outcome == NODE_ARTIFICIAL_REMAINS
            if outcome == NODE_UNCERTIFIED and (
                lp is None or lp.artificial_total > ARTIFICIAL_TOL
            ):
                retreat = True
            if not retreat and len(fixed) >= self.master.fleet_cap:
                retreat = True
            candidates = [] if retreat else rank_candidates(
                lp, self.master, rule
            )
            if not retreat and not candidates:
                retreat = True
            if retreat:
                if not self.backtrack(levels, fixed):
                    return True, max_depth
                continue
            levels.append({
                "candidates": candidates[: self.options.max_alternatives],
                "index": 0,
            })
            fixed.append(candidates[0])
            max_depth = max(max_depth, len(fixed))
            self.emit(
                "fix", depth=len(fixed), rule=rule, alternative=0,
                node_outcome=outcome,
                route=sorted(int(trip) for trip in candidates[0]),
                value=lp.route_values[candidates[0]],
                cost=self.master.cost_of(candidates[0]),
            )

    def backtrack(self, levels, fixed) -> bool:
        while levels:
            level = levels[-1]
            level["index"] += 1
            if level["index"] < len(level["candidates"]):
                key = level["candidates"][level["index"]]
                fixed[-1] = key
                self.emit(
                    "backtrack", depth=len(fixed),
                    alternative=level["index"],
                    route=sorted(int(trip) for trip in key),
                )
                return True
            levels.pop()
            fixed.pop()
            self.emit("retreat", depth=len(fixed))
        return False

    def covers_everything(self, fixed) -> bool:
        covered = set()
        for key in fixed:
            covered |= set(key)
        return covered == self.trip_set

    def record_integer_solution(self, lp, fixed, from_fixings=False) -> None:
        if from_fixings:
            keys = list(fixed)
        else:
            keys = [
                key for key, value in lp.route_values.items()
                if _near(value, 1.0)
            ]
        covered = set()
        for key in keys:
            covered |= set(key)
        if covered != self.trip_set:
            return
        if len(keys) > self.master.fleet_cap:
            return
        self.integer_solution = {
            "buses": len(keys),
            "routes": [
                {
                    "trips": self.master.trips_of(key),
                    "cost": self.master.cost_of(key),
                }
                for key in keys
            ],
            "total_cost": sum(self.master.cost_of(key) for key in keys),
            "from_fixings_only": bool(from_fixings),
            "scope": (
                "feasible integer cover found inside the dive; it is an "
                "incumbent, not an optimality or infeasibility certificate"
            ),
        }
        self.emit("integer_solution", **self.integer_solution)


# --------------------------------------------------------------------------
# artifact publication
# --------------------------------------------------------------------------

def publish_augmented_pool(
    *, source_status, source_result_path, source_journal_path,
    source_result_sha256, source_journal_sha256, generated_records, out_dir,
) -> dict:
    """Write ``cg.json`` + journal in the layout run_exact_pool_mip consumes.

    The original journal is copied byte-for-byte and the new records are
    appended; the source files are never opened for writing.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    augmented_journal = out_dir / "cg.json.columns.jsonl"
    augmented_status_path = out_dir / "cg.json"

    shutil.copyfile(source_journal_path, augmented_journal)
    if augmented_journal.stat().st_size:
        with open(augmented_journal, "rb") as handle:
            handle.seek(-1, os.SEEK_END)
            trailing_newline = handle.read(1) == b"\n"
    else:
        trailing_newline = True
    with open(augmented_journal, "ab") as handle:
        if not trailing_newline:
            handle.write(b"\n")
        for record in generated_records:
            handle.write(
                (json.dumps(record, separators=(",", ":")) + "\n").encode()
            )
        handle.flush()
        os.fsync(handle.fileno())

    status = dict(source_status)
    status["columns_journal"] = str(augmented_journal.resolve())
    status["diving_augmentation"] = {
        "schema": DIVE_SCHEMA,
        "track": "diving_pricing_20260919",
        "source_result": str(source_result_path),
        "source_result_sha256": source_result_sha256,
        "source_journal": str(source_journal_path),
        "source_journal_sha256": source_journal_sha256,
        "appended_records": len(generated_records),
        "witness_columns_used": False,
        "note": (
            "identical to the source cg.json except columns_journal and this "
            "block; appended routes were generated by diving-with-pricing "
            "from the original trip/physics inputs only"
        ),
    }
    augmented_status_path.write_text(json.dumps(status, indent=1))
    return {
        "augmented_result": str(augmented_status_path.resolve()),
        "augmented_result_sha256": file_sha256(augmented_status_path),
        "augmented_journal": str(augmented_journal.resolve()),
        "augmented_journal_sha256": file_sha256(augmented_journal),
        "appended_records": len(generated_records),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

class Options:
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Experimental bounded diving-with-pricing pilot"
    )
    parser.add_argument("--result", type=Path, required=True,
                        help="frozen fresh CG status (cg.json) to dive on")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--fleet-cap", type=int, required=True)
    parser.add_argument("--wall-limit-s", type=float, required=True)
    parser.add_argument("--reserve-s", type=float, default=60.0,
                        help="serialization margin held back from the budget")
    parser.add_argument("--rc-eps", type=float, default=DEFAULT_RC_EPS)
    parser.add_argument("--columns-per-iter", type=int, default=30)
    parser.add_argument("--max-pricing-iters", type=int, default=40,
                        help="pricing rounds per dive node")
    parser.add_argument("--node-time-s", type=float, default=300.0)
    parser.add_argument("--max-nodes", type=int, default=64)
    parser.add_argument("--max-alternatives", type=int, default=3,
                        help="branching alternatives retained per dive level")
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--event-network-cache", type=Path, default=None)
    parser.add_argument("--cache-commit-bridge", action="store_true",
                        help="permit a cache whose only identity difference "
                             "is git_commit, after a recorded method audit")
    parser.add_argument("--skip-cache-hash", action="store_true",
                        help="skip the multi-GB pickle rehash; the producer "
                             "manifest hash is then recorded as unverified")
    parser.add_argument("--build-network", action="store_true",
                        help="build the event network locally when no cache "
                             "is supplied (expensive: hours at k=8)")
    parser.add_argument("--gurobi-log", type=Path, default=None)
    return parser


def resolve_options(args, status, trips) -> Options:
    from audit_giro_known_columns import HORIZON_MIN, build_problem

    options = Options()
    options.physical_replay = True
    options.rc_eps = float(args.rc_eps)
    options.reserve_s = float(args.reserve_s)
    options.max_pricing_iters = int(args.max_pricing_iters)
    options.node_time_s = float(args.node_time_s)
    options.max_nodes = int(args.max_nodes)
    options.max_alternatives = max(1, int(args.max_alternatives))
    options.max_restarts = int(args.max_restarts)
    options.horizon_min = HORIZON_MIN
    options.g_kwh = float(status["g_kwh"])
    options.charge_kw = float(status["charge_kw"])
    options.reserve_kwh = float(status["min_soc_frac"]) * options.g_kwh
    data_dir = (
        Path(args.data_dir).expanduser().resolve() if args.data_dir
        else Path(__file__).resolve().parent.parent / "data"
    )
    options.data_dir = data_dir
    options.csv = status["csv"]
    options.prices_csv = status["prices_csv"]
    # The cache identity is checked against the *status* provenance, so a
    # wrong --data-dir would pass every cache check and still replay routes
    # against a different instance.  Bind the actual bytes on disk to the
    # provenance the frozen pool recorded, as prep_witness.py does.
    provenance = status.get("provenance") or {}
    options.input_hashes = {}
    for key, relative in (
        ("instance_sha256", status["csv"]),
        ("prices_sha256", status["prices_csv"]),
        ("reference_sha256", "Ref_dict.csv"),
        ("deadhead_sha256", "par_ref_dhd.csv"),
    ):
        expected = provenance.get(key)
        path = data_dir / relative
        if expected is None:
            raise DivePilotError(
                f"source status provenance lacks {key}; refusing to run "
                "against unverifiable inputs"
            )
        if not path.is_file():
            raise DivePilotError(f"missing model input {path}")
        observed = file_sha256(path)
        options.input_hashes[key] = observed
        if observed != expected:
            raise DivePilotError(
                f"{path} does not match the frozen pool's {key}: "
                f"expected {expected}, found {observed}"
            )
    options.problem = build_problem(
        data_dir, status["csv"], max_station_to_trip_wait_min=HORIZON_MIN
    )
    if list(options.problem.trips) != trips:
        raise DivePilotError(
            "rebuilt problem trip ids differ from the recorded pool trip ids"
        )
    options.prices_sha256 = (status.get("provenance") or {}).get(
        "prices_sha256"
    )
    return options


def acquire_network(args, status, options, budget):
    """Load an identity-verified cache, or build the network when asked."""

    from diving_cache_identity import load_verified_network
    from event_pricer_network import EventExpandedNetwork
    from exact_pricer_expanded import EVENT_NETWORK_CACHE_SCHEMA
    from utils_v2 import load_station_hourly_prices
    from config import CHARGING_STATIONS

    repo = Path(__file__).resolve().parent.parent
    prices = load_station_hourly_prices(
        options.data_dir / options.prices_csv, CHARGING_STATIONS
    )
    provenance = status.get("provenance") or {}
    expected = {
        "schema": EVENT_NETWORK_CACHE_SCHEMA,
        "git_commit": git_value(repo, "rev-parse", "HEAD"),
        "instance_sha256": provenance.get("instance_sha256"),
        "prices_sha256": provenance.get("prices_sha256"),
        "reference_sha256": provenance.get("reference_sha256"),
        "deadhead_sha256": provenance.get("deadhead_sha256"),
        "soc_step": float(status["soc_step"]),
        "block_min": int(status["block_min"]),
        "g_kwh": float(status["g_kwh"]),
        "charge_kw": float(status["charge_kw"]),
        "reserve_kwh": float(status["min_soc_frac"]) * float(status["g_kwh"]),
        "strict_tariff_coverage": bool(
            status.get("strict_tariff_coverage", False)
        ),
        "event_arc_mode": (
            (status.get("network_metrics") or {}).get("arc_mode") or "lazy"
        ),
    }
    started = time.time()
    if args.event_network_cache is not None:
        network, audit = load_verified_network(
            args.event_network_cache, expected, repo=repo,
            allow_commit_bridge=bool(args.cache_commit_bridge),
            verify_pickle_sha256=not args.skip_cache_hash,
        )
        audit["acquisition"] = "cache"
        audit["acquisition_s"] = time.time() - started
        # Several k=8 graph caches are *stored* under warm-chain directories
        # (``nested_warm_chain_...``, ``cases/w1_k08/``).  That is a storage
        # location, not warm data: an EventExpandedNetwork is a function of
        # the instance and physics only and contains no routes, and every
        # identity field above is checked against this fresh pool's own
        # provenance.  The path is deliberately NOT run through the
        # warm/witness refusal that guards column sources -- so record the
        # fact explicitly instead of leaving it an unremarked exemption.
        cache_parts = Path(args.event_network_cache).parts
        audit["cache_path_warm_named"] = any(
            FORBIDDEN_PATH_PATTERN.search(part) for part in cache_parts
        )
        audit["cache_contains_columns"] = False
        audit["cache_path_exemption_note"] = (
            "graph-only artifact; identity verified field-by-field against "
            "the fresh pool provenance; no route/column is read from it"
        )
        return network, audit
    if not args.build_network:
        raise DivePilotError(
            "no --event-network-cache supplied; pass --build-network to "
            "accept a full local graph build (hours at k=8)"
        )
    network = EventExpandedNetwork(
        options.problem, prices,
        soc_step=float(status["soc_step"]),
        block_min=int(status["block_min"]),
        g_kwh=options.g_kwh,
        charge_kw=options.charge_kw,
        reserve_kwh=options.reserve_kwh,
        strict_tariff_coverage=bool(
            status.get("strict_tariff_coverage", False)
        ),
        arc_mode=expected["event_arc_mode"],
    )
    return network, {
        "schema": "evsp-dr-diving-cache-bridge-v1",
        "acquisition": "local_build",
        "acquisition_s": time.time() - started,
        "expected_identity": expected,
        "network_metrics": network.metrics(),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    repo = Path(__file__).resolve().parent.parent
    budget = Budget(args.wall_limit_s)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    status, journal_path, source_routes, trips = load_source_pool(args.result)
    source_result_path = Path(args.result).expanduser().resolve()
    if out_dir == source_result_path.parent:
        raise DivePilotError(
            "--out-dir must not be the source pool directory; the source "
            "artifacts are immutable inputs"
        )
    source_result_sha256 = file_sha256(source_result_path)
    source_journal_sha256 = file_sha256(journal_path)

    options = resolve_options(args, status, trips)
    timings = {"pool_load_s": budget.elapsed_s()}

    network, cache_audit = acquire_network(args, status, options, budget)
    timings["network_acquisition_s"] = cache_audit.get("acquisition_s")

    from config import BIG_M_PENALTY

    master_started = time.time()
    master = DiveMaster(
        trips, fleet_cap=args.fleet_cap,
        artificial_penalty=BIG_M_PENALTY, seed=args.seed,
        threads=1, log_file=args.gurobi_log,
    )
    master.add_routes(source_routes)
    timings["master_build_s"] = time.time() - master_started
    original_column_count = len(master.keys)

    journal_file = open(out_dir / "dive_journal.jsonl", "w")
    pilot = None
    error = None
    try:
        pricer = EventPricer(network, columns_per_iter=args.columns_per_iter)
        pilot = DivePilot(
            master=master, pricer=pricer, trips=trips, budget=budget,
            journal=journal_file, options=options,
        )
        pilot.run()
    except Exception as exc:  # record the blocker, still publish artifacts
        error = repr(exc)
        if pilot is not None:
            pilot.stop_reason = "error"
        journal_file.write(json.dumps({"kind": "error", "error": error}) + "\n")
    finally:
        journal_file.close()

    generated = pilot.generated if pilot is not None else []
    publication = publish_augmented_pool(
        source_status=status,
        source_result_path=source_result_path,
        source_journal_path=journal_path,
        source_result_sha256=source_result_sha256,
        source_journal_sha256=source_journal_sha256,
        generated_records=generated,
        out_dir=out_dir,
    )

    # Source immutability, re-checked after every write this process made.
    final_result_sha256 = file_sha256(source_result_path)
    final_journal_sha256 = file_sha256(journal_path)
    immutable = (
        final_result_sha256 == source_result_sha256
        and final_journal_sha256 == source_journal_sha256
    )

    manifest = {
        "schema": DIVE_SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv if argv is None else ["diving_pricing_pilot", *argv],
        "git_commit": git_value(repo, "rev-parse", "HEAD"),
        "git_dirty": bool(git_value(repo, "status", "--porcelain")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "fleet_cap": int(args.fleet_cap),
        "seed": int(args.seed),
        "rc_eps": float(args.rc_eps),
        "wall_limit_s": float(args.wall_limit_s),
        "wall_s": budget.elapsed_s(),
        "timings_s": timings,
        "pricing": {
            "calls": pilot.pricer.calls if pilot else 0,
            "total_s": pilot.pricer.total_s if pilot else 0.0,
            "columns_per_iter": int(args.columns_per_iter),
        },
        "source": {
            "result": str(source_result_path),
            "result_sha256": source_result_sha256,
            "journal": str(journal_path),
            "journal_sha256": source_journal_sha256,
            "result_sha256_after_run": final_result_sha256,
            "journal_sha256_after_run": final_journal_sha256,
            "source_immutable": immutable,
            "csv": options.csv,
            "prices_csv": options.prices_csv,
            "provenance": status.get("provenance"),
            "verified_input_sha256": options.input_hashes,
            "original_pool_columns": original_column_count,
            "witness_or_warm_input": False,
        },
        "network_cache_audit": cache_audit,
        "dive": {
            "stop_reason": pilot.stop_reason if pilot else "error",
            "nodes": pilot.node_counter if pilot else 0,
            "node_outcomes": pilot.node_outcomes if pilot else [],
            "columns_generated": len(generated),
            "distinct_new_incidences": (
                len(pilot.generated_keys) if pilot else 0
            ),
            "integer_solution": pilot.integer_solution if pilot else None,
        },
        "augmented_pool": publication,
        "global_certificate": None,
        "scope": (
            "Every LP here is a restricted master over one dive node. No "
            "statement about global optimality or global infeasibility "
            "follows from any node outcome. Fleet proofs come only from the "
            "separate finite-pool MIP run through src/run_exact_pool_mip.py."
        ),
        "error": error,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, default=str))

    print(
        f"[DIVE] stop={manifest['dive']['stop_reason']} "
        f"nodes={manifest['dive']['nodes']} "
        f"new_columns={len(generated)} "
        f"integer={'yes' if manifest['dive']['integer_solution'] else 'no'} "
        f"wall={manifest['wall_s']:.1f}s",
        flush=True,
    )
    if not immutable:
        print("[DIVE] FATAL: source pool artifacts changed during the run",
              flush=True)
        return 3
    return 1 if error else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
