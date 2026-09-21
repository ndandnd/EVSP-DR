"""Tests for the experimental diving-with-pricing pilot."""

import io
import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from audit_giro_known_columns import DEPOT, STATIONS  # noqa: E402
from config import BIG_M_PENALTY, BUS_COST_KX  # noqa: E402
from event_pricer_network import EventExpandedNetwork  # noqa: E402
import diving_pricing_pilot as dive  # noqa: E402
from diving_pricing_pilot import (  # noqa: E402
    Budget,
    DiveMaster,
    DivePilot,
    DivePilotError,
    EventPricer,
    Options,
    load_source_pool,
    publish_augmented_pool,
    rank_candidates,
)


STATION = STATIONS[0]


def gurobi_available():
    """Probe gurobipy directly.

    ``master_lp_gurobi.gurobi_preflight`` additionally demands an explicit
    ``GRB_LICENSE_FILE``, which is a cluster-deployment guard rather than a
    property of the solver; these unit tests only need a working solver.
    """

    try:
        import gurobipy as gp

        model = gp.Model("dive_test_preflight")
        model.Params.OutputFlag = 0
        variable = model.addVar(lb=0.0, ub=1.0, obj=1.0)
        model.addConstr(variable >= 0.5)
        model.optimize()
        ok = model.Status == gp.GRB.OPTIMAL
        model.dispose()
        return ok
    except Exception:
        return False


HAVE_GUROBI = gurobi_available()
needs_gurobi = unittest.skipUnless(
    HAVE_GUROBI, "Gurobi is unavailable in this environment"
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def prices():
    return {
        station.rsplit("_", 1)[0]: {hour: 0.1 for hour in range(27)}
        for station in STATIONS
    }


def chain_problem(n_trips=4):
    trips = tuple(range(n_trips))
    adjacency = {
        DEPOT: [(trip, 0.0, 0.0, "depot_trip") for trip in trips],
    }
    for trip in trips:
        adjacency[trip] = [
            (successor, 0.0, 0.0, "trip_trip")
            for successor in trips[trip + 1:]
        ] + [(DEPOT, 0.0, 0.0, "trip_depot")]
    return SimpleNamespace(
        trips=trips,
        start_min={trip: float(60 * trip) for trip in trips},
        end_min={trip: float(60 * trip + 10) for trip in trips},
        trip_energy={trip: 1.0 for trip in trips},
        adjacency=adjacency,
    )


def chain_network(arc_mode="lazy", n_trips=4):
    return EventExpandedNetwork(
        chain_problem(n_trips), prices(), soc_step=15, block_min=10,
        g_kwh=240.0, charge_kw=240.0, reserve_kwh=0.0, arc_mode=arc_mode,
    )


class StubPricer:
    """Exact pricer over a finite enumerated route universe.

    Mirrors the real adapter's contract: returns ``(node_min_rc, candidates)``
    where ``node_min_rc`` already includes the fleet-cap dual shift.
    """

    def __init__(self, universe, limit=30):
        # universe: {frozenset(trips): (ordered_trips, cost)}
        self.universe = universe
        self.limit = limit
        self.calls = 0
        self.total_s = 0.0

    def price(self, trip_duals, fleet_dual):
        self.calls += 1
        rows = []
        for key, (trips, cost) in self.universe.items():
            rc_path = cost - sum(float(trip_duals.get(t, 0.0)) for t in trips)
            rows.append({
                "trips": list(trips),
                "cost": float(cost),
                "rc_path": rc_path,
                "rc_true": rc_path - float(fleet_dual),
                "record": {"trips": list(trips), "cost": float(cost)},
            })
        rows.sort(key=lambda row: (row["rc_path"], tuple(sorted(row["trips"]))))
        if not rows:
            return math.inf, []
        return rows[0]["rc_true"], rows[: self.limit]


def stub_options(**overrides):
    options = Options()
    options.physical_replay = False
    options.rc_eps = 1e-6
    options.reserve_s = 0.0
    options.max_pricing_iters = 50
    options.node_time_s = 1e9
    options.max_nodes = 200
    options.max_alternatives = 3
    options.max_restarts = 2
    options.prices_sha256 = None
    for name, value in overrides.items():
        setattr(options, name, value)
    return options


def run_stub_pilot(trips, pool, universe, fleet_cap, **option_overrides):
    master = DiveMaster(
        trips, fleet_cap=fleet_cap, artificial_penalty=BIG_M_PENALTY, seed=7
    )
    master.add_routes(pool)
    pilot = DivePilot(
        master=master, pricer=StubPricer(universe), trips=trips,
        budget=Budget(option_overrides.pop("wall_limit_s", None)),
        journal=io.StringIO(), options=stub_options(**option_overrides),
    )
    pilot.run()
    return pilot, master


# --------------------------------------------------------------------------
# 1. reduced costs against direct recomputation
# --------------------------------------------------------------------------

class ReducedCostTests(unittest.TestCase):
    def test_priced_reduced_cost_matches_direct_recomputation(self):
        """rc_true == cost - sum(duals) - mu, recomputed from the record."""

        for arc_mode in ("lazy", "explicit"):
            with self.subTest(arc_mode=arc_mode):
                network = chain_network(arc_mode)
                pricer = EventPricer(network, columns_per_iter=5)
                duals = {0: 110000.0, 1: 90000.0, 2: 105000.0, 3: 70000.0}
                for fleet_dual in (0.0, -250.0, -1e4):
                    node_min_rc, candidates = pricer.price(duals, fleet_dual)
                    self.assertTrue(candidates)
                    for candidate in candidates:
                        direct = (
                            float(candidate["record"]["cost"])
                            - sum(duals[t] for t in candidate["trips"])
                            - fleet_dual
                        )
                        self.assertAlmostEqual(
                            candidate["rc_true"], direct, places=6
                        )
                    self.assertAlmostEqual(
                        node_min_rc,
                        min(c["rc_true"] for c in candidates),
                        places=6,
                    )

    def test_fleet_dual_shift_preserves_the_argmin(self):
        network = chain_network("lazy")
        pricer = EventPricer(network, columns_per_iter=5)
        duals = {0: 110000.0, 1: 90000.0, 2: 105000.0, 3: 70000.0}
        _rc0, base = pricer.price(duals, 0.0)
        _rc1, shifted = pricer.price(duals, -5000.0)
        self.assertEqual(
            [c["trips"] for c in base], [c["trips"] for c in shifted]
        )
        for left, right in zip(base, shifted):
            self.assertAlmostEqual(
                right["rc_true"] - left["rc_true"], 5000.0, places=6
            )

    def test_network_route_dual_argument_is_unsafe_in_lazy_mode(self):
        """Regression witness for why the pilot never passes ``route_dual``.

        ``_min_reduced_cost_route_lazy`` applies ``-BUS_COST_KX`` at the
        source arc for every objective except artificial-elimination and
        fleet-only, so ``combined-cost`` with a nonzero ``route_dual`` is
        shifted by 1e5 relative to the explicit-arc path.
        """

        duals = {0: 110000.0, 1: 90000.0, 2: 105000.0, 3: 70000.0}
        lazy = chain_network("lazy").min_reduced_cost_route(
            duals, objective="combined-cost", route_dual=-100.0
        )
        explicit = chain_network("explicit").min_reduced_cost_route(
            duals, objective="combined-cost", route_dual=-100.0
        )
        self.assertAlmostEqual(
            explicit["rc"] - lazy["rc"], BUS_COST_KX, places=3
        )


# --------------------------------------------------------------------------
# 2. fixed-route contributions in the master
# --------------------------------------------------------------------------

@needs_gurobi
class MasterReducedCostTests(unittest.TestCase):
    def build(self, fleet_cap=2):
        master = DiveMaster(
            [0, 1, 2, 3], fleet_cap=fleet_cap,
            artificial_penalty=BIG_M_PENALTY, seed=11,
        )
        master.add_routes([
            {"trips": [0, 1], "cost": 100.0},
            {"trips": [2, 3], "cost": 120.0},
            {"trips": [1, 2], "cost": 90.0},
            {"trips": [0], "cost": 200.0},
            {"trips": [3], "cost": 210.0},
        ])
        return master

    def test_gurobi_reduced_cost_equals_dual_recomputation(self):
        master = self.build()
        master.set_fixed([])
        lp = master.solve()
        for key in master.keys:
            direct = (
                master.cost_of(key)
                - sum(lp.trip_duals[t] for t in key)
                - lp.fleet_dual
            )
            self.assertAlmostEqual(lp.reduced_costs[key], direct, places=6)

    def test_fixing_a_route_zeroes_its_covered_trip_duals(self):
        master = self.build(fleet_cap=3)
        master.set_fixed([frozenset({0, 1})])
        lp = master.solve()
        self.assertAlmostEqual(lp.route_values[frozenset({0, 1})], 1.0, 6)
        # With cover semantics the fixed route makes rows 0 and 1 slack, so
        # their duals collapse; that is the complementarity mechanism.
        self.assertAlmostEqual(lp.trip_duals[0], 0.0, places=6)
        self.assertAlmostEqual(lp.trip_duals[1], 0.0, places=6)
        # Reduced costs still reconcile exactly, including the fleet row.
        for key in master.keys:
            direct = (
                master.cost_of(key)
                - sum(lp.trip_duals[t] for t in key)
                - lp.fleet_dual
            )
            self.assertAlmostEqual(lp.reduced_costs[key], direct, places=6)

    def test_fleet_cap_row_is_respected_and_its_dual_is_nonpositive(self):
        master = self.build(fleet_cap=2)
        master.set_fixed([])
        lp = master.solve()
        self.assertLessEqual(sum(lp.route_values.values()), 2.0 + 1e-6)
        self.assertLessEqual(lp.fleet_dual, 1e-9)

    def test_refuses_to_fix_more_routes_than_the_fleet_cap(self):
        master = self.build(fleet_cap=1)
        with self.assertRaises(DivePilotError):
            master.set_fixed([frozenset({0, 1}), frozenset({2, 3})])

    def test_releases_bounds_of_previously_fixed_routes(self):
        master = self.build(fleet_cap=3)
        master.set_fixed([frozenset({0, 1})])
        master.solve()
        master.set_fixed([frozenset({2, 3})])
        lp = master.solve()
        self.assertLess(lp.route_values[frozenset({0, 1})], 1.0 + 1e-6)
        self.assertAlmostEqual(lp.route_values[frozenset({2, 3})], 1.0, 6)


# --------------------------------------------------------------------------
# 3. phase-I infeasibility versus pool shortage
# --------------------------------------------------------------------------

@needs_gurobi
class FeasibilitySemanticsTests(unittest.TestCase):
    def test_pool_shortage_is_repaired_by_pricing_not_called_infeasible(self):
        """The pool cannot cover within the cap; the pricer supplies columns."""

        trips = [0, 1, 2, 3]
        pool = [{"trips": [t], "cost": 1000.0} for t in trips]
        universe = {
            frozenset({0, 1}): ((0, 1), 100.0),
            frozenset({2, 3}): ((2, 3), 100.0),
            **{frozenset({t}): ((t,), 1000.0) for t in trips},
        }
        pilot, master = run_stub_pilot(trips, pool, universe, fleet_cap=2)
        outcomes = [row["outcome"] for row in pilot.node_outcomes]
        self.assertNotIn(dive.NODE_ARTIFICIAL_REMAINS, outcomes)
        self.assertIsNotNone(pilot.integer_solution)
        self.assertEqual(pilot.integer_solution["buses"], 2)
        self.assertGreater(len(pilot.generated), 0)

    def test_penalized_artificials_trigger_heuristic_backtrack(self):
        """No column universe can cover 4 trips with 1 bus: node infeasible."""

        trips = [0, 1, 2, 3]
        pool = [{"trips": [t], "cost": 1000.0} for t in trips]
        universe = {frozenset({t}): ((t,), 1000.0) for t in trips}
        pilot, _master = run_stub_pilot(trips, pool, universe, fleet_cap=1)
        self.assertIn(
            dive.NODE_ARTIFICIAL_REMAINS,
            [row["outcome"] for row in pilot.node_outcomes],
        )
        self.assertIsNone(pilot.integer_solution)
        for row in pilot.node_outcomes:
            self.assertEqual(
                row["scope"], "dive node only; never a global certificate"
            )

    def test_manifest_never_asserts_a_global_certificate(self):
        source = inspect_source()
        self.assertIn('"global_certificate": None', source)


def inspect_source():
    return (REPO / "src" / "diving_pricing_pilot.py").read_text()


# --------------------------------------------------------------------------
# 4. a dive that needs generated columns to reach the integer target
# --------------------------------------------------------------------------

@needs_gurobi
class SyntheticIntegerTargetTests(unittest.TestCase):
    """Six trips, fleet cap 2.

    The frozen pool holds only two-trip routes, so its LP cannot cover six
    trips with route weight two and no integer two-bus cover exists in it.
    The three-trip routes ``{0,1,2}`` and ``{3,4,5}`` exist in the pricer's
    universe and are what the dive must generate.
    """

    TRIPS = [0, 1, 2, 3, 4, 5]
    POOL = [
        {"trips": [0, 1], "cost": 210.0},
        {"trips": [2, 3], "cost": 215.0},
        {"trips": [4, 5], "cost": 220.0},
        {"trips": [1, 2], "cost": 212.0},
        {"trips": [3, 4], "cost": 216.0},
    ] + [{"trips": [t], "cost": 400.0} for t in range(6)]

    UNIVERSE = {
        frozenset({0, 1, 2}): ((0, 1, 2), 300.0),
        frozenset({3, 4, 5}): ((3, 4, 5), 305.0),
        frozenset({0, 1}): ((0, 1), 210.0),
        frozenset({2, 3}): ((2, 3), 215.0),
        frozenset({4, 5}): ((4, 5), 220.0),
        frozenset({1, 2}): ((1, 2), 212.0),
        frozenset({3, 4}): ((3, 4), 216.0),
        **{frozenset({t}): ((t,), 400.0) for t in range(6)},
    }

    def test_frozen_pool_alone_has_no_two_bus_integer_cover(self):
        master = DiveMaster(
            self.TRIPS, fleet_cap=2, artificial_penalty=BIG_M_PENALTY, seed=3
        )
        master.add_routes(self.POOL)
        master.set_fixed([])
        lp = master.solve()
        self.assertGreater(lp.artificial_total, 1e-6)

    def test_dive_generates_the_columns_that_enable_the_integer_target(self):
        pilot, master = run_stub_pilot(
            self.TRIPS, self.POOL, self.UNIVERSE, fleet_cap=2
        )
        self.assertIsNotNone(pilot.integer_solution)
        self.assertEqual(pilot.integer_solution["buses"], 2)
        generated = {frozenset(r["trips"]) for r in pilot.generated}
        self.assertIn(frozenset({0, 1, 2}), generated)
        self.assertIn(frozenset({3, 4, 5}), generated)
        covered = set()
        for route in pilot.integer_solution["routes"]:
            covered |= set(route["trips"])
        self.assertEqual(covered, set(self.TRIPS))

    def test_generated_columns_are_recorded_with_their_reduced_cost(self):
        pilot, _master = run_stub_pilot(
            self.TRIPS, self.POOL, self.UNIVERSE, fleet_cap=2
        )
        for record in pilot.generated:
            self.assertIn("dive_rc_true_at_generation", record)
            self.assertIn("dive_fixed_trip_sets_sha256", record)
            self.assertEqual(record["origin"], "diving_pricing")
            self.assertLess(record["dive_rc_true_at_generation"], 0.0)

    def test_dive_is_deterministic_across_repeats(self):
        first, _ = run_stub_pilot(
            self.TRIPS, self.POOL, self.UNIVERSE, fleet_cap=2
        )
        second, _ = run_stub_pilot(
            self.TRIPS, self.POOL, self.UNIVERSE, fleet_cap=2
        )
        self.assertEqual(
            [row["outcome"] for row in first.node_outcomes],
            [row["outcome"] for row in second.node_outcomes],
        )
        self.assertEqual(
            [sorted(r["trips"]) for r in first.generated],
            [sorted(r["trips"]) for r in second.generated],
        )


@needs_gurobi
class DiveMechanismTests(unittest.TestCase):
    """The case the pilot actually exists to test.

    Two disjoint trip triangles, fleet cap three.  The frozen pool holds only
    the six two-trip routes; its LP optimum is *certified* (minimum reduced
    cost zero over the whole route universe), fractional at route weight
    3.000 with no artificials -- the same signature as the real fresh k=8
    pools.  No three routes from the pool cover all six trips, so the pool's
    integer optimum is four.

    The completing routes ``{0,1,2}`` and ``{3,4,5}`` sit at reduced cost
    **+60** under those certified root duals.  Plain column generation has no
    incentive to produce them; this is the miniature of the witness study's
    "reduced costs up to 53" finding.  Only fixing a fractional route moves
    the duals far enough for them to price out.
    """

    TRIPS = [0, 1, 2, 3, 4, 5]
    POOL = [
        {"trips": [0, 1], "cost": 100.0},
        {"trips": [1, 2], "cost": 100.0},
        {"trips": [0, 2], "cost": 100.0},
        {"trips": [3, 4], "cost": 100.0},
        {"trips": [4, 5], "cost": 100.0},
        {"trips": [3, 5], "cost": 100.0},
    ]
    UNIVERSE = {
        **{
            frozenset(route["trips"]): (tuple(route["trips"]), route["cost"])
            for route in POOL
        },
        frozenset({0, 1, 2}): ((0, 1, 2), 210.0),
        frozenset({3, 4, 5}): ((3, 4, 5), 210.0),
    }
    CAP = 3

    def root_lp(self):
        master = DiveMaster(
            self.TRIPS, fleet_cap=self.CAP,
            artificial_penalty=BIG_M_PENALTY, seed=3,
        )
        master.add_routes(self.POOL)
        master.set_fixed([])
        return master, master.solve()

    def test_pool_alone_cannot_cover_within_the_cap_integrally(self):
        import itertools

        best = min(
            size
            for size in range(1, len(self.POOL) + 1)
            for combination in itertools.combinations(self.POOL, size)
            if set().union(*(set(r["trips"]) for r in combination))
            == set(self.TRIPS)
        )
        self.assertEqual(best, 4)
        self.assertGreater(best, self.CAP)

    def test_root_is_certified_fractional_with_no_artificials(self):
        _master, lp = self.root_lp()
        self.assertAlmostEqual(lp.artificial_total, 0.0, places=6)
        self.assertAlmostEqual(sum(lp.route_values.values()), 3.0, places=6)
        self.assertFalse(
            all(
                abs(v) <= 1e-6 or abs(v - 1.0) <= 1e-6
                for v in lp.route_values.values()
            )
        )
        # Minimum reduced cost over the entire route universe is zero: a
        # genuine pricing certificate, not a truncated search.
        pricer = StubPricer(self.UNIVERSE)
        node_min_rc, _candidates = pricer.price(lp.trip_duals, lp.fleet_dual)
        self.assertAlmostEqual(node_min_rc, 0.0, places=6)

    def test_completing_columns_are_lp_suboptimal_at_the_certified_root(self):
        _master, lp = self.root_lp()
        for trips in ((0, 1, 2), (3, 4, 5)):
            rc = (
                210.0
                - sum(lp.trip_duals[t] for t in trips)
                - lp.fleet_dual
            )
            self.assertAlmostEqual(rc, 60.0, places=6)

    def test_fixing_changes_the_duals_and_generates_the_missing_column(self):
        pilot, _master = run_stub_pilot(
            self.TRIPS, self.POOL, self.UNIVERSE, fleet_cap=self.CAP
        )
        outcomes = [row["outcome"] for row in pilot.node_outcomes]
        self.assertEqual(outcomes[0], dive.NODE_FRACTIONAL)
        self.assertEqual(pilot.node_outcomes[0]["columns_added"], 0)
        self.assertGreater(len(pilot.node_outcomes), 1)
        self.assertGreater(pilot.node_outcomes[1]["depth"], 0)
        self.assertGreater(pilot.node_outcomes[1]["columns_added"], 0)
        generated = {frozenset(r["trips"]) for r in pilot.generated}
        self.assertTrue(
            generated & {frozenset({0, 1, 2}), frozenset({3, 4, 5})}
        )
        self.assertIsNotNone(pilot.integer_solution)
        self.assertEqual(pilot.integer_solution["buses"], self.CAP)
        covered = set()
        for route in pilot.integer_solution["routes"]:
            covered |= set(route["trips"])
        self.assertEqual(covered, set(self.TRIPS))


# --------------------------------------------------------------------------
# 5. budget handling
# --------------------------------------------------------------------------

@needs_gurobi
class BudgetTests(unittest.TestCase):
    def test_exhausted_budget_stops_before_any_node(self):
        pilot, _master = run_stub_pilot(
            [0, 1], [{"trips": [0], "cost": 5.0}, {"trips": [1], "cost": 5.0}],
            {frozenset({0, 1}): ((0, 1), 4.0)}, fleet_cap=2,
            wall_limit_s=0.0,
        )
        self.assertEqual(pilot.stop_reason, "wall_limit")
        self.assertEqual(pilot.node_counter, 0)

    def test_node_limit_bounds_the_search(self):
        pilot, _master = run_stub_pilot(
            [0, 1, 2, 3], [{"trips": [t], "cost": 1000.0} for t in range(4)],
            {frozenset({t}): ((t,), 1000.0) for t in range(4)},
            fleet_cap=1, max_nodes=1,
        )
        self.assertLessEqual(pilot.node_counter, 1)
        self.assertIn(pilot.stop_reason, {"node_limit", "search_exhausted"})

    def test_pricing_iteration_cap_marks_the_node_uncertified(self):
        trips = [0, 1, 2, 3]
        pool = [{"trips": [t], "cost": 1000.0} for t in trips]
        universe = {
            frozenset({0, 1}): ((0, 1), 100.0),
            frozenset({2, 3}): ((2, 3), 100.0),
            **{frozenset({t}): ((t,), 1000.0) for t in trips},
        }
        pilot, _master = run_stub_pilot(
            trips, pool, universe, fleet_cap=2, max_pricing_iters=1,
        )
        self.assertIn(
            dive.NODE_UNCERTIFIED,
            [row["outcome"] for row in pilot.node_outcomes],
        )


# --------------------------------------------------------------------------
# 6. no-witness guards and source immutability
# --------------------------------------------------------------------------

def write_pool(directory, *, records, status_extra=None, name="cg.json"):
    directory = Path(directory)
    journal = directory / f"{name}.columns.jsonl"
    with open(journal, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    status = {
        "csv": "fixture.csv",
        "prices_csv": "hourly_prices_flat.csv",
        "soc_step": 2.5,
        "block_min": 5,
        "g_kwh": 240.0,
        "charge_kw": 240.0,
        "min_soc_frac": 0.0,
        "master_sense": "cover",
        "initial_pool": "singletons",
        "column_pool_treatment": "RAW",
        "trip_ids": sorted({t for r in records for t in r["trips"]}),
        "columns_journal": str(journal),
        "provenance": {"prices_sha256": "f" * 64},
        **(status_extra or {}),
    }
    path = directory / name
    path.write_text(json.dumps(status, indent=1))
    return path, journal


class SourceGuardTests(unittest.TestCase):
    RECORDS = [
        {"trips": [0, 1], "cost": 100.0},
        {"trips": [2], "cost": 200.0},
    ]

    def test_accepts_a_raw_fresh_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path, journal = write_pool(directory, records=self.RECORDS)
            status, resolved, routes, trips = load_source_pool(path)
            self.assertEqual(resolved, journal.resolve())
            self.assertEqual(trips, [0, 1, 2])
            self.assertEqual(len(routes), 2)
            self.assertIs(status.get("witness_augmentation"), None)

    def test_refuses_witness_augmented_status(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _journal = write_pool(
                directory, records=self.RECORDS,
                status_extra={"witness_augmentation": {"track": "x"}},
            )
            with self.assertRaises(DivePilotError):
                load_source_pool(path)

    def test_refuses_validated_seed_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _journal = write_pool(
                directory, records=self.RECORDS,
                status_extra={"validated_seed_routes_sha256": "a" * 64},
            )
            with self.assertRaises(DivePilotError):
                load_source_pool(path)

    def test_refuses_inherited_event_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _journal = write_pool(
                directory, records=self.RECORDS,
                status_extra={"inherited_event_pool_status_sha256": "b" * 64},
            )
            with self.assertRaises(DivePilotError):
                load_source_pool(path)

    def test_refuses_non_raw_pool_treatment(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _journal = write_pool(
                directory, records=self.RECORDS,
                status_extra={"column_pool_treatment": "WARM"},
            )
            with self.assertRaises(DivePilotError):
                load_source_pool(path)

    def test_refuses_a_warm_or_witness_looking_path(self):
        with tempfile.TemporaryDirectory() as directory:
            warm = Path(directory) / "mip_warm"
            warm.mkdir()
            path, _journal = write_pool(warm, records=self.RECORDS)
            with self.assertRaises(DivePilotError):
                load_source_pool(path)

    def test_forbidden_pattern_covers_the_witness_track_names(self):
        for name in (
            "mip_warm", "advisor_witness_columns_20260917", "warm",
            "giro_seed", "cg.warm.json",
        ):
            self.assertTrue(
                dive.FORBIDDEN_PATH_PATTERN.search(name), name
            )
        for name in ("c1_k08", "base", "cg.json", "cases"):
            self.assertIsNone(
                dive.FORBIDDEN_PATH_PATTERN.search(name), name
            )


class InputHashGuardTests(unittest.TestCase):
    """The cache identity is checked against the *status* provenance.

    A wrong ``--data-dir`` would therefore pass every cache check and still
    replay routes against a different instance, so the actual bytes on disk
    must be bound to the frozen pool's recorded provenance.
    """

    CSV = "Practice_Custom_TwoDuty_13301_13302.csv"
    STATIC = ("Ref_dict.csv", "par_ref_dhd.csv", "hourly_prices_flat.csv")

    def sha(self, path):
        return dive.file_sha256(Path(path))

    def make_status(self, directory, *, provenance):
        status = {
            "csv": self.CSV, "prices_csv": "hourly_prices_flat.csv",
            "soc_step": 30.0, "block_min": 30, "g_kwh": 240.0,
            "charge_kw": 240.0, "min_soc_frac": 0.0,
            "trip_ids": [], "provenance": provenance,
        }
        return SimpleNamespace(
            data_dir=Path(directory), rc_eps=1e-4, reserve_s=0.0,
            max_pricing_iters=1, node_time_s=1.0, max_nodes=1,
            max_alternatives=1, max_restarts=0,
        ), status

    def real_provenance(self):
        data = REPO / "data"
        return {
            "instance_sha256": self.sha(data / self.CSV),
            "prices_sha256": self.sha(data / "hourly_prices_flat.csv"),
            "reference_sha256": self.sha(data / "Ref_dict.csv"),
            "deadhead_sha256": self.sha(data / "par_ref_dhd.csv"),
        }

    def test_mismatched_instance_bytes_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in self.STATIC:
                shutil.copyfile(REPO / "data" / name, directory / name)
            # A truncated instance: same name, different bytes.
            source = (REPO / "data" / self.CSV).read_text().splitlines()
            (directory / self.CSV).write_text("\n".join(source[:20]) + "\n")
            args, status = self.make_status(
                directory, provenance=self.real_provenance()
            )
            with self.assertRaises(DivePilotError) as caught:
                dive.resolve_options(args, status, [])
            self.assertIn("instance_sha256", str(caught.exception))

    def test_absent_provenance_hash_is_refused(self):
        args, status = self.make_status(
            REPO / "data",
            provenance={
                key: value for key, value in self.real_provenance().items()
                if key != "deadhead_sha256"
            },
        )
        with self.assertRaises(DivePilotError) as caught:
            dive.resolve_options(args, status, [])
        self.assertIn("deadhead_sha256", str(caught.exception))

    def test_matching_inputs_are_accepted_and_recorded(self):
        args, status = self.make_status(
            REPO / "data", provenance=self.real_provenance()
        )
        # trip_ids is empty, so the trip-identity check is what fails -- by
        # then every input hash has already been verified and recorded.
        with self.assertRaises(DivePilotError) as caught:
            dive.resolve_options(args, status, [])
        self.assertIn("trip ids", str(caught.exception))


class PublicationTests(unittest.TestCase):
    RECORDS = [
        {"trips": [0, 1], "cost": 100.0},
        {"trips": [2], "cost": 200.0},
    ]

    def test_original_journal_bytes_are_preserved_as_a_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "source"
            source_dir.mkdir()
            out_dir = Path(directory) / "dive"
            path, journal = write_pool(source_dir, records=self.RECORDS)
            original_bytes = journal.read_bytes()
            original_status_bytes = path.read_bytes()
            new_record = {
                "trips": [3], "cost": 1.0, "origin": "diving_pricing",
            }
            publication = publish_augmented_pool(
                source_status=json.loads(path.read_text()),
                source_result_path=path,
                source_journal_path=journal,
                source_result_sha256="x" * 64,
                source_journal_sha256="y" * 64,
                generated_records=[new_record],
                out_dir=out_dir,
            )
            augmented = Path(publication["augmented_journal"]).read_bytes()
            self.assertTrue(augmented.startswith(original_bytes))
            self.assertIn(b'"origin":"diving_pricing"', augmented)
            # the source is untouched
            self.assertEqual(journal.read_bytes(), original_bytes)
            self.assertEqual(path.read_bytes(), original_status_bytes)
            status = json.loads(
                Path(publication["augmented_result"]).read_text()
            )
            self.assertEqual(
                status["columns_journal"], publication["augmented_journal"]
            )
            self.assertFalse(
                status["diving_augmentation"]["witness_columns_used"]
            )
            self.assertEqual(
                status["diving_augmentation"]["appended_records"], 1
            )

    def test_augmented_pool_is_readable_by_the_trusted_mip_runner_loader(self):
        from run_exact_pool_mip import load_pool

        with tempfile.TemporaryDirectory() as directory:
            source_dir = Path(directory) / "source"
            source_dir.mkdir()
            path, journal = write_pool(source_dir, records=self.RECORDS)
            status = json.loads(path.read_text())
            status["trip_ids"] = [0, 1, 2, 3]
            path.write_text(json.dumps(status))
            publication = publish_augmented_pool(
                source_status=status,
                source_result_path=path,
                source_journal_path=journal,
                source_result_sha256="x" * 64,
                source_journal_sha256="y" * 64,
                generated_records=[{"trips": [3], "cost": 1.0}],
                out_dir=Path(directory) / "dive",
            )
            loaded_status, routes, trips = load_pool(
                Path(publication["augmented_result"])
            )
            self.assertEqual(trips, [0, 1, 2, 3])
            self.assertEqual(len(routes), 3)


# --------------------------------------------------------------------------
# 7. physical validation of generated routes
# --------------------------------------------------------------------------

@needs_gurobi
class PhysicalValidationTests(unittest.TestCase):
    def test_every_generated_route_replays_against_the_model_graph(self):
        from run_exact_pool_mip import validate_injected_route

        problem = chain_problem(4)
        network = chain_network("lazy", 4)
        trips = list(problem.trips)
        master = DiveMaster(
            trips, fleet_cap=2, artificial_penalty=BIG_M_PENALTY, seed=5
        )
        master.add_routes([{"trips": [t], "cost": 100500.0} for t in trips])
        options = stub_options(physical_replay=True)
        options.problem = problem
        options.g_kwh = 240.0
        options.charge_kw = 240.0
        options.reserve_kwh = 0.0
        options.horizon_min = 1560.0
        options.max_pricing_iters = 5
        pilot = DivePilot(
            master=master, pricer=EventPricer(network, columns_per_iter=5),
            trips=trips, budget=Budget(60.0), journal=io.StringIO(),
            options=options,
        )
        pilot.run()
        self.assertGreater(len(pilot.generated), 0)
        for record in pilot.generated:
            self.assertIsNone(validate_injected_route(
                problem, record, 240.0, 240.0, 0.0, 1560.0,
                arrival_grace_min=0.0,
            ))
            self.assertEqual(record["origin"], "diving_pricing")
            self.assertIn("dive_rc_true_at_generation", record)
            self.assertAlmostEqual(
                float(record["cost"]),
                float(record["expanded_grid_cost"]), places=9,
            )
            self.assertEqual(
                record["cost_semantics"], "expanded_grid_cost"
            )

    def test_a_route_failing_replay_is_refused(self):
        options = stub_options(physical_replay=True)
        options.problem = chain_problem(4)
        options.g_kwh = 240.0
        options.charge_kw = 240.0
        options.reserve_kwh = 0.0
        options.horizon_min = 1560.0
        pilot = DivePilot(
            master=None, pricer=None, trips=[0, 1, 2, 3],
            budget=Budget(None), journal=io.StringIO(), options=options,
        )
        with self.assertRaises(DivePilotError):
            pilot.validate_record({
                "trips": [0, 1],
                "route_nodes": [DEPOT, 0, "NOT_A_NODE", 1, DEPOT],
                "charging_stops": {
                    "stations": [], "cst": [], "cet": [], "kwh": [],
                },
                "cost": 1.0,
            })


# --------------------------------------------------------------------------
# 8. branching determinism and CLI wiring
# --------------------------------------------------------------------------

@needs_gurobi
class BranchingTests(unittest.TestCase):
    def test_candidate_ranking_is_total_and_rule_dependent(self):
        master = DiveMaster(
            [0, 1, 2], fleet_cap=2, artificial_penalty=BIG_M_PENALTY, seed=1
        )
        master.add_routes([
            {"trips": [0, 1], "cost": 100.0},
            {"trips": [1, 2], "cost": 100.0},
            {"trips": [0, 2], "cost": 100.0},
        ])
        master.set_fixed([])
        lp = master.solve()
        ranked = {
            rule: rank_candidates(lp, master, rule)
            for rule in dive.RESTART_RULES
        }
        for rule, keys in ranked.items():
            self.assertEqual(len(keys), len(set(keys)), rule)
            self.assertEqual(
                keys, rank_candidates(lp, master, rule), rule
            )
        with self.assertRaises(DivePilotError):
            rank_candidates(lp, master, "not_a_rule")


class CliWiringTests(unittest.TestCase):
    def test_parser_requires_the_safety_relevant_arguments(self):
        parser = dive.build_parser()
        required = {
            action.dest for action in parser._actions if action.required
        }
        self.assertTrue({"result", "out_dir", "fleet_cap", "wall_limit_s"}
                        <= required)

    def test_cache_bridge_and_build_are_opt_in(self):
        parser = dive.build_parser()
        defaults = {
            action.dest: action.default for action in parser._actions
        }
        self.assertFalse(defaults["cache_commit_bridge"])
        self.assertFalse(defaults["build_network"])
        self.assertFalse(defaults["skip_cache_hash"])
        self.assertIsNone(defaults["event_network_cache"])

    def test_resolve_options_always_enables_physical_replay(self):
        self.assertIn(
            "options.physical_replay = True", inspect_source()
        )


# --------------------------------------------------------------------------
# 9. cache identity bridge
# --------------------------------------------------------------------------

class CacheIdentityTests(unittest.TestCase):
    def identity(self):
        return {
            "schema": "evsp-dr-event-network-cache-v1",
            "git_commit": "a" * 40,
            "instance_sha256": "b" * 64,
            "prices_sha256": "c" * 64,
            "reference_sha256": "d" * 64,
            "deadhead_sha256": "e" * 64,
            "soc_step": 15.0,
            "block_min": 10,
            "g_kwh": 240.0,
            "charge_kw": 240.0,
            "reserve_kwh": 0.0,
            "strict_tariff_coverage": False,
            "event_arc_mode": "lazy",
        }

    def make_cache(self, directory, identity):
        from exact_pricer_expanded import _write_event_network_cache

        network = chain_network("lazy", 3)
        cache = Path(directory) / "network.pkl"
        _write_event_network_cache(cache, network, identity, 1.0)
        return cache, network

    def test_exact_identity_loads(self):
        from diving_cache_identity import load_verified_network

        with tempfile.TemporaryDirectory() as directory:
            identity = self.identity()
            cache, network = self.make_cache(directory, identity)
            loaded, audit = load_verified_network(
                cache, identity, repo=REPO
            )
            self.assertEqual(loaded.metrics(), network.metrics())
            self.assertFalse(audit["commit_bridge_used"])
            self.assertEqual(audit["differing_identity_fields"], [])

    def test_commit_difference_is_refused_without_the_bridge_flag(self):
        from diving_cache_identity import (
            CacheIdentityError, load_verified_network,
        )

        with tempfile.TemporaryDirectory() as directory:
            identity = self.identity()
            cache, _network = self.make_cache(directory, identity)
            wanted = dict(identity, git_commit="f" * 40)
            with self.assertRaises(CacheIdentityError):
                load_verified_network(cache, wanted, repo=REPO)

    def test_physics_difference_is_refused_even_with_the_bridge_flag(self):
        from diving_cache_identity import (
            CacheIdentityError, load_verified_network,
        )

        with tempfile.TemporaryDirectory() as directory:
            identity = self.identity()
            cache, _network = self.make_cache(directory, identity)
            wanted = dict(identity, g_kwh=200.0)
            with self.assertRaises(CacheIdentityError):
                load_verified_network(
                    cache, wanted, repo=REPO, allow_commit_bridge=True
                )

    def test_tampered_pickle_is_refused(self):
        from diving_cache_identity import (
            CacheIdentityError, load_verified_network,
        )

        with tempfile.TemporaryDirectory() as directory:
            identity = self.identity()
            cache, _network = self.make_cache(directory, identity)
            with open(cache, "ab") as handle:
                handle.write(b"\x00")
            with self.assertRaises(CacheIdentityError):
                load_verified_network(cache, identity, repo=REPO)

    def test_method_audit_reports_graph_critical_methods(self):
        from diving_cache_identity import (
            GRAPH_CRITICAL_METHODS, compare_graph_methods,
        )

        head = dive.git_value(REPO, "rev-parse", "HEAD")
        audit = compare_graph_methods(REPO, head)
        self.assertTrue(audit["identical"])
        self.assertEqual(audit["missing_methods"], [])
        self.assertEqual(
            audit["graph_critical_methods"], list(GRAPH_CRITICAL_METHODS)
        )

    def test_method_audit_reports_a_differing_graph_method(self):
        """The negative case for the pilot's only relaxation of identity.

        The method-source comparison is the single place the commit bridge
        loosens the production loader's strict identity check, so it must be
        shown to actually detect a difference.
        """

        from diving_cache_identity import compare_graph_methods

        head = dive.git_value(REPO, "rev-parse", "HEAD")
        original = EventExpandedNetwork._build_arcs

        def _stubbed_build_arcs(self):  # deliberately different source text
            raise NotImplementedError("stub for the audit negative test")

        EventExpandedNetwork._build_arcs = _stubbed_build_arcs
        try:
            audit = compare_graph_methods(REPO, head)
        finally:
            EventExpandedNetwork._build_arcs = original
        self.assertFalse(audit["identical"])
        self.assertIn("_build_arcs", audit["differing_methods"])
        self.assertEqual(audit["missing_methods"], [])
        # and the real class is restored
        self.assertTrue(compare_graph_methods(REPO, head)["identical"])

    def test_method_audit_refuses_a_load_when_a_graph_method_differs(self):
        from diving_cache_identity import (
            CacheIdentityError, load_verified_network,
        )

        original = EventExpandedNetwork._build_arcs

        def _stubbed_build_arcs(self):
            raise NotImplementedError("stub for the audit negative test")

        with tempfile.TemporaryDirectory() as directory:
            identity = self.identity()
            cache, _network = self.make_cache(directory, identity)
            head = dive.git_value(REPO, "rev-parse", "HEAD")
            wanted = dict(identity, git_commit="f" * 40)
            # Pretend the cache was produced at the current HEAD, then make a
            # graph-critical method differ: the bridge must refuse.
            manifest_path = Path(str(cache) + ".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            manifest["identity"] = dict(identity, git_commit=head)
            manifest_path.write_text(json.dumps(manifest))
            EventExpandedNetwork._build_arcs = _stubbed_build_arcs
            try:
                with self.assertRaises(CacheIdentityError) as caught:
                    load_verified_network(
                        cache, wanted, repo=REPO, allow_commit_bridge=True
                    )
            finally:
                EventExpandedNetwork._build_arcs = original
            self.assertIn("commit bridge refused", str(caught.exception))

    def test_warm_named_cache_directories_are_real_and_flagged(self):
        """c3-c5's graph caches are stored under ``nested_warm_*`` paths.

        That is a storage location, not warm data: the pickle is an
        instance+physics graph with no routes in it. The pilot deliberately
        does not run the cache path through the warm/witness refusal that
        guards *column* sources, so the exemption is recorded rather than
        silent, and the guard still rejects those paths as column sources.
        """

        table = json.loads(
            (REPO / "scripts" / "research" / "diving_pricing_20260919"
             / "cases_k08.json").read_text()
        )
        warm_named = {
            case_id: spec["event_network_cache"]
            for case_id, spec in table["cases"].items()
            if any(
                dive.FORBIDDEN_PATH_PATTERN.search(part)
                for part in Path(spec["event_network_cache"]).parts
            )
        }
        # The situation the note describes must actually exist, or the note
        # is stale.
        self.assertTrue(warm_named)
        # Those same paths are still refused as *column* sources.
        for path in warm_named.values():
            with self.assertRaises(DivePilotError):
                dive._refuse_witness_paths([Path(path)])
        # And the runner records the exemption rather than hiding it.
        source = inspect_source()
        self.assertIn('audit["cache_path_warm_named"]', source)
        self.assertIn('audit["cache_contains_columns"] = False', source)

    def test_method_audit_passes_for_the_real_producer_commit(self):
        from diving_cache_identity import compare_graph_methods

        base = dive.git_value(REPO, "rev-parse", "HEAD")
        # e091a4db produced the k=8 caches; it differs from this worktree only
        # in the fixed-sequence replay accelerator, which is not graph
        # critical, so the audit must still pass.
        producer = "e091a4dba549510238507ef5e5367abea958bd30"
        if dive.git_value(REPO, "cat-file", "-t", producer) != "commit":
            self.skipTest("producer commit is unavailable in this checkout")
        audit = compare_graph_methods(REPO, producer)
        self.assertEqual(audit["missing_methods"], [])
        self.assertEqual(audit["differing_methods"], [])
        self.assertTrue(audit["identical"])
        self.assertNotEqual(
            audit["producer_source_sha256"]["src/event_pricer_network.py"],
            audit["consumer_source_sha256"]["src/event_pricer_network.py"],
        )
        self.assertIsNotNone(base)


if __name__ == "__main__":
    unittest.main()
