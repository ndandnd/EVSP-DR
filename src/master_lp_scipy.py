"""Gurobi-free restricted-master LP solved with SciPy's HiGHS backend.

This module deliberately accepts an incidence matrix and route-cost vector,
rather than EVSP route dictionaries.  Keeping model construction separate
from route parsing makes the LP backend small, testable, and reusable by the
column-generation runner without importing Gurobi.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Hashable, Iterable, Sequence

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, eye, hstack, issparse


class RestrictedMasterInputError(ValueError):
    """Raised when the restricted-master data are inconsistent."""


class RestrictedMasterSolveError(RuntimeError):
    """Raised when HiGHS does not return an optimal restricted-master LP."""


@dataclass(frozen=True)
class LPBackendMetadata:
    """Solver provenance recorded alongside an LP solution."""

    solver: str
    method: str
    scipy_version: str


@dataclass(frozen=True)
class RestrictedMasterLPResult:
    """Optimal restricted-master LP values and trip-coverage duals."""

    objective: float
    route_values: tuple[float, ...]
    artificial_values: dict[Hashable, float]
    trip_duals: dict[Hashable, float]
    status: str
    solver_status: int
    message: str
    runtime_s: float
    backend: LPBackendMetadata

    @property
    def route_weight(self) -> float:
        return float(sum(self.route_values))

    @property
    def artificial_total(self) -> float:
        return float(sum(self.artificial_values.values()))


def _validated_trip_ids(trip_ids: Sequence[Hashable]) -> tuple[Hashable, ...]:
    trips = tuple(trip_ids)
    if not trips:
        raise RestrictedMasterInputError("trip_ids must contain at least one trip")
    try:
        unique_count = len(set(trips))
    except TypeError as exc:
        raise RestrictedMasterInputError("trip_ids must be hashable") from exc
    if unique_count != len(trips):
        raise RestrictedMasterInputError("trip_ids must be unique")
    return trips


def build_route_incidence(
    trip_ids: Sequence[Hashable],
    route_trip_ids: Sequence[Iterable[Hashable]],
) -> csr_matrix:
    """Build a binary trip-by-route incidence matrix.

    Every route must contain at least one known trip and may not repeat a trip.
    The returned column order matches ``route_trip_ids``.
    """

    trips = _validated_trip_ids(trip_ids)
    trip_position = {trip: row for row, trip in enumerate(trips)}
    rows: list[int] = []
    columns: list[int] = []

    for column, route in enumerate(route_trip_ids):
        route_trips = tuple(route)
        if not route_trips:
            raise RestrictedMasterInputError(
                f"route {column} contains no trips"
            )
        try:
            route_unique = set(route_trips)
        except TypeError as exc:
            raise RestrictedMasterInputError(
                f"route {column} contains an unhashable trip id"
            ) from exc
        if len(route_unique) != len(route_trips):
            raise RestrictedMasterInputError(
                f"route {column} repeats a trip id"
            )
        unknown = [trip for trip in route_trips if trip not in trip_position]
        if unknown:
            raise RestrictedMasterInputError(
                f"route {column} contains trips outside trip_ids: {unknown[:10]}"
            )
        rows.extend(trip_position[trip] for trip in route_trips)
        columns.extend([column] * len(route_trips))

    matrix = coo_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(len(trips), len(route_trip_ids)),
        dtype=float,
    )
    return matrix.tocsr()


def _validated_incidence(
    route_incidence,
    *,
    n_trips: int,
    n_routes: int,
) -> csr_matrix:
    if issparse(route_incidence):
        incidence = route_incidence.tocsr().astype(float)
    else:
        try:
            incidence = csr_matrix(np.asarray(route_incidence, dtype=float))
        except (TypeError, ValueError) as exc:
            raise RestrictedMasterInputError(
                "route_incidence must be a numeric two-dimensional matrix"
            ) from exc

    if incidence.shape != (n_trips, n_routes):
        raise RestrictedMasterInputError(
            "route_incidence shape must be "
            f"({n_trips}, {n_routes}), found {incidence.shape}"
        )
    incidence.sum_duplicates()
    incidence.eliminate_zeros()
    if incidence.data.size:
        if not np.isfinite(incidence.data).all():
            raise RestrictedMasterInputError(
                "route_incidence contains a non-finite coefficient"
            )
        if not np.all(np.isclose(incidence.data, 1.0, atol=1e-12, rtol=0.0)):
            raise RestrictedMasterInputError(
                "route_incidence must contain only binary 0/1 coefficients"
            )
    return incidence


def solve_restricted_master_lp(
    *,
    trip_ids: Sequence[Hashable],
    route_incidence,
    route_costs: Sequence[float],
    artificial_penalty: float,
    method: str = "highs-ds",
    time_limit_s: float | None = None,
    feasibility_tolerance: float = 1e-7,
) -> RestrictedMasterLPResult:
    """Solve the set-covering restricted-master LP.

    The model is

    ``min route_costs @ a + artificial_penalty * sum(q)``

    subject to ``route_incidence @ a + q >= 1`` and ``a, q >= 0``.

    SciPy represents inequalities as ``A_ub x <= b_ub``.  Coverage rows are
    therefore negated, and the economically conventional nonnegative dual for
    the original ``>=`` row is the *negative* of HiGHS' inequality marginal.
    """

    trips = _validated_trip_ids(trip_ids)
    try:
        costs = np.asarray(tuple(route_costs), dtype=float)
    except (TypeError, ValueError) as exc:
        raise RestrictedMasterInputError(
            "route_costs must be a one-dimensional numeric sequence"
        ) from exc
    if costs.ndim != 1:
        raise RestrictedMasterInputError("route_costs must be one-dimensional")
    if not np.isfinite(costs).all():
        raise RestrictedMasterInputError("route_costs contains a non-finite value")
    if np.any(costs < 0):
        raise RestrictedMasterInputError("route_costs must be nonnegative")

    try:
        penalty = float(artificial_penalty)
    except (TypeError, ValueError) as exc:
        raise RestrictedMasterInputError(
            "artificial_penalty must be a positive finite number"
        ) from exc
    if not np.isfinite(penalty) or penalty <= 0:
        raise RestrictedMasterInputError(
            "artificial_penalty must be a positive finite number"
        )
    try:
        tolerance = float(feasibility_tolerance)
    except (TypeError, ValueError) as exc:
        raise RestrictedMasterInputError(
            "feasibility_tolerance must be a positive finite number"
        ) from exc
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise RestrictedMasterInputError(
            "feasibility_tolerance must be a positive finite number"
        )
    time_limit = None
    if time_limit_s is not None:
        try:
            time_limit = float(time_limit_s)
        except (TypeError, ValueError) as exc:
            raise RestrictedMasterInputError(
                "time_limit_s must be a positive finite number when provided"
            ) from exc
        if not np.isfinite(time_limit) or time_limit <= 0:
            raise RestrictedMasterInputError(
                "time_limit_s must be a positive finite number when provided"
            )
    allowed_methods = {"highs", "highs-ds", "highs-ipm"}
    if method not in allowed_methods:
        raise RestrictedMasterInputError(
            f"method must be one of {sorted(allowed_methods)}, found {method!r}"
        )

    incidence = _validated_incidence(
        route_incidence,
        n_trips=len(trips),
        n_routes=len(costs),
    )
    full_incidence = hstack(
        [incidence, eye(len(trips), dtype=float, format="csr")],
        format="csr",
    )
    objective = np.concatenate(
        [costs, np.full(len(trips), penalty, dtype=float)]
    )
    options = {"presolve": True}
    if time_limit is not None:
        options["time_limit"] = time_limit

    started = perf_counter()
    result = linprog(
        objective,
        A_ub=-full_incidence,
        b_ub=-np.ones(len(trips), dtype=float),
        bounds=(0.0, None),
        method=method,
        options=options,
    )
    runtime_s = perf_counter() - started

    status_names = {
        0: "optimal",
        1: "limit_reached",
        2: "infeasible",
        3: "unbounded",
        4: "solver_error",
    }
    status = status_names.get(int(result.status), f"unknown_{result.status}")
    if not result.success or result.x is None:
        raise RestrictedMasterSolveError(
            "SciPy/HiGHS restricted-master LP failed: "
            f"status={status} ({result.status}), message={result.message}"
        )
    if result.ineqlin is None or result.ineqlin.marginals is None:
        raise RestrictedMasterSolveError(
            "SciPy/HiGHS returned no inequality marginals for trip duals"
        )

    route_values_array = np.asarray(result.x[: len(costs)], dtype=float)
    artificial_array = np.asarray(result.x[len(costs) :], dtype=float)
    dual_array = -np.asarray(result.ineqlin.marginals, dtype=float)
    for values in (route_values_array, artificial_array, dual_array):
        values[np.abs(values) < tolerance] = 0.0

    coverage = incidence @ route_values_array + artificial_array
    minimum_coverage = float(np.min(coverage))
    if minimum_coverage < 1.0 - tolerance:
        raise RestrictedMasterSolveError(
            "SciPy/HiGHS returned a coverage-infeasible solution: "
            f"minimum coverage={minimum_coverage}"
        )

    return RestrictedMasterLPResult(
        objective=float(result.fun),
        route_values=tuple(float(value) for value in route_values_array),
        artificial_values={
            trip: float(artificial_array[row])
            for row, trip in enumerate(trips)
        },
        trip_duals={trip: float(dual_array[row]) for row, trip in enumerate(trips)},
        status=status,
        solver_status=int(result.status),
        message=str(result.message),
        runtime_s=runtime_s,
        backend=LPBackendMetadata(
            solver="scipy.optimize.linprog/HiGHS",
            method=method,
            scipy_version=scipy.__version__,
        ),
    )
