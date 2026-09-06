"""Persistent Gurobi restricted master for exact EVSP column generation.

The model is deliberately small in scope: it owns the trip rows, artificial
variables, and real route columns for one exact restricted master.  Columns
are added incrementally, so the exact-CG loop does not rebuild the LP at every
pricing iteration.  SciPy/HiGHS remains a separate explicit backend for
historical audits and backend comparisons.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Hashable, Iterable, Sequence

from master_lp_scipy import (
    LPBackendMetadata,
    RestrictedMasterInputError,
    RestrictedMasterLPResult,
    RestrictedMasterSolveError,
    _validated_trip_ids,
)


GUROBI_LICENSE_PATH = "/share/apps/software/gurobi/gurobi.lic"


def _import_gurobi():
    try:
        import gurobipy as gp
    except Exception as exc:  # pragma: no cover - depends on host install
        raise RestrictedMasterSolveError(
            "Gurobi backend requires an importable gurobipy installation"
        ) from exc
    return gp


def _require_license_path() -> str:
    path = os.environ.get("GRB_LICENSE_FILE")
    if not path:
        raise RestrictedMasterSolveError(
            "GRB_LICENSE_FILE is not set; refusing to start Gurobi"
        )
    license_path = Path(path).expanduser()
    if not license_path.is_file() or not os.access(license_path, os.R_OK):
        raise RestrictedMasterSolveError(
            f"Gurobi license is missing or unreadable: {license_path}"
        )
    return str(license_path)


def gurobi_preflight() -> dict:
    """Validate the selected license and optimize a tiny model.

    Slurm workers call this before loading a column pool.  A stale inherited
    ``GRB_LICENSE_FILE`` therefore fails before substantive work begins.
    """

    license_path = _require_license_path()
    gp = _import_gurobi()
    model = None
    try:
        model = gp.Model("evsp_gurobi_preflight")
        model.Params.OutputFlag = 0
        model.Params.Threads = 1
        variable = model.addVar(lb=0.0, ub=1.0, obj=1.0)
        model.addConstr(variable >= 0.5)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RestrictedMasterSolveError(
                f"Gurobi preflight did not reach OPTIMAL: status={model.Status}"
            )
        version = ".".join(str(value) for value in gp.gurobi.version())
        return {
            "solver": "gurobi",
            "version": version,
            "license_path": license_path,
            "status": "OPTIMAL",
        }
    except RestrictedMasterSolveError:
        raise
    except Exception as exc:
        raise RestrictedMasterSolveError(
            f"Gurobi preflight failed with license {license_path}: {exc}"
        ) from exc
    finally:
        if model is not None:
            model.dispose()


def _finite_nonnegative(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RestrictedMasterInputError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number < 0.0:
        raise RestrictedMasterInputError(
            f"{label} must be finite and nonnegative"
        )
    return number


class GurobiRestrictedMaster:
    """One persistent exact restricted-master LP.

    Existing route prefixes are immutable.  A changed route identity,
    incidence, or objective coefficient raises before optimization; callers
    must publish a new model if they intentionally change a prior column.
    """

    def __init__(
        self,
        *,
        trip_ids: Sequence[Hashable],
        artificial_penalty: float,
        coverage_sense: str = "partition",
        feasibility_tolerance: float = 1e-6,
        threads: int = 1,
        time_limit_s: float | None = None,
        log_file: str | Path | None = None,
        model_name: str = "exact_restricted_master",
    ) -> None:
        self.trip_ids = _validated_trip_ids(trip_ids)
        self.trip_position = {
            trip: index for index, trip in enumerate(self.trip_ids)
        }
        if coverage_sense not in {"cover", "partition"}:
            raise RestrictedMasterInputError(
                "coverage_sense must be 'cover' or 'partition'"
            )
        self.coverage_sense = coverage_sense
        self.artificial_penalty = _finite_nonnegative(
            artificial_penalty, "artificial_penalty"
        )
        if self.artificial_penalty <= 0.0:
            raise RestrictedMasterInputError(
                "artificial_penalty must be positive"
            )
        self.feasibility_tolerance = _finite_nonnegative(
            feasibility_tolerance, "feasibility_tolerance"
        )
        if self.feasibility_tolerance <= 0.0:
            raise RestrictedMasterInputError(
                "feasibility_tolerance must be positive"
            )
        if not isinstance(threads, int) or isinstance(threads, bool) or threads < 1:
            raise RestrictedMasterInputError("threads must be a positive integer")
        self.threads = threads
        self._gp = _import_gurobi()
        _require_license_path()
        self.model = self._gp.Model(model_name)
        self.model.Params.OutputFlag = 1 if log_file else 0
        self.model.Params.Threads = threads
        self.model.Params.Method = 1
        self.model.Params.BarHomogeneous = 1
        if log_file is not None:
            log_path = Path(log_file).expanduser().resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.Params.LogFile = str(log_path)
        self._trip_constraints = {}
        for trip in self.trip_ids:
            expression = self._gp.LinExpr()
            if coverage_sense == "partition":
                constraint = self.model.addConstr(
                    expression == 1.0, name=f"coverage[{trip}]"
                )
            else:
                constraint = self.model.addConstr(
                    expression >= 1.0, name=f"coverage[{trip}]"
                )
            self._trip_constraints[trip] = constraint
        self._artificial = {}
        self._routes = []
        self._route_indices_by_trip = {trip: [] for trip in self.trip_ids}
        self._closed = False
        self._set_time_limit(time_limit_s)
        for trip in self.trip_ids:
            column = self._gp.Column()
            column.addTerms(1.0, self._trip_constraints[trip])
            self._artificial[trip] = self.model.addVar(
                lb=0.0, ub=self._gp.GRB.INFINITY,
                obj=self.artificial_penalty,
                column=column,
                name=f"artificial[{trip}]",
            )
        self.model.update()

    def _set_time_limit(self, time_limit_s: float | None) -> None:
        if time_limit_s is None:
            self.model.Params.TimeLimit = self._gp.GRB.INFINITY
            return
        value = float(time_limit_s)
        if not math.isfinite(value) or value <= 0.0:
            raise RestrictedMasterInputError(
                "time_limit_s must be positive and finite"
            )
        self.model.Params.TimeLimit = value

    def set_time_limit(self, time_limit_s: float | None) -> None:
        self._set_time_limit(time_limit_s)

    def _normalize_route(
        self, route_trip_ids: Iterable[Hashable], cost: float, index: int
    ) -> tuple[tuple[Hashable, ...], float]:
        route = tuple(route_trip_ids)
        if not route:
            raise RestrictedMasterInputError(f"route {index} contains no trips")
        if len(set(route)) != len(route):
            raise RestrictedMasterInputError(
                f"route {index} repeats a trip id"
            )
        unknown = [trip for trip in route if trip not in self.trip_position]
        if unknown:
            raise RestrictedMasterInputError(
                f"route {index} contains unknown trips: {unknown[:10]}"
            )
        return route, _finite_nonnegative(cost, f"route {index} cost")

    def add_routes(
        self,
        route_trip_ids: Sequence[Iterable[Hashable]],
        route_costs: Sequence[float],
        *,
        allow_lower_cost_rewrites: bool = False,
    ) -> int:
        if len(route_trip_ids) != len(route_costs):
            raise RestrictedMasterInputError(
                "route_trip_ids and route_costs must have equal length"
            )
        normalized = [
            self._normalize_route(route, cost, index)
            for index, (route, cost) in enumerate(zip(route_trip_ids, route_costs))
        ]
        prefix = len(self._routes)
        if len(normalized) < prefix:
            raise RestrictedMasterInputError(
                "route list is shorter than the persistent route prefix"
            )

        rewrites = []
        for index, (route, cost) in enumerate(normalized):
            if index < prefix:
                old_route, old_cost, variable = self._routes[index]
                if route != old_route:
                    raise RestrictedMasterInputError(
                        f"route prefix mismatch at column {index}"
                    )
                if math.isclose(cost, old_cost, rel_tol=0.0, abs_tol=1e-9):
                    continue
                if allow_lower_cost_rewrites and cost < old_cost:
                    rewrites.append((index, cost, variable))
                    continue
                raise RestrictedMasterInputError(
                    f"route prefix cost mismatch at column {index}: "
                    f"old={old_cost} new={cost}"
                )

        for index, cost, variable in rewrites:
            variable.Obj = cost
            route, _, _ = self._routes[index]
            self._routes[index] = (route, cost, variable)

        for index, (route, cost) in enumerate(normalized[prefix:], start=prefix):
            column = self._gp.Column()
            for trip in route:
                column.addTerms(1.0, self._trip_constraints[trip])
            variable = self.model.addVar(
                lb=0.0, ub=self._gp.GRB.INFINITY, obj=cost,
                column=column, name=f"route[{index}]",
            )
            self._routes.append((route, cost, variable))
            for trip in route:
                self._route_indices_by_trip[trip].append(index)
        self.model.update()
        return len(normalized) - prefix

    def sync_routes(self, routes: Sequence[dict]) -> int:
        """Synchronize a route pool, allowing cheaper same-incidence rewrites.

        Exact pricing can rediscover the same ordered trip sequence after a
        cheaper charging realization is found.  Gurobi columns are immutable
        in incidence but their objective coefficient may safely decrease.
        """

        return self.add_routes(
            [route["trips"] for route in routes],
            [route["cost"] for route in routes],
            allow_lower_cost_rewrites=True,
        )

    def _validate_solution(self, route_values, artificial_values) -> tuple[float, float]:
        maximum_row_violation = 0.0
        minimum_primal = min(
            [*route_values, *artificial_values], default=0.0
        )
        maximum_bound_violation = max(0.0, -float(minimum_primal))
        for row, trip in enumerate(self.trip_ids):
            coverage = float(artificial_values[row])
            coverage += sum(
                float(route_values[index])
                for index in self._route_indices_by_trip[trip]
            )
            if self.coverage_sense == "partition":
                maximum_row_violation = max(
                    maximum_row_violation, abs(coverage - 1.0)
                )
            else:
                maximum_row_violation = max(
                    maximum_row_violation, max(0.0, 1.0 - coverage)
                )
        if maximum_bound_violation > self.feasibility_tolerance:
            raise RestrictedMasterSolveError(
                "Gurobi returned a bound-infeasible restricted master"
            )
        if maximum_row_violation > self.feasibility_tolerance:
            raise RestrictedMasterSolveError(
                "Gurobi returned a row-infeasible restricted master: "
                f"violation={maximum_row_violation}"
            )
        return maximum_row_violation, maximum_bound_violation

    def _optimize_with_fallback(self) -> tuple[str, int]:
        statuses = self._gp.GRB
        self.model.Params.Method = 1
        self.model.optimize()
        if self.model.Status == statuses.OPTIMAL:
            return "dual_simplex_method_1", int(self.model.Status)
        if self.model.Status == statuses.TIME_LIMIT:
            raise RestrictedMasterSolveError(
                "Gurobi restricted-master LP reached its time limit"
            )
        numerical = {
            getattr(statuses, "NUMERIC", 12),
            getattr(statuses, "SUBOPTIMAL", 13),
        }
        if self.model.Status not in numerical:
            raise RestrictedMasterSolveError(
                "Gurobi restricted-master LP failed: "
                f"status={self.model.Status}"
            )
        for method, value in (
            ("barrier_crossover", 2),
            ("automatic", 0),
        ):
            self.model.reset()
            self.model.Params.Method = value
            if method == "barrier_crossover":
                self.model.Params.Crossover = 1
            self.model.optimize()
            if self.model.Status == statuses.OPTIMAL:
                return method, int(self.model.Status)
            if self.model.Status == statuses.TIME_LIMIT:
                raise RestrictedMasterSolveError(
                    "Gurobi restricted-master fallback reached its time limit"
                )
        raise RestrictedMasterSolveError(
            "Gurobi restricted-master LP remained non-optimal after numerical "
            "fallbacks"
        )

    def solve(self) -> RestrictedMasterLPResult:
        if self._closed:
            raise RestrictedMasterSolveError("Gurobi restricted master is closed")
        started = time.perf_counter()
        effective_method, solver_status = self._optimize_with_fallback()
        route_values = [float(variable.X) for _, _, variable in self._routes]
        artificial_values = [
            float(self._artificial[trip].X) for trip in self.trip_ids
        ]
        maximum_row_violation, maximum_bound_violation = self._validate_solution(
            route_values, artificial_values
        )
        # Gurobi reports the mathematical row dual.  For both equality rows
        # and the minimization covering rows ``coverage >= 1``, that matches
        # the economically conventional dual exposed by the legacy HiGHS
        # adapter.
        dual_sign = 1.0
        dual_values = [
            dual_sign * float(self._trip_constraints[trip].Pi)
            for trip in self.trip_ids
        ]
        if not all(math.isfinite(value) for value in dual_values):
            raise RestrictedMasterSolveError(
                "Gurobi returned non-finite restricted-master duals"
            )
        objective = sum(
            cost * value for (_, cost, _), value in zip(self._routes, route_values)
        ) + self.artificial_penalty * sum(artificial_values)
        return RestrictedMasterLPResult(
            objective=float(objective),
            route_values=tuple(route_values),
            artificial_values={
                trip: value for trip, value in zip(self.trip_ids, artificial_values)
            },
            trip_duals={
                trip: value for trip, value in zip(self.trip_ids, dual_values)
            },
            status="optimal",
            solver_status=solver_status,
            message="Gurobi optimal restricted-master LP",
            runtime_s=time.perf_counter() - started,
            max_row_violation=maximum_row_violation,
            max_bound_violation=maximum_bound_violation,
            feasibility_tolerance=self.feasibility_tolerance,
            backend=LPBackendMetadata(
                solver="gurobipy/Gurobi",
                method=effective_method,
                solver_version=".".join(
                    str(value) for value in self._gp.gurobi.version()
                ),
                requested_method="dual_simplex_method_1",
                threads=self.threads,
                parameters={
                    "Method": int(self.model.Params.Method),
                    "Threads": self.threads,
                    "CoverageSense": self.coverage_sense,
                    "LicensePath": os.environ.get("GRB_LICENSE_FILE"),
                },
            ),
        )

    def close(self) -> None:
        if not self._closed:
            self.model.dispose()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):  # pragma: no cover - interpreter shutdown behavior
        try:
            self.close()
        except Exception:
            pass
