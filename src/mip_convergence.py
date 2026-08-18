"""Durable, opt-in convergence observations for exact-pool MIP solves.

The JSON files written here are observational checkpoints. They are not
Gurobi tree checkpoints and cannot restart branch-and-bound search.
"""

from __future__ import annotations

import hashlib
import json
import math
import signal
import time
from pathlib import Path
from typing import Callable

from durable_io import atomic_write_json


SCHEMA = "evsp-dr-mip-convergence-v1"


def checkpoint_schedule_s(time_limit_s: float) -> list[float]:
    limit = max(0.0, float(time_limit_s))
    marks = [0.0, 60.0, 300.0, 900.0, 1800.0]
    hour = 3600.0
    while hour <= limit + 1e-9:
        marks.append(hour)
        hour += 3600.0
    return sorted(set(mark for mark in marks if mark <= limit + 1e-9))


def route_vector_hash(indices: list[int]) -> str:
    normalized = sorted(int(index) for index in indices)
    encoded = json.dumps(
        normalized, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class TerminationRequest:
    """Signal-safe flag checked by Gurobi callbacks."""

    def __init__(self):
        self.requested = False
        self.signal_name = None
        self._previous = {}

    def _handle(self, signum, _frame):
        self.requested = True
        try:
            self.signal_name = signal.Signals(signum).name
        except ValueError:
            self.signal_name = str(signum)

    def install(self) -> None:
        for candidate in ("SIGUSR1", "SIGTERM", "SIGINT"):
            signum = getattr(signal, candidate, None)
            if signum is None:
                continue
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()


class MIPProgressRecorder:
    """Cache callback observations and publish sparse atomic checkpoints."""

    def __init__(
        self,
        directory: Path,
        *,
        time_limit_s: float,
        metadata: dict,
        clock: Callable[[], float] = time.monotonic,
        writer: Callable[[Path, dict], None] = atomic_write_json,
    ):
        self.directory = Path(directory).expanduser().resolve()
        self.time_limit_s = float(time_limit_s)
        self.metadata = dict(metadata)
        self.clock = clock
        self.writer = writer
        self.schedule = checkpoint_schedule_s(self.time_limit_s)
        self.started = float(clock())
        self.stage = None
        self.stage_started = self.started
        self.latest_stats = {
            "statistics_incumbent_fleet": None,
            "fleet_bound": None,
            "objective_bound": None,
            "fleet_gap": None,
            "node_count": 0.0,
            "solution_count": 0,
        }
        self.latest_incumbent = None
        self.first_feasible_s = None
        self.incumbent_events = []
        self.incumbent_observations = []
        self.statistics_events = []
        self.stage_events = []
        self.emitted = set()
        self.disabled_reason = None
        self.finalized = False
        self.directory.mkdir(parents=True, exist_ok=False)

    def elapsed_s(self) -> float:
        return max(0.0, float(self.clock()) - self.started)

    def stage_elapsed_s(self) -> float:
        return max(0.0, float(self.clock()) - self.stage_started)

    @staticmethod
    def _finite(value):
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return (
            number
            if math.isfinite(number) and abs(number) < 1e100
            else None
        )

    def transition_stage(self, stage: str, *, elapsed_s=None) -> None:
        if stage not in {"fleet", "cost", "single"}:
            raise ValueError(f"invalid MIP convergence stage: {stage}")
        now_elapsed = (
            self.elapsed_s() if elapsed_s is None else float(elapsed_s)
        )
        if self.stage is not None or now_elapsed > 0.0:
            self.publish_due(now_elapsed)
        self.stage = stage
        self.stage_started = self.started + now_elapsed
        self.latest_stats.update({
            "statistics_incumbent_fleet": None,
            "fleet_bound": None,
            "objective_bound": None,
            "fleet_gap": None,
            "node_count": 0.0,
            "solution_count": 0,
        })
        self.stage_events.append({
            "total_elapsed_s": now_elapsed,
            "stage": stage,
        })
        self.statistics_events.append({
            "total_elapsed_s": now_elapsed,
            "stage": stage,
            "statistics": {
                **self.latest_stats,
                "stage_elapsed_s": 0.0,
            },
        })
        self._publish_latest("stage_transition", now_elapsed)

    def record_initial_incumbent(
        self,
        indices: list[int],
        *,
        objective: float,
        fleet: int,
        kind: str = "initial_mip_start_at_t0",
    ) -> None:
        self._record_incumbent(
            indices,
            fleet=fleet,
            objective=objective,
            elapsed_s=0.0,
            stage_elapsed_s=0.0,
            kind=kind,
        )

    def emit_zero(self) -> None:
        if 0.0 in self.schedule and 0.0 not in self.emitted:
            self._publish_checkpoint(0.0, observed_elapsed_s=0.0)

    def observe_stats(
        self,
        *,
        elapsed_s: float,
        stage_elapsed_s: float,
        statistics_incumbent_fleet=None,
        fleet_bound=None,
        objective_bound=None,
        fleet_gap=None,
        node_count=None,
        solution_count=None,
    ) -> None:
        elapsed = max(0.0, float(elapsed_s))
        updates = {
            "statistics_incumbent_fleet": self._finite(
                statistics_incumbent_fleet
            ),
            "fleet_bound": self._finite(fleet_bound),
            "objective_bound": self._finite(objective_bound),
            "fleet_gap": self._finite(fleet_gap),
            "node_count": self._finite(node_count),
            "solution_count": (
                int(solution_count)
                if solution_count is not None else None
            ),
        }
        for key, value in updates.items():
            if value is not None:
                self.latest_stats[key] = value
        self.latest_stats["stage_elapsed_s"] = max(
            0.0, float(stage_elapsed_s)
        )
        self.statistics_events.append({
            "total_elapsed_s": elapsed,
            "stage": self.stage,
            "statistics": dict(self.latest_stats),
        })
        self.publish_due(elapsed)
        self._publish_latest("statistics", elapsed)

    def record_incumbent(
        self,
        indices: list[int],
        *,
        fleet: int,
        objective: float,
        elapsed_s: float,
        stage_elapsed_s: float,
    ) -> None:
        elapsed = max(0.0, float(elapsed_s))
        # Marks crossed before this MIPSOL must not receive a future incumbent.
        self.publish_due(elapsed)
        previous = self.latest_incumbent
        if previous is None:
            kind = "first_feasible"
        elif int(fleet) < int(previous["fleet"]):
            kind = "fleet_improvement"
        elif (
            int(fleet) == int(previous["fleet"])
            and float(objective) < float(previous["objective"]) - 1e-9
        ):
            kind = "objective_improvement"
        else:
            kind = "incumbent_refresh"
        self._record_incumbent(
            indices,
            fleet=fleet,
            objective=objective,
            elapsed_s=elapsed,
            stage_elapsed_s=stage_elapsed_s,
            kind=kind,
        )

    def _record_incumbent(
        self,
        indices,
        *,
        fleet,
        objective,
        elapsed_s,
        stage_elapsed_s,
        kind,
    ):
        indices = sorted(int(index) for index in indices)
        incumbent = {
            "stage": self.stage,
            "total_elapsed_s": float(elapsed_s),
            "stage_elapsed_s": float(stage_elapsed_s),
            "fleet": int(fleet),
            "objective": float(objective),
            "selected_route_indices": indices,
            "route_vector_sha256": route_vector_hash(indices),
            "event": kind,
        }
        if self.first_feasible_s is None:
            self.first_feasible_s = float(elapsed_s)
        previous = self.latest_incumbent
        should_append = (
            previous is None
            or int(fleet) < int(previous["fleet"])
            or (
                int(fleet) == int(previous["fleet"])
                and float(objective) < float(previous["objective"]) - 1e-9
            )
            or kind in {
                "validated_partition_at_t0", "initial_mip_start_at_t0"
            }
        )
        self.latest_incumbent = incumbent
        self.incumbent_observations.append(dict(incumbent))
        if should_append:
            self.incumbent_events.append(dict(incumbent))
        self._publish_latest(kind, float(elapsed_s))

    def publish_due(self, elapsed_s: float) -> None:
        for mark in self.schedule:
            if mark <= elapsed_s + 1e-9 and mark not in self.emitted:
                self._publish_checkpoint(mark, observed_elapsed_s=elapsed_s)

    def _incumbent_snapshot(self, checkpoint_s: float) -> tuple[str, dict | None]:
        incumbent = next((
            event for event in reversed(self.incumbent_observations)
            if event["total_elapsed_s"] <= checkpoint_s + 1e-9
        ), None)
        if incumbent is None:
            return "no_incumbent_yet", None
        if incumbent["total_elapsed_s"] > checkpoint_s + 1e-9:
            return "no_incumbent_yet", None
        if abs(incumbent["total_elapsed_s"] - checkpoint_s) <= 1e-9:
            state = "current_incumbent_at_checkpoint"
        else:
            state = "reused_most_recent_earlier_incumbent"
        return state, dict(incumbent)

    def _statistics_snapshot(self, checkpoint_s: float) -> tuple[dict, float | None]:
        event = next((
            item for item in reversed(self.statistics_events)
            if item["total_elapsed_s"] <= checkpoint_s + 1e-9
        ), None)
        if event is None:
            return {
                "statistics_incumbent_fleet": None,
                "fleet_bound": None,
                "objective_bound": None,
                "fleet_gap": None,
                "node_count": 0.0,
                "solution_count": 0,
                "stage_elapsed_s": 0.0,
            }, None
        return dict(event["statistics"]), float(event["total_elapsed_s"])

    def _stage_snapshot(self, checkpoint_s: float):
        event = next((
            item for item in reversed(self.stage_events)
            if item["total_elapsed_s"] <= checkpoint_s + 1e-9
        ), None)
        return event["stage"] if event is not None else None

    def _payload(
        self,
        *,
        kind: str,
        checkpoint_s=None,
        observed_elapsed_s=None,
        solver_ended_before_checkpoint=False,
        final=None,
    ) -> dict:
        reference = (
            float(checkpoint_s)
            if checkpoint_s is not None
            else float(observed_elapsed_s or 0.0)
        )
        incumbent_state, incumbent = self._incumbent_snapshot(reference)
        statistics, statistics_observed_s = self._statistics_snapshot(
            reference
        )
        stage = self._stage_snapshot(reference)
        return {
            "schema": SCHEMA,
            "kind": kind,
            "observational_only": True,
            "gurobi_tree_restart_supported": False,
            "stage": stage,
            "checkpoint_elapsed_s": checkpoint_s,
            "observed_total_elapsed_s": observed_elapsed_s,
            "observed_stage_elapsed_s": self.latest_stats.get(
                "stage_elapsed_s", 0.0
            ),
            "solver_ended_before_checkpoint": (
                bool(solver_ended_before_checkpoint)
            ),
            "incumbent_state": incumbent_state,
            "incumbent": incumbent,
            "first_feasible_incumbent_s": self.first_feasible_s,
            "latest_statistics": statistics,
            "latest_statistics_observed_s": statistics_observed_s,
            "incumbent_improvements": list(self.incumbent_events),
            "metadata": dict(self.metadata),
            "disabled_reason": self.disabled_reason,
            "final": final,
        }

    def _safe_write(self, path: Path, payload: dict) -> None:
        if self.disabled_reason is not None:
            return
        try:
            self.writer(path, payload)
        except Exception as exc:  # observational I/O must not change the solve
            self.disabled_reason = f"{type(exc).__name__}: {exc}"
            print(
                "[MIP-PROGRESS] disabling checkpoint output after I/O error: "
                f"{self.disabled_reason}",
                flush=True,
            )

    def _publish_checkpoint(
        self,
        checkpoint_s: float,
        *,
        observed_elapsed_s: float,
        solver_ended_before_checkpoint: bool = False,
    ) -> None:
        minutes = int(round(checkpoint_s / 60.0))
        payload = self._payload(
            kind="checkpoint",
            checkpoint_s=float(checkpoint_s),
            observed_elapsed_s=float(observed_elapsed_s),
            solver_ended_before_checkpoint=solver_ended_before_checkpoint,
        )
        self._safe_write(
            self.directory / f"checkpoint_{minutes:04d}m.json",
            payload,
        )
        self.emitted.add(float(checkpoint_s))
        self._safe_write(self.directory / "latest.json", payload)

    def _publish_latest(self, event: str, elapsed_s: float) -> None:
        payload = self._payload(
            kind="latest",
            observed_elapsed_s=float(elapsed_s),
        )
        payload["latest_event"] = event
        self._safe_write(self.directory / "latest.json", payload)

    def finalize(self, *, elapsed_s: float, final: dict) -> None:
        if self.finalized:
            return
        elapsed = max(0.0, float(elapsed_s))
        self.publish_due(elapsed)
        for mark in self.schedule:
            if mark not in self.emitted:
                self._publish_checkpoint(
                    mark,
                    observed_elapsed_s=elapsed,
                    solver_ended_before_checkpoint=True,
                )
        payload = self._payload(
            kind="final",
            observed_elapsed_s=elapsed,
            final=dict(final),
        )
        self._safe_write(self.directory / "final.json", payload)
        self._safe_write(self.directory / "latest.json", payload)
        self.finalized = True


class GurobiProgressObserver:
    """Translate Gurobi callback state into sparse recorder observations."""

    def __init__(
        self,
        recorder: MIPProgressRecorder,
        *,
        GRB,
        variables,
        routes: list[dict],
        bus_cost: float,
        stage: str,
        fixed_fleet: int | None = None,
        termination: TerminationRequest | None = None,
        statistics_throttle_s: float = 2.0,
    ):
        self.recorder = recorder
        self.GRB = GRB
        self.variables = variables
        if hasattr(variables, "items"):
            entries = list(variables.items())
            try:
                entries.sort(key=lambda item: int(item[0]))
                self.variable_entries = [
                    (int(key), key, variable)
                    for key, variable in entries
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "MIP variables must use integer route-index keys"
                ) from exc
        else:
            self.variable_entries = [
                (index, index, variable)
                for index, variable in enumerate(list(variables))
            ]
        route_indices = [
            route_index
            for route_index, _key, _variable in self.variable_entries
        ]
        if (
            route_indices != list(range(len(routes)))
            or len(self.variable_entries) != len(routes)
        ):
            raise ValueError(
                "MIP variables do not align one-to-one with route indices"
            )
        self.routes = routes
        self.bus_cost = float(bus_cost)
        self.stage = stage
        self.fixed_fleet = fixed_fleet
        self.termination = termination
        self.statistics_throttle_s = float(statistics_throttle_s)
        self.last_statistics_s = -math.inf

    def _selected_route_indices(self, values) -> list[int]:
        if hasattr(values, "items"):
            selected = []
            for route_index, key, variable in self.variable_entries:
                if key in values:
                    value = values[key]
                elif variable in values:
                    value = values[variable]
                else:
                    raise ValueError(
                        f"callback solution lacks route variable {key!r}"
                    )
                if float(value) > 0.5:
                    selected.append(route_index)
            return selected
        sequence = list(values)
        if len(sequence) != len(self.variable_entries):
            raise ValueError(
                "callback solution length differs from route variables"
            )
        return [
            route_index
            for (route_index, _key, _variable), value
            in zip(self.variable_entries, sequence)
            if float(value) > 0.5
        ]

    def _statistics_incumbent_fleet(self, best, *, route_fleet=None):
        if self.stage == "fleet":
            return MIPProgressRecorder._finite(best)
        if self.stage == "cost":
            return self.fixed_fleet
        return route_fleet

    @staticmethod
    def _get(model, callback_api, name, default=None):
        key = getattr(callback_api, name, None)
        if key is None:
            return default
        try:
            return model.cbGet(key)
        except Exception:
            return default

    def __call__(self, model, where):
        callback_api = getattr(self.GRB, "Callback", None)
        if callback_api is None:
            return
        if self.termination is not None and self.termination.requested:
            terminate = getattr(model, "terminate", None)
            if callable(terminate):
                terminate()
        elapsed = self.recorder.elapsed_s()
        stage_elapsed = self.recorder.stage_elapsed_s()
        if where == getattr(callback_api, "MIPSOL", object()):
            try:
                values = model.cbGetSolution(self.variables)
            except Exception:
                values = None
            if values is not None:
                try:
                    indices = self._selected_route_indices(values)
                except (KeyError, TypeError, ValueError):
                    return
                fleet = len(indices)
                objective = float(sum(
                    self.routes[index]["cost"] for index in indices
                ))
                bound = self._get(
                    model, callback_api, "MIPSOL_OBJBND"
                )
                finite_bound = MIPProgressRecorder._finite(bound)
                statistics_incumbent_fleet = (
                    self._statistics_incumbent_fleet(
                        self._get(
                            model, callback_api, "MIPSOL_OBJBST"
                        ),
                        route_fleet=fleet,
                    )
                )
                fleet_bound = (
                    finite_bound if self.stage == "fleet"
                    else self.fixed_fleet
                )
                objective_bound = (
                    None if self.stage == "fleet"
                    else (
                        self.bus_cost * int(self.fixed_fleet) + finite_bound
                        if (
                            finite_bound is not None
                            and self.fixed_fleet is not None
                        )
                        else None
                    )
                )
                gap = (
                    max(
                        0.0,
                        float(statistics_incumbent_fleet)
                        - float(fleet_bound),
                    )
                    / max(1.0, float(statistics_incumbent_fleet))
                    if (
                        statistics_incumbent_fleet is not None
                        and fleet_bound is not None
                    )
                    else None
                )
                self.recorder.observe_stats(
                    elapsed_s=elapsed,
                    stage_elapsed_s=stage_elapsed,
                    statistics_incumbent_fleet=(
                        statistics_incumbent_fleet
                    ),
                    fleet_bound=fleet_bound,
                    objective_bound=objective_bound,
                    fleet_gap=gap,
                    node_count=self._get(
                        model, callback_api, "MIPSOL_NODCNT"
                    ),
                    solution_count=self._get(
                        model, callback_api, "MIPSOL_SOLCNT"
                    ),
                )
                self.recorder.record_incumbent(
                    indices,
                    fleet=fleet,
                    objective=objective,
                    elapsed_s=elapsed,
                    stage_elapsed_s=stage_elapsed,
                )
            return
        mip_where = {
            getattr(callback_api, "MIP", object()): "MIP",
            getattr(callback_api, "MIPNODE", object()): "MIPNODE",
        }
        prefix = mip_where.get(where)
        if prefix is None:
            # POLLING is useful solely for signal-triggered termination.
            return
        if elapsed - self.last_statistics_s < self.statistics_throttle_s:
            return
        self.last_statistics_s = elapsed
        best = self._get(model, callback_api, f"{prefix}_OBJBST")
        bound = self._get(model, callback_api, f"{prefix}_OBJBND")
        finite_bound = MIPProgressRecorder._finite(bound)
        if self.stage == "fleet":
            incumbent_fleet = self._statistics_incumbent_fleet(best)
            fleet_bound = finite_bound
            objective_bound = None
        elif self.stage == "cost":
            incumbent_fleet = self.fixed_fleet
            fleet_bound = self.fixed_fleet
            objective_bound = (
                self.bus_cost * int(self.fixed_fleet) + finite_bound
                if finite_bound is not None and self.fixed_fleet is not None
                else None
            )
        else:
            incumbent_fleet = (
                self.recorder.latest_incumbent or {}
            ).get("fleet")
            fleet_bound = None
            objective_bound = finite_bound
        gap = (
            max(0.0, float(incumbent_fleet) - float(fleet_bound))
            / max(1.0, float(incumbent_fleet))
            if incumbent_fleet is not None and fleet_bound is not None
            else None
        )
        self.recorder.observe_stats(
            elapsed_s=elapsed,
            stage_elapsed_s=stage_elapsed,
            statistics_incumbent_fleet=incumbent_fleet,
            fleet_bound=fleet_bound,
            objective_bound=objective_bound,
            fleet_gap=gap,
            node_count=self._get(
                model, callback_api, f"{prefix}_NODCNT"
            ),
            solution_count=self._get(
                model, callback_api, f"{prefix}_SOLCNT"
            ),
        )
