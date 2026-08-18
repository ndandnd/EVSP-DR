#!/usr/bin/env python3
"""Recover censored MIP checkpoint/final files from a durable pending result."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def write_new(path, payload):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def recover(progress_dir):
    progress = Path(progress_dir).resolve()
    pending = json.loads((progress / "result_pending.json").read_text())
    latest = json.loads((progress / "latest.json").read_text())
    schedule = (pending.get("progress") or {}).get("checkpoint_schedule_s")
    if not isinstance(schedule, list) or pending.get("runtime_s") is None:
        raise ValueError("pending result lacks progress schedule/runtime")
    elapsed = float(pending["runtime_s"])
    for mark in schedule:
        path = progress / f"checkpoint_{int(round(float(mark)/60)):04d}m.json"
        if path.exists():
            continue
        crossed = float(mark) <= elapsed + 1e-9
        payload = dict(latest) if not crossed else {
            "schema": latest.get("schema"),
            "observational_only": True,
            "gurobi_tree_restart_supported": False,
            "stage": None,
            "incumbent_state": "observation_unavailable_due_interruption",
            "incumbent": None,
            "first_feasible_incumbent_s": None,
            "latest_statistics": {
                "statistics_incumbent_fleet": None,
                "fleet_bound": None,
                "objective_bound": None,
                "fleet_gap": None,
                "node_count": None,
                "solution_count": None,
                "stage_elapsed_s": None,
            },
            "latest_statistics_observed_s": None,
            "incumbent_improvements": [],
            "metadata": latest.get("metadata") or {},
            "disabled_reason": latest.get("disabled_reason"),
        }
        payload.update({
            "kind": "checkpoint",
            "checkpoint_elapsed_s": float(mark),
            "observed_total_elapsed_s": elapsed,
            "solver_ended_before_checkpoint": not crossed,
            "recovery": {
                "schema":
                    "evsp-dr-scale-ladder-mip-progress-recovery-v1",
                "source": "result_pending_and_latest",
                "observational_only": True,
                "observation_availability": (
                    "unavailable_interrupted_before_checkpoint_publication"
                    if crossed else "censored_solver_ended_before_mark"
                ),
            },
        })
        write_new(path, payload)
    final = dict(latest)
    final.update({
        "kind": "final",
        "observed_total_elapsed_s": elapsed,
        "final": {
            "status": pending.get("status"),
            "status_name": pending.get("status_name"),
            "incumbent_found": pending.get("incumbent_found"),
            "buses": pending.get("buses"),
            "mip_obj": pending.get("mip_obj"),
            "mip_bound": pending.get("mip_bound"),
            "mip_gap": pending.get("mip_gap"),
            "fleet_proven": pending.get("fleet_proven"),
            "optimal_scope": pending.get("optimal_scope"),
            "route_vector_sha256": (
                (latest.get("incumbent") or {}).get(
                    "route_vector_sha256"
                )
            ),
            "termination_signal": (
                (pending.get("progress") or {}).get("termination_signal")
            ),
        },
        "recovery": {
            "schema": "evsp-dr-scale-ladder-mip-progress-recovery-v1",
            "source": "result_pending_and_latest",
            "observational_only": True,
        },
    })
    write_new(progress / "final.json", final)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    recover(args.progress_dir)
    print("RECOVERED CENSORED MIP PROGRESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
