#!/usr/bin/env python3
"""License-free integer master for one resolution-cost exact-CG pool."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from durable_io import atomic_write_json, read_jsonl_records
from exact_cg_telemetry import peak_rss_bytes
from exact_pricer_expanded import load_column_pool
from master_lp_scipy import build_route_incidence
from run_exact_pool_mip import validate_final_selected_routes


SCHEMA = "evsp-dr-resolution-cost-pool-mip-v1"


def _peak_rss_mb():
    return peak_rss_bytes() / (1024.0 * 1024.0)


def run(args):
    started = time.perf_counter()
    status_path = Path(args.cg_status)
    status = json.loads(status_path.read_text())
    journal = Path(status["columns_journal"])
    if not journal.is_absolute():
        journal = status_path.parent / journal
    routes = list(load_column_pool(
        read_jsonl_records(journal, repair_trailing=False),
        list(status["trip_ids"]),
    ).values())
    trips = list(status["trip_ids"])
    incidence = build_route_incidence(
        trips, [route["trips"] for route in routes],
    )
    payload = {
        "schema": SCHEMA, "method_arm": "exact_cg",
        "cg_status": str(status_path), "cg_certified": bool(
            status.get("certified", status.get("certified_rc_optimal")),
        ),
        "variables": len(routes), "constraints": len(trips),
        "nonzeros": int(incidence.nnz), "integer_result": None,
        "integer_proven": False, "stop_reason": "initializing",
        "peak_rss_mb": _peak_rss_mb(),
    }
    try:
        result = milp(
            c=np.ones(len(routes)),
            integrality=np.ones(len(routes), dtype=np.uint8),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(
                incidence, np.ones(len(trips)), np.ones(len(trips)),
            ),
            options={
                "time_limit": float(args.time_limit_s),
                "mip_rel_gap": float(args.mip_rel_gap),
            },
        )
        selected = (
            [route for route, value in zip(routes, result.x) if value > 0.5]
            if result.x is not None else []
        )
        if selected:
            counts = Counter(
                trip for route in selected for trip in route["trips"]
            )
            if any(counts[trip] != 1 for trip in trips):
                raise RuntimeError("integer master witness is not a partition")
            validate_final_selected_routes(status, trips, selected)
            payload["integer_result"] = len(selected)
        bound = getattr(result, "mip_dual_bound", None)
        payload.update({
            "solver_status": int(result.status),
            "solver_message": str(result.message),
            "mip_bound": (
                float(bound) if bound is not None and math.isfinite(bound)
                else None
            ),
            "mip_gap": (
                float(result.mip_gap)
                if getattr(result, "mip_gap", None) is not None else None
            ),
        })
        payload["integer_proven"] = bool(
            payload["integer_result"] is not None
            and (
                result.status == 0
                or (
                    payload["mip_bound"] is not None
                    and math.ceil(payload["mip_bound"] - 1e-7)
                    >= payload["integer_result"]
                )
            )
        )
        payload["stop_reason"] = (
            "certified" if payload["integer_proven"]
            else "wall_limit" if result.status == 1
            else "infeasible" if result.status == 2
            else "solver_error"
        )
    except MemoryError:
        payload["stop_reason"] = "memory"
    payload["wall_s"] = time.perf_counter() - started
    payload["peak_rss_mb"] = _peak_rss_mb()
    atomic_write_json(args.out, payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cg-status", type=Path, required=True)
    parser.add_argument("--time-limit-s", type=float, required=True)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
