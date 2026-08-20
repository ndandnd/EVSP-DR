#!/usr/bin/env python3
"""Instrumented direct arc-flow LP/MIP arm for the resolution-cost study."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from arcflow_oracle import (
    build_model,
    build_network,
    gate_g1,
    gate_g4,
    index_active_arcs,
    solve,
)
from durable_io import atomic_write_json
from exact_cg_telemetry import peak_rss_bytes
from exact_pricer_expanded import _block_minutes, _file_sha256


SCHEMA = "evsp-dr-resolution-cost-arcflow-v1"


def _peak_rss_mb():
    return peak_rss_bytes() / (1024.0 * 1024.0)


def _model_size(data, arcs):
    internal = int(np.count_nonzero(arcs.node_row >= 0))
    constraints = internal + len(data.problem.trips) + 1
    counts = np.full(arcs.size, 2, dtype=np.int8)
    counts += arcs.trip >= 0
    counts += arcs.tail == 0
    counts -= (arcs.tail == 0) | (arcs.head == data.network.SINK)
    return {
        "variables": arcs.size,
        "constraints": constraints,
        "nonzeros": int(counts.sum()),
    }


def _remaining(started, limit):
    if limit is None:
        return None
    return max(0.0, float(limit) - (time.perf_counter() - started))


def _limit_hit(memory_limit_mb):
    return (
        memory_limit_mb is not None
        and _peak_rss_mb() >= float(memory_limit_mb)
    )


def run(args):
    started = time.perf_counter()
    output = Path(args.out)
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    payload = {
        "schema": SCHEMA,
        "method_arm": "arc_flow",
        "scope": "exact_for_named_discretized_model_only",
        "csv": args.csv,
        "prices_csv": args.prices_csv,
        "instance_sha256": _file_sha256(
            Path(args.data_dir) / args.csv
        ),
        "g_kwh": args.g_kwh,
        "charge_kw": args.charge_kw,
        "min_soc_frac": args.reserve_kwh / args.g_kwh,
        "soc_step": args.soc_step,
        "block_min": args.block_min,
        "memory_limit_mb": args.memory_limit_mb,
        "time_limit_s": args.time_limit_s,
        "certified": False,
        "integer_proven": False,
        "lp_bound": None,
        "integer_result": None,
        "stop_reason": "initializing",
    }

    def publish(stage):
        payload.update({
            "stage": stage,
            "wall_s": time.perf_counter() - started,
            "peak_rss_mb": _peak_rss_mb(),
        })
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, payload)

    try:
        network_started = time.perf_counter()
        data = build_network(
            args.csv, prices_csv=args.prices_csv,
            soc_step=args.soc_step, block_min=args.block_min,
            g_kwh=args.g_kwh, charge_kw=args.charge_kw,
            reserve_kwh=args.reserve_kwh, data_dir=Path(args.data_dir),
        )
        payload["dag_build_wall_s"] = time.perf_counter() - network_started
        payload["dag_nodes"] = len(data.network.node_meta)
        payload["dag_arcs"] = data.network.n_arcs
        index_started = time.perf_counter()
        arcs = index_active_arcs(data.network)
        payload["arc_index_wall_s"] = time.perf_counter() - index_started
        payload["active_arcs"] = arcs.size
        payload["network_gate"] = gate_g1(data, arcs)
        payload.update(_model_size(data, arcs))
        publish("network_indexed")
        if _limit_hit(args.memory_limit_mb):
            payload["stop_reason"] = "memory"
            publish("stopped")
            return payload

        model_started = time.perf_counter()
        model = build_model(data, arcs)
        payload["model_build_wall_s"] = time.perf_counter() - model_started
        if (
            model.matrix.shape != (
                payload["constraints"], payload["variables"]
            )
            or model.matrix.nnz != payload["nonzeros"]
        ):
            raise AssertionError("precomputed and built model sizes differ")
        publish("model_built")
        if _limit_hit(args.memory_limit_mb):
            payload["stop_reason"] = "memory"
            publish("stopped")
            return payload

        remaining = _remaining(started, args.time_limit_s)
        lp_limit = remaining
        if remaining is not None:
            lp_limit = min(
                remaining,
                float(args.lp_time_limit_s)
                if args.lp_time_limit_s is not None
                else remaining * args.lp_time_fraction,
            )
        lp, _lp_primal = solve(
            model, objective_kind="fleet", integrality="none",
            time_limit_s=lp_limit,
        )
        payload["lp"] = asdict(lp)
        payload["lp_bound"] = lp.vehicles
        payload["lp_wall_s"] = lp.solve_s
        publish("lp_finished")
        if lp.status != "optimal":
            payload["stop_reason"] = (
                "wall_limit" if lp.status == "limit_reached"
                else f"lp_{lp.status}"
            )
            publish("stopped")
            return payload
        if _limit_hit(args.memory_limit_mb):
            payload["stop_reason"] = "memory"
            publish("stopped")
            return payload

        remaining = _remaining(started, args.time_limit_s)
        if remaining is not None and remaining <= 0:
            payload["stop_reason"] = "wall_limit"
            publish("stopped")
            return payload
        mip, primal = solve(
            model, objective_kind="fleet", integrality="all",
            time_limit_s=remaining, mip_rel_gap=args.mip_rel_gap,
        )
        payload["mip"] = asdict(mip)
        payload["mip_wall_s"] = mip.solve_s
        if primal is not None and mip.all_arcs_integral:
            audit, routes = gate_g4(model, primal)
            payload["integer_audit"] = audit
            payload["integer_result"] = mip.vehicles
            payload["integer_route_count"] = len(routes)
        payload["integer_proven"] = bool(
            payload["integer_result"] is not None
            and (
                mip.status == "optimal"
                or math.ceil(lp.vehicles - 1e-7)
                == round(payload["integer_result"])
            )
        )
        payload["certified"] = bool(
            lp.status == "optimal" and payload["integer_proven"]
        )
        payload["stop_reason"] = (
            "certified" if payload["certified"]
            else "wall_limit" if mip.status == "limit_reached"
            else f"mip_{mip.status}"
        )
        publish("complete")
        return payload
    except MemoryError:
        payload["stop_reason"] = "memory"
        payload["memory_stage"] = payload.get("stage")
        publish("stopped")
        return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--prices-csv", default="hourly_prices_flat.csv")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[1] / "data")
    parser.add_argument("--soc-step", type=float, required=True)
    parser.add_argument("--block-min", type=_block_minutes, required=True)
    parser.add_argument("--g-kwh", type=float, required=True)
    parser.add_argument("--charge-kw", type=float, required=True)
    parser.add_argument("--reserve-kwh", type=float, default=0.0)
    parser.add_argument("--time-limit-s", type=float)
    parser.add_argument("--lp-time-limit-s", type=float)
    parser.add_argument("--lp-time-fraction", type=float, default=0.4)
    parser.add_argument("--memory-limit-mb", type=float)
    parser.add_argument("--mip-rel-gap", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.lp_time_fraction <= 1:
        parser.error("--lp-time-fraction must be in (0,1]")
    if args.g_kwh <= 0 or args.charge_kw <= 0 or args.reserve_kwh < 0:
        parser.error("invalid physics")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
