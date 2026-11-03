"""Test whether the original root columns can dispatch eight buses, without a MIP start."""
import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
from time import monotonic

from run_recombination import authenticate, sha


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read_root(path, count, expected_hash, trip_count, plugs):
    by_trip = [[] for _ in range(trip_count)]
    by_minute = defaultdict(list)
    terminal = []
    digest = hashlib.sha256()
    from capacity_event_pricer import record_occupancy
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for index, line in enumerate(itertools.islice(stream, count)):
            digest.update(line)
            route = json.loads(line)
            trips = route["trips"]
            if (not trips or len(trips) != len(set(trips))
                    or any(type(t) is not int or not 0 <= t < trip_count for t in trips)):
                raise ValueError("Invalid trip incidence in source route")
            if route.get("physical_realization", {}).get("status") != "valid_event_time_realized":
                raise ValueError("Source pool contains an unverified column")
            for trip in trips:
                by_trip[trip].append(index)
            for (station, minute), units in record_occupancy(route).items():
                if plugs[station] is not None:
                    by_minute[station, minute].append((index, units))
            energy = route["terminal_energy_kwh"]
            if not math.isfinite(energy):
                raise ValueError("Nonfinite source energy")
            terminal.append(energy)
    if len(terminal) != count or digest.hexdigest() != expected_hash:
        raise ValueError("Truncated or unauthenticated root pool")
    return by_trip, by_minute, terminal


def search(by_trip, by_minute, terminal, plugs, fleet_cap, terminal_floor, output):
    import gurobipy as gp
    from gurobipy import GRB
    clock = monotonic()
    with gp.Model("original_root_exact_dispatch") as model:
        model.Params.LogFile = str(output / "fleet.gurobi.log")
        model.Params.Threads = 2
        model.Params.TimeLimit = 180
        model.Params.Seed = 0
        model.Params.SolutionLimit = 1
        model.Params.DualReductions = 0
        buses = model.addVars(len(terminal), vtype=GRB.BINARY)
        for choices in by_trip:
            model.addConstr(gp.quicksum(buses[i] for i in choices) == 1)
        for (station, _), occupancies in by_minute.items():
            model.addConstr(gp.quicksum(n * buses[i] for i, n in occupancies) <= plugs[station])
        model.addConstr(buses.sum() <= fleet_cap)
        model.addConstr(gp.LinExpr(terminal, list(buses.values())) >= terminal_floor)
        model.setObjective(gp.LinExpr(), GRB.MINIMIZE)
        model.update()
        build_seconds = monotonic() - clock
        model.optimize()
        selected = {i for i, x in buses.items() if model.SolCount and x.X > 0.5}
        if selected:
            outcome = "incumbent_requires_replay"
        elif model.Status == GRB.INFEASIBLE:
            outcome = "infeasible_in_original_root_pool"
        elif model.Status == GRB.TIME_LIMIT:
            outcome = "time_limit_no_incumbent"
        else:
            outcome = "other_solver_stop"
        return selected, dict(outcome=outcome, solver_status=int(model.Status),
                              solver_version=list(gp.gurobi.version()), binaries=model.NumVars,
                              rows=model.NumConstrs, nonzeros=model.NumNZs, solutions=model.SolCount,
                              solve_seconds=model.Runtime, model_build_seconds=build_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-fee", type=float, choices=[5], required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    clock = monotonic()
    receipt = dict(arm="unchanged_root_exact_mip", validated=False,
                   manifest_sha256=sha(args.manifest), driver_sha256=sha(__file__),
                   time_limit_seconds=180, threads=2, seed=0, supplied_mip_start=False)
    phase = "authenticate"
    try:
        manifest = json.loads(args.manifest.read_text())
        receipt["case"] = manifest["case"]
        summary, solver, data, inputs = authenticate(manifest)
        sys.path.insert(0, str(solver / "src"))
        count = summary["columns"]
        prefix_hash = summary["columns_sha256"]
        pool = Path(manifest["pool"])
        fleet_target = manifest["target_fleet"]
        if fleet_target != 8:
            raise ValueError("Only the registered eight-bus feasibility control is enabled")
        receipt.update(root_columns=count, root_prefix_sha256=prefix_hash,
                       root_pricing_certified=summary["cg_pricing_certified"],
                       solver_commit=manifest["solver_commit"], input_sha256=manifest["input_sha256"],
                       graph_seconds=summary["graph_seconds"], root_cg_seconds=summary["cg_seconds"])
        with inputs["instance"].open(newline="") as f:
            trip_count = sum(1 for _ in csv.DictReader(f))
        phase = "load_root"
        load_time = monotonic()
        by_trip, by_minute, terminal = read_root(pool, count, prefix_hash, trip_count, summary["plugs"])
        receipt.update(load_seconds=monotonic()-load_time, trip_count=trip_count, plug_minutes=len(by_minute))
        phase = "solve"
        selected, stats = search(by_trip, by_minute, terminal, summary["plugs"], fleet_target,
                                 summary["terminal_target_kwh"], out)
        receipt.update(stats)
        if selected:
            phase = "replay"
            replay_start = monotonic()
            opener = gzip.open if pool.suffix == ".gz" else open
            with opener(pool, "rb") as stream:
                schedules = [json.loads(line) for i, line in enumerate(itertools.islice(stream, count))
                             if i in selected]
            selected_path = out / "selected_routes.json"
            write_json(selected_path, schedules)
            # The registered validator's augmented mode independently replays supplied
            # routes without consulting GIRO duties or requiring a historic MIP incumbent.
            from validate_selection import validate
            replay_summary = dict(summary, fleet_cap=fleet_target,
                                  inputs={"data_dir":str(data),"instance":str(inputs["instance"]),
                                          "tariff":str(inputs["prices"])},
                                  input_hashes={str(path):sha(path) for path in inputs.values()})
            source = out / "replay_summary.json"
            write_json(source, replay_summary)
            proof = dict(arm="existing_root_control", validated=False)
            validate(source, selected_path, "augmented", proof)
            write_json(out / "replay.json", proof)
            if not proof["validated"] or not proof["exact_once"] or proof["grid_plug_overload_minutes"]:
                raise ValueError("Independent replay rejected the selection")
            receipt.update(outcome="validated_candidate", validated=True, fleet=proof["fleet"],
                           exact_once=True, grid_plug_overload_minutes=0, duplicate_service=proof["duplicate_service"],
                           objective_nonfleet=proof["objective_nonfleet"], continuous_nonfleet=proof["continuous_nonfleet"],
                           peak_simultaneous_trips=proof["peak_simultaneous_trips"],
                           selected_sha256=proof["selected_sha256"], replay_seconds=monotonic()-replay_start)
    except Exception as exc:
        receipt.update(outcome="technical_or_validation_error", failure_phase=phase,
                       error_class=type(exc).__name__)
        write_json(out / "private_error.json", {"message":str(exc)})
    receipt["total_elapsed_seconds"] = monotonic()-clock
    write_json(out / "receipt.json", receipt)
    return 3 if receipt["outcome"] in {"technical_or_validation_error", "other_solver_stop"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
