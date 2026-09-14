#!/usr/bin/env python3
"""Fixed-master capacity pricing diagnostics around the frozen 309d98d2 driver."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_copy(source: Path, destination: Path, expected_sha256: str) -> str:
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"source pool hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists() or destination.exists():
        raise FileExistsError(destination)
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)
    observed = sha256_file(destination)
    if observed != expected_sha256:
        raise ValueError("destination pool copy hash mismatch")
    return observed


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_driver(code_root: Path):
    source = code_root / "src" / "run_capacity_speed_event_cg.py"
    sys.path.insert(0, str(code_root / "src"))
    try:
        spec = importlib.util.spec_from_file_location("fixed_dual_driver", source)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def case_args(case: dict, output: Path, pool: Path, *, selector: str | None = None):
    return SimpleNamespace(
        mode="cg",
        arm=case["arm"],
        instance=Path(case["instance_path"]),
        prices=Path(case["prices_path"]),
        reference_data_dir=Path(case["reference_data_dir"]),
        out=output,
        pool_out=pool,
        pool=None,
        cg_status=None,
        battery_kwh=float(case["battery_kwh"]),
        non_parx_kw=float(case["non_parx_kw"]),
        reserve_kwh=float(case["reserve_kwh"]),
        soc_step=float(case["soc_step_kwh"]),
        block_min=int(case["event_block_min"]),
        capacity_selector=selector or case["capacity_selector"],
        rc_eps=float(case["rc_eps"]),
        max_iters=int(case["max_iters"]),
        cg_wall_s=float(case["cg_wall_s"]),
        resume=True,
        mip_wall_s=0.0,
        mip_gap=1e-4,
        threads=1,
        expected_commit=case["execution_commit"],
        require_clean=True,
    )


def require_inputs(case: dict) -> None:
    checks = {
        case["instance_path"]: case["instance_sha256"],
        case["prices_path"]: case["prices_sha256"],
        str(Path(case["reference_data_dir"]) / "Ref_dict.csv"): case["reference_sha256"],
        str(Path(case["reference_data_dir"]) / "par_ref_dhd.csv"): case["deadhead_sha256"],
        case["source_pool_path"]: case["source_pool_sha256"],
    }
    for name, expected in checks.items():
        observed = sha256_file(Path(name))
        if observed != expected:
            raise ValueError(f"input hash mismatch {name}: {observed} != {expected}")


def verify_checkout(code_root: Path, expected_commit: str) -> None:
    import subprocess
    def git(*parts):
        return subprocess.run(
            ["git", "-C", str(code_root), *parts], text=True,
            capture_output=True, check=False,
        )
    observed = git("rev-parse", "HEAD").stdout.strip()
    if observed != expected_commit:
        raise ValueError(f"checkout commit mismatch: {observed} != {expected_commit}")
    if git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise ValueError("tracked execution checkout is dirty")
    if git("symbolic-ref", "-q", "HEAD").returncode != 1:
        raise ValueError("execution checkout is not detached")


def normalized_dual_payload(problem, lp: dict) -> dict:
    trip = [[str(key), round(float(lp["trip_duals"][key]), 9)] for key in problem.trips]
    capacity = [
        [str(site), int(minute), round(float(value), 9)]
        for (site, minute), value in sorted(lp["capacity_duals"].items())
    ]
    return {"trip_duals": trip, "nonzero_capacity_duals": capacity}


def raw_dual_hash(problem, lp: dict) -> str:
    value = {
        "trip_duals": [[str(key), float(lp["trip_duals"][key]).hex()] for key in problem.trips],
        "nonzero_capacity_duals": [
            [str(site), int(minute), float(dual).hex()]
            for (site, minute), dual in sorted(lp["capacity_duals"].items())
        ],
    }
    return canonical_sha(value)


def observed_dual_hashes(trip_duals: dict, capacity_duals: dict) -> dict:
    normalized = {
        "trip_duals": [[str(key), round(float(value), 9)]
                       for key, value in trip_duals.items()],
        "nonzero_capacity_duals": [
            [str(site), int(minute), round(float(value), 9)]
            for (site, minute), value in sorted(capacity_duals.items())
            if abs(float(value)) > 1e-10
        ],
    }
    raw = {
        "trip_duals": [[str(key), float(value).hex()]
                       for key, value in trip_duals.items()],
        "nonzero_capacity_duals": [
            [str(site), int(minute), float(value).hex()]
            for (site, minute), value in sorted(capacity_duals.items())
            if abs(float(value)) > 1e-10
        ],
    }
    return {
        "normalized_9dp_dual_vector_sha256": canonical_sha(normalized),
        "raw_dual_vector_sha256": canonical_sha(raw),
        "trip_dual_count": len(normalized["trip_duals"]),
        "nonzero_capacity_dual_count": len(normalized["nonzero_capacity_duals"]),
    }


def audit_starting_state(
    driver, code_root: Path, case: dict, pool: Path, log_path: Path,
) -> dict:
    args = case_args(case, log_path.with_suffix(".unused.json"), pool)
    started = time.perf_counter()
    with working_directory(code_root):
        problem = driver.build_problem(
            args.instance.parent, args.instance.name,
            reference_data_dir=args.reference_data_dir,
        )
        prices = driver.load_station_hourly_prices(
            args.prices,
            sorted({driver.base_station_name(site) for site in driver.STATIONS}),
        )
        provenance = driver.provenance(
            args, args.instance, args.prices, args.reference_data_dir,
        )
        expected_checkpoint = driver.checkpoint_id(args, problem, provenance)
        if expected_checkpoint != case["checkpoint_id"]:
            raise ValueError("checkpoint identity mismatch")
        network_started = time.perf_counter()
        network = driver.build_network(args, problem, prices)
        network_wall = time.perf_counter() - network_started
        master = driver.ExactCapacityMaster(
            problem.trips, capacity=driver.ARMS[args.arm]["capacity"],
            threads=1, log_path=log_path,
        )
        routes = driver.load_resume_pool(
            pool, expected_id=expected_checkpoint, trips=problem.trips,
            route_validator=lambda route: driver.validate_injected_route(
                problem, route, args.battery_kwh, args.non_parx_kw,
                args.reserve_kwh, driver.HORIZON_MIN, arrival_grace_min=0.0,
                station_charge_kw=driver.station_power(args.arm),
            ),
        )
        for route in routes:
            master.add_route(route)
        max_found = max(int(route.get("found_iter", 0)) for route in routes)
        if max_found + 1 != args.max_iters:
            raise ValueError(
                f"exact-one-call invariant failed: max found {max_found}, max-iters {args.max_iters}"
            )
        solve_started = time.perf_counter()
        lp = master.solve()
        rmp_wall = time.perf_counter() - solve_started
    normalized = normalized_dual_payload(problem, lp)
    capacity_values = [row[2] for row in normalized["nonzero_capacity_duals"]]
    return {
        "schema": "evsp-dr-fixed-capacity-dual-start-v1",
        "recorded_utc": now(),
        "source_pool_sha256": sha256_file(pool),
        "checkpoint_id": expected_checkpoint,
        "route_count": len(routes),
        "source_max_found_iter": max_found,
        "next_iteration": max_found + 1,
        "dual_vector_sha256": canonical_sha(normalized),
        "normalized_9dp_dual_vector_sha256": canonical_sha(normalized),
        "raw_dual_vector_sha256": raw_dual_hash(problem, lp),
        "dual_vector": normalized,
        "trip_dual_count": len(normalized["trip_duals"]),
        "nonzero_capacity_dual_count": len(capacity_values),
        "capacity_dual_min": min(capacity_values) if capacity_values else None,
        "capacity_dual_max": max(capacity_values) if capacity_values else None,
        "capacity_dual_l1": sum(abs(v) for v in capacity_values),
        "rmp": {
            "objective": float(lp["objective"]),
            "artificial_total": float(lp["artificial_total"]),
            "route_weight": float(lp["route_weight"]),
            "rows": int(lp["rows"]),
            "columns": int(lp["columns"]),
            "nonzeros": int(lp["nonzeros"]),
            "reported_solve_s": float(lp["runtime_s"]),
            "observed_solve_wall_s": rmp_wall,
        },
        "network": network.metrics(),
        "network_build_wall_s": network_wall,
        "audit_wall_s": time.perf_counter() - started,
    }


class ObservedNetwork:
    def __init__(self, network, driver, telemetry_path: Path, case: dict):
        self._network = network
        self._driver = driver
        self._telemetry_path = telemetry_path
        self._case = case
        self._call_count = 0

    def __getattr__(self, name):
        return getattr(self._network, name)

    def min_reduced_cost_route(self, *args, **kwargs):
        self._call_count += 1
        if self._call_count != 1:
            raise RuntimeError(
                f"fixed-dual diagnostic attempted {self._call_count} pricing calls"
            )
        if not args:
            raise ValueError("pricing observer did not receive trip duals")
        actual_hashes = observed_dual_hashes(
            args[0], kwargs.get("capacity_duals") or {},
        )
        started = time.perf_counter()
        record = {
            "schema": "evsp-dr-fixed-capacity-pricing-call-v1",
            "status": "running",
            "started_utc": now(),
            "attempted_calls": 1,
            "case_id": self._case["case_id"],
            "pair_id": self._case["pair_id"],
            "capacity_selector": self._case["capacity_selector"],
            **actual_hashes,
        }
        atomic_json(self._telemetry_path, record)
        try:
            candidate = self._network.min_reduced_cost_route(*args, **kwargs)
        except self._driver.PricingDeadlineExceeded:
            record.update(
                status="censored", stop_reason="pricing_deadline",
                completed_calls=0,
                elapsed_s=time.perf_counter() - started, ended_utc=now(),
                min_reduced_cost=None, candidate_route_key_sha256=None,
                candidate_payload_sha256=None,
            )
            atomic_json(self._telemetry_path, record)
            raise
        route = candidate.get("_event_record") if candidate is not None else None
        record.update(
            status="complete", stop_reason=None, completed_calls=1,
            elapsed_s=time.perf_counter() - started, ended_utc=now(),
            min_reduced_cost=(float(candidate["rc"]) if candidate is not None else None),
            candidate_route_key_sha256=(self._driver.route_key(route) if route else None),
            candidate_payload_sha256=(canonical_sha(route) if route else None),
            candidate_trip_count=(len(route["trips"]) if route else None),
        )
        atomic_json(self._telemetry_path, record)
        return candidate


def run_one_call(
    driver, code_root: Path, case: dict, pool: Path, output: Path,
    telemetry_path: Path,
) -> dict:
    args = case_args(case, output, pool)
    original_build = driver.build_network

    def observed_build(*parts, **options):
        return ObservedNetwork(
            original_build(*parts, **options), driver, telemetry_path, case,
        )

    with working_directory(code_root):
        problem = driver.build_problem(
            args.instance.parent, args.instance.name,
            reference_data_dir=args.reference_data_dir,
        )
        prices = driver.load_station_hourly_prices(
            args.prices,
            sorted({driver.base_station_name(site) for site in driver.STATIONS}),
        )
        provenance = driver.provenance(
            args, args.instance, args.prices, args.reference_data_dir,
        )
        driver.build_network = observed_build
        try:
            return driver.run_cg(args, problem, prices, provenance, output, pool)
        finally:
            driver.build_network = original_build


def validate_endpoint(case: dict, start: dict, call: dict, result: dict) -> dict:
    if start["dual_vector_sha256"] != case["starting_dual_vector_sha256"]:
        raise ValueError("starting dual vector hash mismatch")
    if start["source_pool_sha256"] != case["source_pool_sha256"]:
        raise ValueError("starting pool hash mismatch")
    if start["next_iteration"] != case["max_iters"]:
        raise ValueError("starting next iteration mismatch")
    if (start["raw_dual_vector_sha256"]
            != case["starting_raw_dual_vector_sha256"]
            or start["normalized_9dp_dual_vector_sha256"]
            != case["starting_normalized_9dp_dual_vector_sha256"]):
        raise ValueError("frozen starting raw/normalized dual hash mismatch")
    if call.get("attempted_calls") != 1:
        raise ValueError("diagnostic did not attempt exactly one pricing call")
    if (call.get("raw_dual_vector_sha256")
            != case["starting_raw_dual_vector_sha256"]
            or call.get("normalized_9dp_dual_vector_sha256")
            != case["starting_normalized_9dp_dual_vector_sha256"]):
        raise ValueError("actual pricing-call dual arguments differ from frozen start")
    iterations = result.get("iterations") or []
    if call["status"] == "complete":
        if call.get("completed_calls") != 1 or len(iterations) != 1:
            raise ValueError("completed call/result iteration mismatch")
        if iterations[0]["iteration"] != case["max_iters"]:
            raise ValueError("unexpected completed iteration")
        if not math.isclose(
            float(call["min_reduced_cost"]),
            float(iterations[0]["min_reduced_cost"]), abs_tol=1e-8,
        ):
            raise ValueError("instrumented/driver reduced cost mismatch")
    elif call["status"] == "censored":
        if call.get("completed_calls") != 0 or iterations:
            raise ValueError("censored call incorrectly reported complete iteration")
        if result.get("stop_reason") != "pricing_deadline":
            raise ValueError("censored call lacks pricing_deadline endpoint")
        if result.get("terminal_exact_min_reduced_cost") is not None:
            raise ValueError("censored call claims terminal reduced cost")
        if result.get("certified_rc_optimal"):
            raise ValueError("censored call claims certificate")
    else:
        raise ValueError(f"unknown call status {call.get('status')}")
    return {
        "schema": "evsp-dr-fixed-capacity-pricing-diagnostic-v1",
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "capacity_selector": case["capacity_selector"],
        "starting_dual_vector_sha256": start["dual_vector_sha256"],
        "starting_raw_dual_vector_sha256": start["raw_dual_vector_sha256"],
        "actual_pricing_raw_dual_vector_sha256": call["raw_dual_vector_sha256"],
        "source_pool_sha256": case["source_pool_sha256"],
        "checkpoint_id": case["checkpoint_id"],
        "next_iteration": case["max_iters"],
        "pricing_call": call,
        "driver_endpoint": {
            "status": result["status"],
            "stop_reason": result["stop_reason"],
            "certified_rc_optimal": result["certified_rc_optimal"],
            "terminal_exact_min_reduced_cost": result["terminal_exact_min_reduced_cost"],
            "runtime_s": result["runtime_s"],
            "network_build_s": result["network_build_s"],
            "final": result["final"],
            "result_pool_sha256": result["pool_sha256"],
        },
        "proof_scope": (
            "One exact pricing call at a hash-bound fixed RMP dual vector. "
            "This is not a complete CG run or a full-model certificate unless "
            "the driver itself returns exact_nonnegative_reduced_cost."
        ),
    }


def execute_case(root: Path, case_id: str, attempt_id: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256_file(manifest_path)
    for name, expected in manifest["tooling_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"tooling hash mismatch: {name}")
    case = manifest["cases"][case_id]
    verify_checkout(Path(manifest["code_root"]), case["execution_commit"])
    require_inputs(case)
    attempt = root / "results" / case_id / attempt_id
    attempt.mkdir(parents=True, exist_ok=False)
    state_path = attempt / "worker_status.json"
    state = {
        "schema": "evsp-dr-fixed-capacity-worker-v1", "case_id": case_id,
        "attempt_id": attempt_id, "manifest_sha256": manifest_sha,
        "status": "preflight", "started_utc": now(),
    }
    atomic_json(state_path, state)
    pool = attempt / "pool.jsonl"
    try:
        atomic_copy(Path(case["source_pool_path"]), pool, case["source_pool_sha256"])
        driver = load_driver(Path(manifest["code_root"]))
        start = audit_starting_state(
            driver, Path(manifest["code_root"]), case, pool,
            attempt / "starting_rmp.gurobi.log",
        )
        start.update(
            case_id=case_id, pair_id=case["pair_id"],
            capacity_selector=case["capacity_selector"],
            manifest_sha256=manifest_sha,
            input_sha256={
                "instance": case["instance_sha256"],
                "prices": case["prices_sha256"],
                "reference": case["reference_sha256"],
                "deadhead": case["deadhead_sha256"],
            },
        )
        atomic_json(attempt / "starting_state.json", start)
        state.update(status="pricing", starting_dual_vector_sha256=start["dual_vector_sha256"])
        atomic_json(state_path, state)
        def watchdog(_signum, _frame):
            raise TimeoutError("outer pricing diagnostic watchdog expired")

        previous_alarm = signal.signal(signal.SIGALRM, watchdog)
        signal.setitimer(signal.ITIMER_REAL, float(case["outer_watchdog_s"]))
        try:
            result = run_one_call(
                driver, Path(manifest["code_root"]), case, pool,
                attempt / "cg.json", attempt / "pricing_call.json",
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_alarm)
        call = json.loads((attempt / "pricing_call.json").read_text())
        summary = validate_endpoint(case, start, call, result)
        summary["manifest_sha256"] = manifest_sha
        atomic_json(attempt / "diagnostic.json", summary)
        artifacts = {}
        for name in ("starting_state.json", "pricing_call.json", "cg.json", "diagnostic.json", "pool.jsonl"):
            path = attempt / name
            artifacts[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        final = {
            **state, "status": "finished", "ended_utc": now(),
            "artifacts": artifacts,
            "pricing_status": call["status"],
            "certified_rc_optimal": result["certified_rc_optimal"],
        }
        atomic_json(attempt / "completion.json", final)
        link = root / "results" / case_id / "completion.json"
        temporary = link.with_name(f".{link.name}.tmp.{os.getpid()}")
        temporary.symlink_to((attempt / "completion.json").resolve())
        os.replace(temporary, link)
        atomic_json(state_path, final)
    except BaseException as error:
        state.update(status="failed", ended_utc=now(), error=repr(error))
        atomic_json(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--attempt", required=True)
    args = parser.parse_args()
    execute_case(args.root.resolve(), args.case, args.attempt)


if __name__ == "__main__":
    main()
