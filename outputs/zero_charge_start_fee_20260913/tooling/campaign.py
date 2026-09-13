#!/usr/bin/env python3
"""Run and collect authenticated fee0/fee5 EVSP experiments; one treatment per allocation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import sys
import time


LICENSE = "/share/apps/software/gurobi/gurobi.lic"


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_available(path):
    try:
        return read(path) if Path(path).is_file() else {}
    except (OSError, ValueError) as exc:
        return {"read_error": repr(exc)}


def atomic_write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def authenticate(path, expected):
    path = Path(path).resolve()
    observed = digest(path)
    if observed != expected:
        raise ValueError(f"hash mismatch {path}: expected {expected}, got {observed}")
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def code_check(path, expected):
    path = Path(path).resolve()
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    pinned = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", expected + "^{commit}"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    branch = subprocess.run(
        ["git", "-C", str(path), "symbolic-ref", "-q", "HEAD"],
        text=True, capture_output=True,
    )
    if actual != pinned:
        raise ValueError(f"checkout {path}: {actual} != {pinned}")
    if dirty:
        raise ValueError(f"tracked dirty checkout {path}")
    if branch.returncode == 0:
        raise ValueError(f"checkout {path} is not detached: {branch.stdout.strip()}")
    return {"path": str(path), "commit": actual, "detached": True, "tracked_clean": True}


def command_flags(command):
    flags = {}
    for index, value in enumerate(command or []):
        if str(value).startswith("--"):
            flags[value] = (
                command[index + 1]
                if index + 1 < len(command) and not str(command[index + 1]).startswith("--")
                else True
            )
    return flags


def cg_command(manifest, pair, arm, arm_root):
    common = manifest["common"]
    treatment = manifest["arms"][arm]
    source = manifest["inputs"][pair["case_id"]]
    command = [
        manifest["python"], "-u", str(Path(manifest["code"]) / "src/exact_pricer_expanded.py"),
        "--csv", source["csv"], "--prices_csv", "hourly_prices_flat.csv",
        "--time-model", "event", "--event-arc-mode", "lazy",
        "--event-network-cache", source["cache"], "--event-network-cache-mode", "require",
        "--soc-step", str(common["soc_step_kwh"]), "--block-min", str(common["block_minutes"]),
        "--max-iters", "50000", "--columns_per_iter", str(common["columns_per_iter"]),
        "--column-selection", "reduced_cost", "--column-diversity-weight", "0",
        "--column-candidate-multiplier", "4", "--rc-eps", str(common["rc_epsilon"]),
        "--master-sense", "cover", "--master-backend", "gurobi", "--initial-pool", "singletons",
        "--wall-limit-s", str(common["cg_seconds"]), "--checkpoint-every", "25",
        "--g-kwh", str(common["battery_kwh"]), "--charge-kw", str(common["charge_kw"]),
        "--charge-start-cost", str(treatment["charge_start_cost"]),
        "--min-soc-frac", "0", "--inherit-event-pool-from", source["parent_descriptor"],
        "--inherit-event-pool-workers", str(common["inherit_workers"]),
        "--inherit-max-columns", str(treatment["inherit_max_columns"]),
        "--inherit-time-limit-s", str(common["inherit_time_limit_s"]),
        "--phase-telemetry", str(arm_root / "cg" / "phases.jsonl"),
        "--gurobi-log", str(arm_root / "cg" / "gurobi.log"),
        "--out", str(arm_root / "cg.json"),
    ]
    if source.get("cache_source_commit"):
        command += ["--event-network-cache-source-commit", source["cache_source_commit"]]
    if treatment["fixed_sequence_index"]:
        command.append("--fixed-sequence-index")
    if treatment["skip_gurobi_incidence"]:
        command.append("--skip-gurobi-incidence")
    return command


def mip_command(manifest, arm_root, arm):
    common = manifest["common"]
    return [
        manifest["python"], "-u", str(Path(manifest["mip_code"]) / "src/run_exact_pool_mip.py"),
        "--result", str(arm_root / "cg.json"),
        "--data-dir", str(Path(manifest["code"]) / "data"),
        "--reference-data-dir", str(Path(manifest["code"]) / "data"),
        "--cover", "--two-stage", "--timelimit", str(common["mip_seconds"]),
        "--stage1-timelimit", str(common["stage1_seconds"]),
        "--charge-start-cost", str(manifest["arms"][arm]["charge_start_cost"]),
        "--threads", str(common["threads"]), "--mipgap", "0.0001",
        "--gurobi-log", str(arm_root / "mip" / "gurobi.log"),
        "--progress-dir", str(arm_root / "mip" / "progress"),
        "--out", str(arm_root / "mip" / "result.json"),
    ]


def clean_environment(extra=None):
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH", "LM_LICENSE_FILE"):
        env.pop(key, None)
    env.update({
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "GRB_LICENSE_FILE": LICENSE,
    })
    if extra:
        env.update(extra)
    return env


def run_process(command, cwd, process_dir, watchdog_seconds, env, resources):
    process_dir = Path(process_dir)
    process_dir.mkdir(parents=True, exist_ok=False)
    recorded_environment = {key: env[key] for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "PYTHONHASHSEED", "GRB_LICENSE_FILE")}
    for key in ("EVSP_EXPECTED_COMMIT", "EVSP_REQUIRE_DETACHED",
                "EVSP_MIP_EXPECTED_RESULT_SHA256", "EVSP_MIP_EXPECTED_JOURNAL_SHA256"):
        if key in env:
            recorded_environment[key] = env[key]
    record = {
        "argv": [str(value) for value in command], "cwd": str(cwd), "started_utc": now(),
        "watchdog_seconds": watchdog_seconds, "host": socket.getfqdn(), "resources": resources,
        "environment": recorded_environment,
    }
    execution = process_dir / "execution.json"
    atomic_write(execution, record, exclusive=True)
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with (process_dir / "stdout.log").open("xb") as stdout, (process_dir / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=stdout, stderr=stderr, env=env, start_new_session=True
        )
        try:
            process.wait(timeout=watchdog_seconds)
            record["watchdog_triggered"] = False
        except subprocess.TimeoutExpired:
            record["watchdog_triggered"] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    record.update({
        "ended_utc": now(), "wall_seconds": time.monotonic() - started,
        "returncode": process.returncode,
        "rusage": {"user_seconds": usage_after.ru_utime - usage_before.ru_utime,
                   "system_seconds": usage_after.ru_stime - usage_before.ru_stime,
                   "children_maxrss_kib": usage_after.ru_maxrss},
    })
    atomic_write(execution, record)
    return record


def journal_path(status_path, status):
    path = Path(status.get("columns_journal") or str(status_path) + ".columns.jsonl")
    return path if path.is_absolute() else status_path.parent / path


def completed_journal_audit(path):
    journal_hash = hashlib.sha256()
    ordered = hashlib.sha256()
    normalized = []
    count = 0
    with Path(path).open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            journal_hash.update(line)
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise ValueError(f"invalid completed journal line {line_number}: {exc}") from exc
            if record.get("origin") != "inherited_event_pool_replayed_in_child_graph":
                continue
            sequence = record.get("inherited_source_ordered_trip_ids")
            encoded = json.dumps(sequence, separators=(",", ":"), sort_keys=True).encode()
            ordered.update(encoded + b"\n")
            identity = {key: record.get(key) for key in (
                "inherited_source_ordered_trip_ids", "trips", "cost", "route_nodes",
                "charging_stops", "expanded_grid_charging_stops",
                "continuous_realized_charging_blocks", "physical_realization")}
            normalized.append(hashlib.sha256(json.dumps(
                identity, separators=(",", ":"), sort_keys=True
            ).encode()).hexdigest())
            count += 1
    normalized.sort()
    return {
        "journal_sha256": journal_hash.hexdigest(),
        "inherited_records": count,
        "inherited_ordered_sequence_sha256": ordered.hexdigest(),
        "normalized_inherited_pool_sha256": hashlib.sha256(json.dumps(
            normalized, separators=(",", ":")
        ).encode()).hexdigest(),
    }


def cg_pool_gate(execution, status_path, completion=None):
    if execution.get("returncode") != 0:
        return {"allow_mip": False, "reason": "cg_process_failed"}
    if execution.get("watchdog_triggered"):
        return {"allow_mip": False, "reason": "cg_watchdog_triggered"}
    status = read_available(status_path)
    if status.get("read_error") or not status:
        return {"allow_mip": False, "reason": "cg_status_unreadable"}
    final = status.get("final") or {}
    if not isinstance(final.get("iter"), (int, float)) or final["iter"] <= 0:
        return {"allow_mip": False, "reason": "cg_has_no_completed_iteration"}
    if not isinstance(final.get("artificials"), (int, float)) or final["artificials"] != 0:
        return {"allow_mip": False, "reason": "cg_pool_has_artificials"}
    pool = journal_path(Path(status_path), status)
    if not pool.is_file():
        return {"allow_mip": False, "reason": "cg_journal_missing"}
    return {"allow_mip": True, "reason": "valid_finite_pool",
            "cg_certified_rc_optimal": status.get("certified_rc_optimal") is True,
            "cg_certificate_observed": status.get("certified_rc_optimal"),
            "journal": str(pool),
            "journal_sha256": (completion or {}).get("journal_sha256") or digest(pool)}


def authenticate_case(manifest, pair):
    source = manifest["inputs"][pair["case_id"]]
    code = Path(manifest["code"])
    checks = [
        authenticate(code / "data" / source["csv"], source["input_sha256"]),
        authenticate(source["cache"], source["cache_sha256"]),
        authenticate(str(source["cache"]) + ".manifest.json", source["cache_manifest_sha256"]),
        authenticate(source["parent_descriptor"], source["parent_descriptor_sha256"]),
        authenticate(source["parent_journal"], source["parent_journal_sha256"]),
        authenticate(code / "data" / source["parent_csv"], source["parent_csv_sha256"]),
    ]
    checks.extend(authenticate(item["path"], item["sha256"]) for item in manifest.get("static_files", []))
    descriptor = read(source["parent_descriptor"])
    final = descriptor.get("final") or {}
    if final.get("iter", 0) <= 0 or final.get("artificials") != 0:
        raise ValueError("parent descriptor has no usable finite pool")
    if digest(source["parent_journal"]) != source["parent_journal_sha256"]:
        raise ValueError("parent journal changed during authentication")
    recorded_journal = Path(descriptor.get("columns_journal", "")).resolve()
    if recorded_journal != Path(source["parent_journal"]).resolve():
        raise ValueError("parent descriptor does not point to authenticated journal")
    if descriptor.get("csv") != source["parent_csv"]:
        raise ValueError("parent descriptor CSV differs from authenticated parent CSV")
    if (descriptor.get("provenance") or {}).get("instance_sha256") != source["parent_csv_sha256"]:
        raise ValueError("parent descriptor does not bind the authenticated parent CSV")
    return checks


def preflight_command(python):
    script = (
        "import gurobipy as gp; m=gp.Model('controlled_comparison_preflight'); "
        "m.Params.OutputFlag=0; m.Params.Threads=1; "
        "x=m.addVars(2101,vtype=gp.GRB.BINARY); "
        "m.addConstr(gp.quicksum(x.values())>=0); m.setObjective(0); m.optimize(); "
        "assert m.Status==gp.GRB.OPTIMAL, m.Status; print('GUROBI_NATIVE_PREFLIGHT_OK',gp.gurobi.version())"
    )
    return [python, "-c", script]


def arm_measurement_completed(value):
    """A measured solver budget exhaustion is distinct from a broken worker."""
    if value.get("status") == "finished":
        return (value.get("mip_execution", {}).get("returncode") == 0
                and not value.get("mip_execution", {}).get("watchdog_triggered"))
    cg = value.get("cg_execution", {})
    return (value.get("status") == "mip_skipped"
            and value.get("mip_skip_reason") in {
                "cg_has_no_completed_iteration", "cg_pool_has_artificials"}
            and value.get("cg_stop_reason") == "wall_limit"
            and cg.get("returncode") == 0 and not cg.get("watchdog_triggered"))


def worker(args):
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = read(manifest_path)
    pair = manifest["pairs"][args.index]
    token = os.environ.get("SLURM_JOB_ID", "local") + "_r" + os.environ.get("SLURM_RESTART_COUNT", "0")
    attempt = root / "cases" / pair["id"] / token
    attempt.mkdir(parents=True, exist_ok=False)
    state = {"pair": pair, "attempt": token, "order": pair["order"], "started_utc": now(), "status": "running", "arms": {}}
    atomic_write(attempt / "pair_status.json", state, exclusive=True)
    resources = {"cpus": pair["cpus"], "mem": pair["mem"], "slurm_time": pair["slurm_time"]}
    authentication_started_utc = now()
    authentication_started = time.monotonic()
    try:
        identities = {
            "cg_code": code_check(manifest["code"], manifest["cg_commit"]),
            "mip_code": code_check(manifest["mip_code"], manifest["mip_commit"]),
            "assets": authenticate_case(manifest, pair),
        }
        allocation = {"pair": pair, "manifest_sha256": digest(manifest_path), "identities": identities,
                      "authentication_started_utc": authentication_started_utc,
                      "authentication_wall_seconds": time.monotonic() - authentication_started,
                      "host": socket.getfqdn(), "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")}}
        atomic_write(attempt / "allocation.json", allocation, exclusive=True)
        if not Path(LICENSE).is_file():
            raise ValueError(f"Gurobi license missing or unreadable: {LICENSE}")
    except Exception as exc:
        state.update(status="authentication_failed", error=repr(exc), ended_utc=now())
        atomic_write(attempt / "pair_status.json", state)
        return 1
    env = clean_environment()
    preflight = run_process(preflight_command(manifest["python"]), manifest["code"],
                            attempt / "license_preflight", 120, env, resources)
    if preflight["returncode"] != 0 or preflight["watchdog_triggered"]:
        state.update(status="license_preflight_failed", preflight=preflight, ended_utc=now())
        atomic_write(attempt / "pair_status.json", state)
        return 1
    common = manifest["common"]
    required = common["cg_seconds"] + 180 + common["mip_seconds"] + 600
    end_epoch = int(os.environ.get("SLURM_JOB_END_TIME", "0") or "0")
    for arm in pair["order"]:
        if end_epoch and end_epoch - time.time() < required:
            state["arms"][arm] = {"status": "not_started_insufficient_allocation_time",
                                  "required_seconds": required, "remaining_seconds": end_epoch - time.time()}
            atomic_write(attempt / "pair_status.json", state)
            continue
        arm_root = attempt / arm
        arm_root.mkdir(parents=True, exist_ok=False)
        arm_state = {"status": "cg_running", "started_utc": now()}
        state["arms"][arm] = arm_state
        atomic_write(attempt / "pair_status.json", state)
        try:
            cg_execution = run_process(cg_command(manifest, pair, arm, arm_root), manifest["code"],
                                       arm_root / "cg", common["cg_seconds"] + 180, env, resources)
            status = read_available(arm_root / "cg.json")
            completion = {}
            if cg_execution.get("returncode") == 0 and status and not status.get("read_error"):
                pool = journal_path(arm_root / "cg.json", status)
                if pool.is_file():
                    completion = {"status_sha256": digest(arm_root / "cg.json"),
                                  "journal": str(pool),
                                  **completed_journal_audit(pool), "completed_utc": now()}
                    atomic_write(arm_root / "completion.json", completion, exclusive=True)
            gate = cg_pool_gate(cg_execution, arm_root / "cg.json", completion)
            arm_state.update(cg_execution=cg_execution, cg_gate=gate,
                             cg_stop_reason=status.get("stop_reason"),
                             cg_certified_rc_optimal=gate.get("cg_certified_rc_optimal"))
            if not gate["allow_mip"]:
                arm_state.update(status="mip_skipped", mip_skip_reason=gate["reason"], ended_utc=now())
                atomic_write(attempt / "pair_status.json", state)
                continue
            mip_env = clean_environment({
                "EVSP_EXPECTED_COMMIT": manifest["mip_commit"], "EVSP_REQUIRE_DETACHED": "1",
                "EVSP_MIP_EXPECTED_RESULT_SHA256": completion["status_sha256"],
                "EVSP_MIP_EXPECTED_JOURNAL_SHA256": completion["journal_sha256"],
            })
            arm_state["status"] = "mip_running"
            atomic_write(attempt / "pair_status.json", state)
            mip_execution = run_process(mip_command(manifest, arm_root, arm), manifest["mip_code"],
                                        arm_root / "mip", common["mip_seconds"] + 600,
                                        mip_env, resources)
            arm_state.update(status="finished", mip_execution=mip_execution, ended_utc=now())
        except Exception as exc:
            arm_state.update(status="execution_error", error=repr(exc), ended_utc=now())
        atomic_write(attempt / "pair_status.json", state)
    all_success = all(arm_measurement_completed(value)
                      for value in state["arms"].values())
    insufficient = any(value.get("status") == "not_started_insufficient_allocation_time"
                       for value in state["arms"].values())
    state.update(status=("finished" if all_success else
                         "incomplete_allocation_budget" if insufficient else "finished_with_failures"),
                 ended_utc=now())
    atomic_write(attempt / "pair_status.json", state)
    return 0 if all_success else 1


def observe(path, *, hash_content=True):
    path = Path(path)
    result = {"path": str(path), "exists": path.is_file()}
    if not result["exists"]:
        return result
    try:
        stat = path.stat()
        result.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if hash_content:
            result["sha256"] = digest(path)
    except OSError as exc:
        result["read_error"] = repr(exc)
    return result


def scalar_dict(value):
    return {key: item for key, item in (value or {}).items()
            if item is None or isinstance(item, (str, int, float, bool))}


def execution_summary(value):
    keys = ("started_utc", "ended_utc", "wall_seconds", "watchdog_seconds",
            "watchdog_triggered", "returncode", "host", "resources", "rusage")
    result = {key: value[key] for key in keys if key in value}
    result["process_success"] = value.get("returncode") == 0 and value.get("watchdog_triggered") is False
    result["args"] = command_flags(value.get("argv"))
    return result


def charging_metrics(result, fee):
    """Separate electricity, starts and energy using replayed tariff blocks."""
    routes = result.get("selected_routes")
    if not isinstance(routes, list) or not routes or fee is None:
        return {"available": False, "reason": "missing_selected_routes_or_fee"}
    grid_energy = continuous_energy = grid_electricity = continuous_electricity = 0.0
    grid_starts = continuous_starts = 0
    grid_cost = continuous_cost = 0.0
    grid_terminal = continuous_terminal = 0.0
    for route in routes:
        grid = route.get("expanded_grid_charging_stops")
        realized = route.get("charging_stops")
        blocks = route.get("continuous_realized_charging_blocks")
        physical = route.get("physical_realization") or {}
        if not isinstance(grid, dict) or not isinstance(realized, dict) or not isinstance(blocks, list):
            return {"available": False, "reason": "route_lacks_replayed_charge_detail"}
        required = (route.get("expanded_grid_cost"), route.get("continuous_realized_cost"),
                    physical.get("expanded_grid_terminal_soc_kwh"), physical.get("continuous_terminal_soc_kwh"))
        if any(value is None for value in required):
            return {"available": False, "reason": "route_lacks_cost_or_terminal_energy"}
        grid_starts += len(grid.get("cst", []))
        continuous_starts += len(realized.get("cst", []))
        grid_energy += sum(grid.get("kwh", []))
        continuous_energy += sum(realized.get("kwh", []))
        grid_electricity += sum(b["price_per_kwh"] * b["expanded_grid_kwh"] for b in blocks)
        continuous_electricity += sum(b["price_per_kwh"] * b["realized_kwh"] for b in blocks)
        grid_cost += route["expanded_grid_cost"] - 100000.0
        continuous_cost += route["continuous_realized_cost"] - 100000.0
        grid_terminal += physical["expanded_grid_terminal_soc_kwh"]
        continuous_terminal += physical["continuous_terminal_soc_kwh"]
    grid_error = grid_cost - grid_electricity - fee * grid_starts
    continuous_error = continuous_cost - continuous_electricity - fee * continuous_starts
    return dict(available=True, charge_start_cost=fee, buses=len(routes),
                expanded_grid_electricity_cost=grid_electricity,
                continuous_electricity_cost=continuous_electricity,
                expanded_grid_charging_starts=grid_starts,
                continuous_charging_starts=continuous_starts,
                expanded_grid_charging_kwh=grid_energy, continuous_charging_kwh=continuous_energy,
                expanded_grid_terminal_kwh=grid_terminal, continuous_terminal_kwh=continuous_terminal,
                expanded_grid_start_fees=fee * grid_starts, continuous_start_fees=fee * continuous_starts,
                expanded_grid_charging_total=grid_cost, continuous_charging_total=continuous_cost,
                expanded_grid_cost_reconstruction_error=grid_error,
                continuous_cost_reconstruction_error=continuous_error,
                cost_components_reconcile=abs(grid_error) <= 1e-4 and abs(continuous_error) <= 1e-4,
                physical_replay_validated=result.get("physical_replay_validated"),
                scope="selected_routes_before_duplicate_trip_removal; shared_capacity_not_inferred")


def collect(args):
    root = args.root.resolve()
    manifest_path = root / "manifest.json"
    manifest = read_available(manifest_path)
    pairs = {pair["id"]: pair for pair in manifest.get("pairs", [])}
    attempts, cg_rows, mip_rows, pair_status_artifacts = [], [], [], []
    seen = set()
    for status_path in sorted((root / "cases").glob("*/*/pair_status.json")):
        directory = status_path.parent
        pair_id, attempt_id = directory.parent.name, directory.name
        pair, pair_state = pairs.get(pair_id, {}), read_available(status_path)
        seen.add(pair_id)
        attempts.append({"pair_id": pair_id, "attempt": attempt_id,
                         "status": pair_state.get("status", "not_written"),
                         "order": pair_state.get("order", pair.get("order")),
                         "arms": pair_state.get("arms", {}),
                         "allocation": observe(directory / "allocation.json"),
                         "license_preflight": execution_summary(read_available(directory / "license_preflight/execution.json")),
                         "pair_status": observe(status_path)})
        pair_status_artifacts.append(observe(status_path))
        for arm in pair.get("order", []):
            arm_root = directory / arm
            cg_path, mip_path = arm_root / "cg.json", arm_root / "mip/result.json"
            cg_status, mip_status = read_available(cg_path), read_available(mip_path)
            cg_exec = read_available(arm_root / "cg/execution.json")
            mip_exec = read_available(arm_root / "mip/execution.json")
            completion = read_available(arm_root / "completion.json")
            final = cg_status.get("final") or {}
            provenance = cg_status.get("provenance") or {}
            provenance_compact = scalar_dict(provenance)
            provenance_compact["args"] = provenance.get("args")
            source = manifest.get("inputs", {}).get(pair.get("case_id"), {})
            if cg_path.is_file():
                cg_rows.append({
                "case_id": f"{pair_id}/{arm}", "source_case_id": pair.get("case_id"),
                "pair_id": pair_id, "attempt": attempt_id, "arm": arm,
                "charge_start_cost": manifest.get("arms", {}).get(arm, {}).get("charge_start_cost"),
                "contrast": pair.get("contrast"), "repetition": pair.get("repetition"), "order": pair.get("order"),
                "path": str(cg_path), "sha256": completion.get("status_sha256") or observe(cg_path).get("sha256"),
                "process": execution_summary(cg_exec),
                "certified_rc_optimal": cg_status.get("certified_rc_optimal"),
                "proof_scope": {"pricing": cg_status.get("pricing_certificate_scope", provenance.get("pricing_certificate_scope", "unknown")),
                                "finite_pool_mip": None, "physical_validation": None, "giro_target_attainment": None},
                "scalar_final": scalar_dict(final), "final": scalar_dict(final),
                "iterations": cg_status.get("iterations"),
                "attempt_iterations": cg_status.get("attempt_iterations"),
                "history": cg_status.get("history_tail"),
                "phase_paths": {"cg": str(arm_root / "cg/phases.jsonl")},
                "input": {key: source.get(key) for key in ("csv", "input_sha256", "cache_sha256",
                          "cache_manifest_sha256", "parent_descriptor_sha256", "parent_journal_sha256", "target_k")},
                "outputs": {"journal": completion.get("journal"), "journal_sha256": completion.get("journal_sha256"),
                            "inherited_records": completion.get("inherited_records"),
                            "inherited_ordered_sequence_sha256": completion.get("inherited_ordered_sequence_sha256"),
                            "normalized_inherited_pool_sha256": completion.get("normalized_inherited_pool_sha256")},
                "provenance": provenance_compact,
                "args": provenance.get("args") or command_flags(cg_exec.get("argv")),
                "inherited_event_pool_audit": cg_status.get("inherited_event_pool_audit"),
                "stop_reason": cg_status.get("stop_reason"), "runtime_s": cg_status.get("wall_s"),
                "wall_s": cg_status.get("wall_s"), "columns": cg_status.get("columns"),
                "csv": cg_status.get("csv"), "prices_csv": cg_status.get("prices_csv"),
                "soc_step": cg_status.get("soc_step"), "block_min": cg_status.get("block_min"),
                "g_kwh": cg_status.get("g_kwh"), "charge_kw": cg_status.get("charge_kw"),
                "min_soc_frac": cg_status.get("min_soc_frac"),
                "master_sense": cg_status.get("master_sense"),
                "master_backend": cg_status.get("master_backend"),
                "initial_pool": cg_status.get("initial_pool"),
                "pricing_certificate_scope": cg_status.get(
                    "pricing_certificate_scope", provenance.get("pricing_certificate_scope")),
                })
            mip_provenance = mip_status.get("mip_provenance") or {}
            mip_provenance_compact = scalar_dict(mip_provenance)
            mip_provenance_compact["args"] = mip_provenance.get("arguments")
            if mip_path.is_file():
                mip_rows.append({
                "case_id": f"{pair_id}/{arm}", "source_case_id": pair.get("case_id"),
                "pair_id": pair_id, "attempt": attempt_id, "arm": arm,
                "charge_start_cost": manifest.get("arms", {}).get(arm, {}).get("charge_start_cost"),
                "contrast": pair.get("contrast"), "repetition": pair.get("repetition"), "order": pair.get("order"),
                "path": str(mip_path), "sha256": observe(mip_path).get("sha256"),
                "process": execution_summary(mip_exec),
                "proof_scope": {"pricing": mip_status.get("pricing_certificate_scope"),
                                "finite_pool_mip": mip_status.get("optimal_scope"),
                                "mip_bound_scope": mip_status.get("mip_bound_scope"),
                                "physical_validation": mip_status.get("physical_replay_validated"),
                                "giro_target_attainment": None},
                "scalar_final": {key: mip_status.get(key) for key in (
                    "status", "status_name", "mip_obj", "mip_bound", "mip_bound_scope", "mip_gap",
                    "buses", "incumbent_found", "fleet_proven", "runtime_s", "physical_replay_validated")},
                "history": scalar_dict(mip_status.get("two_stage")),
                "phase_paths": {"mip_progress": str(arm_root / "mip/progress")},
                "input": {"cg_result_sha256": mip_status.get("source_result_sha256"),
                          "cg_journal_sha256": mip_status.get("source_journal_sha256")},
                "outputs": {"selected_route_set_sha256": mip_status.get("selected_route_set_sha256"),
                            "selected_charging_block_set_sha256": mip_status.get("selected_charging_block_set_sha256")},
                "provenance": mip_provenance_compact,
                "args": mip_provenance.get("arguments") or command_flags(mip_exec.get("argv")),
                "status": mip_status.get("status"), "status_name": mip_status.get("status_name"),
                "optimal_scope": mip_status.get("optimal_scope"), "mip_obj": mip_status.get("mip_obj"),
                "mip_bound": mip_status.get("mip_bound"), "mip_bound_scope": mip_status.get("mip_bound_scope"),
                "mip_gap": mip_status.get("mip_gap"), "buses": mip_status.get("buses"),
                "incumbent_found": mip_status.get("incumbent_found"),
                "fleet_proven": mip_status.get("fleet_proven"),
                "fleet_bound": mip_status.get("fleet_bound"),
                "charging_cost": mip_status.get("charging_cost"),
                "continuous_realized_charging_cost": mip_status.get("continuous_realized_charging_cost"),
                "runtime_s": mip_status.get("runtime_s"),
                "physical_replay_validated": mip_status.get("physical_replay_validated"),
                "physical_replay_scope": mip_status.get("physical_replay_scope"),
                "duplicate_trip_removal_validated": mip_status.get("duplicate_trip_removal_validated"),
                "cross_route_charger_capacity_validated": mip_status.get("cross_route_charger_capacity_validated"),
                "pool_columns": mip_status.get("pool_columns"),
                "instance": mip_status.get("instance"),
                "partitioning": mip_status.get("partitioning"),
                "overcovered_trips": mip_status.get("overcovered_trips"),
                "gurobi_optimize_wall_s": mip_status.get("gurobi_optimize_wall_s"),
                "source_cg_iterations": mip_status.get("source_cg_iterations"),
                "source_cg_wall_s": mip_status.get("source_cg_wall_s"),
                "pricing_certificate_scope": mip_status.get("pricing_certificate_scope"),
                "two_stage": scalar_dict(mip_status.get("two_stage")),
                "physics": mip_status.get("physics"),
                "charging_comparison_metrics": charging_metrics(mip_status, manifest.get("arms", {}).get(arm, {}).get("charge_start_cost")),
                })
    pending = [{"pair_id": pair_id, "status": "no_attempt"} for pair_id in pairs if pair_id not in seen]
    result = {
        "schema": "evsp-zero-charge-start-fee-v1", "root": str(root), "collected_utc": now(),
        "manifest_identity": {**observe(manifest_path), "schema": manifest.get("schema"),
                              "cg_commit": manifest.get("cg_commit"), "mip_commit": manifest.get("mip_commit")},
        "pairs": list(pairs.values()), "attempts": attempts, "pending": pending,
        "cg": cg_rows, "mip": mip_rows,
        "workflow": {"manifest": observe(manifest_path), "jobs": observe(root / "jobs.json"),
                     "pair_status": pair_status_artifacts},
        "interpretation": "Process success is execution evidence only. CG certificates, finite-pool MIP proof, physical replay, and GIRO attainment are separate.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--root", type=Path, required=True)
    worker_parser.add_argument("--index", type=int, required=True)
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    return worker(arguments) if arguments.action == "worker" else collect(arguments)


if __name__ == "__main__":
    sys.exit(main())
