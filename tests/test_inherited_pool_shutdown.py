"""Process-isolated regressions for importer deadlines and fork signal state.

Each scenario has an outer process-group timeout so the historical hang fails
cleanly without orphaning replay workers or hanging the test runner.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))
import exact_pricer_expanded as exact


class SlowNetwork:
    def fixed_sequence_record(self, trips):
        time.sleep(20)
        return None


class IgnoreTermNetwork:
    def fixed_sequence_record(self, trips):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(20)
        return None


class RaiseOrSlowNetwork:
    def fixed_sequence_record(self, trips):
        if trips == [0]:
            raise RuntimeError("injected replay failure")
        time.sleep(20)
        return None


class PartlySlowNetwork:
    def __init__(self, delegate):
        self.delegate = delegate

    def fixed_sequence_record(self, trips):
        if trips != [0]:
            time.sleep(20)
        return self.delegate.fixed_sequence_record(trips)


def fixture(directory, network, trips=(0,)):
    csv = directory / "instance.csv"
    csv.write_text("count_trip_id,Ordered_Trip_ID\n" + "".join(
        f"{trip},{trip + 10}\n" for trip in trips))
    journal = directory / "pool.jsonl"
    journal.write_text("".join(json.dumps({"trips": [trip], "cost": 100000})
                               + "\n" for trip in trips))
    status = directory / "parent.json"
    status.write_text(json.dumps({
        "csv": str(csv), "trip_ids": list(trips),
        "columns_journal": str(journal),
        "provenance": {"instance_sha256": hashlib.sha256(csv.read_bytes()).hexdigest()},
    }))
    return status, dict(child_csv_path=csv,
                        child_problem=SimpleNamespace(trips=trips),
                        child_network=network, g_kwh=240,
                        charge_kw=240, reserve_kwh=0)


def scenario(name, directory):
    import multiprocessing
    children_before = {child.pid for child in multiprocessing.active_children()}
    parent_handler = lambda *_: None
    signal.signal(signal.SIGTERM, parent_handler)
    phases = []
    network = IgnoreTermNetwork() if name == "ignore_term" else SlowNetwork()
    if name == "exception":
        network = RaiseOrSlowNetwork()
    status, kwargs = fixture(directory, network,
                             (0, 1) if name == "exception" else (0,))
    if name == "partial_cancel":
        from test_event_pricer_network import (
            EventExpandedNetwork, four_trip_chain_problem, prices,
        )
        problem = four_trip_chain_problem()
        real = EventExpandedNetwork(problem, prices(), soc_step=15,
                                    block_min=5, g_kwh=240, charge_kw=240,
                                    reserve_kwh=0)
        status, kwargs = fixture(directory, PartlySlowNetwork(real), problem.trips)
        kwargs["child_problem"] = problem
    started = time.monotonic()
    if name == "prep_cancel":
        kwargs["cancel_requested"] = lambda: True
    elif name in {"partial_cancel", "cancel_without_deadline"}:
        kwargs["cancel_requested"] = lambda: time.monotonic() - started >= .3
    def phase(*args):
        phases.append(args)
        if name == "callback_error" and args[0] == "replay" and args[2] != "started":
            raise RuntimeError("injected telemetry failure")
    kwargs["phase_callback"] = phase
    error = None
    try:
        records, audit = exact.inherited_event_pool_records(
            status, **kwargs, workers=2 if name == "exception" else 1,
            time_limit_s=(30 if name == "exception" else
                          0 if name in {"partial_cancel", "cancel_without_deadline"}
                          else .2),
        )
    except RuntimeError as exc:
        if name not in {"exception", "callback_error"}:
            raise
        error = str(exc)
        records, audit = [], {}
    elapsed = time.monotonic() - started
    assert signal.getsignal(signal.SIGTERM) is parent_handler
    assert exact._INHERITED_EVENT_REPLAY_CONTEXT is None
    assert not ({child.pid for child in multiprocessing.active_children()}
                - children_before), "replay children remained alive"
    return dict(records=records, audit=audit, elapsed=elapsed, error=error,
                phases=phases)


def run_scenario(name, tmp_path):
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), name, str(tmp_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"importer exceeded outer timeout: {name}\n{stdout}\n{stderr}")
    assert process.returncode == 0, stdout + "\n" + stderr
    return json.loads(stdout.strip().splitlines()[-1])


def test_deadline_resets_inherited_sigterm_handler(tmp_path):
    result = run_scenario("deadline", tmp_path)
    assert result["elapsed"] < 2
    assert result["records"] == []
    audit = result["audit"]
    assert audit["import_deadline_reached"]
    assert not audit["import_cancelled"]
    assert not audit["inherited_lp_certificate"]
    assert audit["pool_shutdown"]["shutdown_complete"]
    assert audit["pool_shutdown"]["worker_pids"]
    assert not audit["pool_shutdown"]["forced_kill_pids"]
    assert audit["unprocessed_selected"] == 1
    for field in ("preparation_s", "replay_s", "cleanup_s", "total_import_s"):
        assert audit[field] >= 0
    assert audit["total_import_s"] >= audit["replay_s"]
    assert result["phases"]


def test_deadline_escalates_worker_ignoring_sigterm(tmp_path):
    result = run_scenario("ignore_term", tmp_path)
    assert result["elapsed"] < 5
    assert result["records"] == []
    shutdown = result["audit"]["pool_shutdown"]
    assert shutdown["shutdown_complete"]
    assert shutdown["forced_kill_pids"]
    assert set(shutdown["forced_kill_pids"]) <= set(shutdown["worker_pids"])


def test_worker_exception_aborts_other_pending_replays(tmp_path):
    result = run_scenario("exception", tmp_path)
    assert result["elapsed"] < 2
    assert "injected replay failure" in result["error"]


def test_preparation_cancellation_returns_without_replaying(tmp_path):
    result = run_scenario("prep_cancel", tmp_path)
    assert result["elapsed"] < 1
    assert result["records"] == []
    assert result["audit"]["import_cancelled"]
    assert not result["audit"]["inherited_lp_certificate"]
    assert result["audit"]["pool_shutdown"] is None


def test_cancellation_without_replay_deadline_interrupts_blocking_worker(tmp_path):
    result = run_scenario("cancel_without_deadline", tmp_path)
    assert result["elapsed"] < 2
    assert result["audit"]["import_cancelled"]
    assert not result["audit"]["import_deadline_reached"]
    assert result["audit"]["pool_shutdown"]["shutdown_complete"]


def test_cancellation_retains_already_validated_records(tmp_path):
    result = run_scenario("partial_cancel", tmp_path)
    assert result["elapsed"] < 2
    assert [record["trips"] for record in result["records"]] == [[0]]
    assert result["records"][0]["physical_realization"]["status"] == (
        "valid_event_time_realized")
    assert result["audit"]["accepted_columns"] == 1
    assert result["audit"]["import_cancelled"]
    assert not result["audit"]["inherited_lp_certificate"]
    assert result["audit"]["unprocessed_selected"] == 3


def test_phase_callback_error_cannot_bypass_worker_cleanup(tmp_path):
    result = run_scenario("callback_error", tmp_path)
    assert result["elapsed"] < 2
    assert result["error"] is None
    assert any("injected telemetry failure" in item["error"]
               for item in result["audit"]["telemetry_errors"])
    assert result["audit"]["pool_shutdown"]["shutdown_complete"]


if __name__ == "__main__":
    print(json.dumps(scenario(sys.argv[1], Path(sys.argv[2]))))
