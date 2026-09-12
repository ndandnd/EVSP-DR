"""Exact replay equivalence, cache compatibility, and bounded importer checks."""
import itertools
import json
import multiprocessing
import pickle
from copy import deepcopy
from unittest import mock

import pytest

from test_event_pricer_network import (
    EventExpandedNetwork, STATIONS, four_trip_chain_problem, prices,
    two_trip_problem,
)
from test_inherited_event_pool import sha256, write_instance
import exact_pricer_expanded as exact


def network(problem=None, mode="lazy", step=15, tariff=None):
    return EventExpandedNetwork(
        problem or four_trip_chain_problem(), tariff or prices(),
        soc_step=step, block_min=5, g_kwh=240, charge_kw=240,
        reserve_kwh=0, arc_mode=mode,
    )


@pytest.mark.parametrize("mode", ["explicit", "lazy"])
@pytest.mark.parametrize("case", ["chain", "tight", "ties", "empty_trip"])
def test_exhaustive_records_and_realized_actions(mode, case):
    p, tariff, step = four_trip_chain_problem(), prices(), 15
    if case in {"tight", "ties"}:
        p, step = two_trip_problem(191.2), 2.5
    if case == "ties":
        p.end_min[0], p.start_min[1], p.end_min[1] = 59.75, 120, 130
        p.adjacency[0].append((STATIONS[1], 0., 0., "trip_station"))
        p.adjacency[STATIONS[1]] = list(p.adjacency[STATIONS[0]])
        for curve in tariff.values():
            curve.update({0: .3, 1: .05, 2: .4})
    if case == "empty_trip":
        p.trip_energy[2] = 241
    n = network(p, mode, step, tariff)
    sequences = [()] + [seq for length in range(1, 5)
                             for seq in itertools.product(p.trips, repeat=length)]
    sequences += [(999999,), (p.trips[0], 999999)]
    original_record = n._record
    traces = []

    def record(actions):
        traces.append(deepcopy(actions))
        return original_record(actions)

    with mock.patch.object(n, "_record", side_effect=record):
        baseline = [n.fixed_sequence_record(seq) for seq in sequences]
        baseline_traces = deepcopy(traces)
        traces.clear()
        n._selected_action_cache.clear()
        n._window_cache.clear()
        n.set_fixed_sequence_index()
        indexed = [n.fixed_sequence_record(seq) for seq in sequences]
    assert indexed == baseline
    assert traces == baseline_traces
    # The subset includes all SOC targets, in precisely original row order.
    for source in range(len(n.node_meta)):
        for successor in (*p.trips, None):
            expected = [(target, cost) for target, cost in n._iter_arcs(source)
                        if (target == n.SINK if successor is None else
                            n.node_meta[target][0] == "trip" and
                            n.node_meta[target][1] == successor)]
            assert list(n._iter_sequence_arcs(source, successor)) == expected


@pytest.mark.parametrize("mode", ["explicit", "lazy"])
def test_current_and_legacy_cache_roundtrip(mode, tmp_path):
    n = network(mode=mode)
    expected = n.fixed_sequence_record((0, 1, 2, 3))
    n.set_fixed_sequence_index()
    identity = {"git_commit": "same-execution-source", "schema": "test"}
    path = tmp_path / "graph.pkl"
    exact._write_event_network_cache(path, n, identity, 0)
    loaded, _ = exact._load_event_network_cache(path, identity)
    assert not loaded.fixed_sequence_index
    with mock.patch.object(loaded, "_validate_replay_rows",
                           side_effect=AssertionError("new graph must not scan")):
        loaded.set_fixed_sequence_index()
        assert loaded.fixed_sequence_record((0, 1, 2, 3)) == expected
    with pytest.raises(ValueError, match="identity mismatch"):
        exact._load_event_network_cache(path, {**identity, "git_commit": "other"})
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        exact._load_event_network_cache(path, identity)
    # Emulate an actual old pickle with no new runtime/invariant fields.
    state = n.__getstate__()
    for key in ("fixed_sequence_index", "_replay_sorted_rows_version"):
        state.pop(key, None)
    old = EventExpandedNetwork.__new__(EventExpandedNetwork)
    old.__setstate__(state)
    old = pickle.loads(pickle.dumps(old))
    assert not old.fixed_sequence_index
    with mock.patch.object(old, "_validate_replay_rows",
                           wraps=old._validate_replay_rows) as validate:
        old.set_fixed_sequence_index()
        old.set_fixed_sequence_index(False)
        old.set_fixed_sequence_index()
        assert validate.call_count == 1
    assert old.fixed_sequence_record((0, 1, 2, 3)) == expected


@pytest.mark.parametrize("mode", ["explicit", "lazy"])
def test_legacy_unsorted_rows_fail_closed(mode):
    n = network(mode=mode)
    del n._replay_sorted_rows_version
    if mode == "explicit":
        n.out[0].reverse()
    else:
        start, end = n._arc_slices[0]
        n._arc_targets[start:end] = n._arc_targets[start:end][::-1]
    with pytest.raises(ValueError, match="unsorted"):
        n.set_fixed_sequence_index()
    assert not n.fixed_sequence_index


@pytest.mark.parametrize("max_columns,time_limit_s", [(2, 900), (512, 900), (0, 0)])
def test_bounded_import_same_selection_records_and_rejections(
        tmp_path, monkeypatch, max_columns, time_limit_s):
    p = four_trip_chain_problem()
    rows = [{"count_trip_id": i, "Ordered_Trip_ID": 10 + i} for i in p.trips]
    parent, child = tmp_path / "parent.csv", tmp_path / "child.csv"
    write_instance(parent, rows)
    write_instance(child, rows)
    # Distinct incidence sets; includes an impossible backwards transition.
    sequences = [(0, 1, 2, 3), (3, 2, 1), (0, 2), (1,), (2, 3)]
    journal = tmp_path / "pool.jsonl"
    journal.write_text("".join(json.dumps({"trips": s, "cost": 100000 + i})
                               + "\n" for i, s in enumerate(sequences)))
    status = tmp_path / "parent.json"
    status.write_text(json.dumps({"csv": "parent.csv", "trip_ids": list(p.trips),
        "columns_journal": str(journal),
        "provenance": {"instance_sha256": sha256(parent)}}))
    monkeypatch.setattr(exact, "DATA_DIR", tmp_path)
    selections = []
    real_context = multiprocessing.get_context("fork")

    class Context:
        def Pool(self, workers, **kwargs):
            pool = real_context.Pool(workers, **kwargs)
            def instrument(original):
                def capture(func, items, chunksize):
                    selections.append(deepcopy(items))
                    return original(func, items, chunksize)
                return capture
            pool.imap_unordered = instrument(pool.imap_unordered)
            pool.imap = instrument(pool.imap)
            return pool

    monkeypatch.setattr(exact.multiprocessing, "get_context", lambda _: Context())
    n = network(p)
    results = []
    for enabled in (False, True):
        n.set_fixed_sequence_index(enabled)
        records, audit = exact.inherited_event_pool_records(
            status, child_csv_path=child, child_problem=p, child_network=n,
            g_kwh=240, charge_kw=240, reserve_kwh=0,
            max_columns=max_columns, time_limit_s=time_limit_s, workers=8,
        )
        assert audit["pool_shutdown"]["shutdown_complete"]
        assert not audit["pool_shutdown"]["alive_worker_pids"]
        # Elapsed times and OS-assigned PIDs differ across executions. Compare
        # every scientific audit field and the complete accepted route records.
        operational = {"preparation_s", "replay_s", "cleanup_s", "total_import_s", "pool_shutdown"}
        results.append((sorted(records, key=lambda r: r["trips"]),
                        {k: v for k, v in audit.items() if k not in operational}))
    assert selections[0] == selections[1]
    assert results[0] == results[1]
    assert results[0][1]["accepted_columns"] == (1 if max_columns == 2 else 4)
    assert results[0][1]["rejected_columns"] == 1
    assert not results[0][1]["import_deadline_reached"]
    assert not results[0][1]["inherited_lp_certificate"]


def test_legacy_validation_checks_chunk_boundary():
    from array import array
    import numpy as np
    n = network()
    n._arc_targets = array("I", [2]) * 65539
    # Inversion crosses two validation chunks, rather than lying within one.
    n._arc_targets[65536] = 3
    n._arc_targets_np = np.frombuffer(n._arc_targets, dtype=np.uint32)
    n._arc_slices = [(0, len(n._arc_targets))]
    with pytest.raises(ValueError, match="unsorted"):
        n._validate_replay_rows()


def test_noncontiguous_trip_nodes_fail_closed():
    n = network()
    n.node_meta[3] = ("trip", 1, 0)
    with pytest.raises(ValueError, match="noncontiguous"):
        n.set_fixed_sequence_index()
