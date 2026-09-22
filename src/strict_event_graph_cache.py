"""Atomic, same-source caches for strict packed event graphs.

Only consume locally prepared, trusted artifacts: checksums detect corruption,
not an adversary who can replace both serialized metadata and its manifest.
Packed buffers stream in bounded chunks instead of pickle's whole-array copy.
"""
from __future__ import annotations

from array import array
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import struct
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from audit_giro_known_columns import DEPOT, HORIZON_MIN, STATIONS
from config import BIG_M_PENALTY, BUS_COST_KX, CHARGE_START_COST, charge_cost_premium
from event_pricer_network import EventExpandedNetwork, _event_times
from expanded_path_realization import normalize_event_station_prices

SCHEMA = "evsp-dr-strict-packed-graph-cache-v1"
MAGIC = b"EVSP-STRICT-GRAPH-v1\n"
CHUNK = 8 * 1024 * 1024
MAX_MANIFEST = 2 * 1024 * 1024
BUFFERS = (("_arc_targets", "I"), ("_arc_costs", "d"), ("_arc_recipes", "I"))


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def problem_sha(problem):
    return canonical_sha({
        "trips": list(problem.trips),
        "start": [(trip, problem.start_min[trip]) for trip in problem.trips],
        "end": [(trip, problem.end_min[trip]) for trip in problem.trips],
        "energy": [(trip, problem.trip_energy[trip]) for trip in problem.trips],
        "adjacency": list(problem.adjacency.items()),
    })


def graph_identity(args, problem, prices, prov, *, arm):
    if getattr(args, "arc_mode", "explicit") != "lazy" or arm["capacity"]:
        raise ValueError("strict graph cache requires lazy packed arcs without shared capacity")
    source = Path(__file__).resolve().parent
    normalized = normalize_event_station_prices(prices, horizon_min=HORIZON_MIN,
                                                strict_tariff_coverage=False)
    events = _event_times(problem, normalized, args.block_min)
    grid = [round(index * args.soc_step, 9)
            for index in range(int(args.battery_kwh / args.soc_step) + 1)]
    return {
        "schema": SCHEMA,
        "implementation_git_commit": prov["git_commit"],
        # Conservative full Python-source invalidation also covers transitive
        # problem construction, pricing, recovery and physical replay helpers.
        "source_sha256": {str(p.relative_to(source)): file_sha(p)
                          for p in sorted(source.rglob("*.py"))},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pandas": pd.__version__, "byteorder": sys.byteorder,
                    "system": platform.system(), "machine": platform.machine(),
                    "array_itemsize": {code: array(code).itemsize for _, code in BUFFERS}},
        "inputs": {key: prov[key] for key in (
            "instance_sha256", "prices_sha256", "reference_sha256", "deadhead_sha256")},
        "problem_sha256": problem_sha(problem),
        "trip_order": list(problem.trips),
        "normalized_prices_sha256": canonical_sha(normalized),
        "event_lattice_sha256": canonical_sha(events),
        "physics": {"battery_kwh": args.battery_kwh, "initial_soc_kwh": args.battery_kwh,
            "reserve_kwh": args.reserve_kwh, "terminal_soc_constraint": "reserve_only",
            "soc_step_kwh": args.soc_step, "soc_grid": grid,
            "event_block_min": args.block_min, "non_parx_kw": args.non_parx_kw,
            "station_charge_kw": {"PARX": float(arm["parx_kw"])},
            "max_station_wait_min": getattr(args, "max_station_wait_min", 220.0),
            "horizon_min": HORIZON_MIN, "depot": DEPOT, "station_order": list(STATIONS),
            "strict_tariff_coverage": False, "arc_mode": "lazy",
            "capacity_selector": getattr(args, "capacity_selector", "reference"),
            "capacity_enforced": False, "charger_counts": {}, "capacity_grid_min": 1,
            "parx_capacity": "unlimited"},
        "objective": {"master": "set_covering", "master_sense": "minimize",
            "pricing_objective": "combined-cost", "bus_cost_kx": BUS_COST_KX,
            "charge_start_cost": CHARGE_START_COST, "charge_cost_premium": charge_cost_premium,
            "artificial_cost": BIG_M_PENALTY},
    }


def graph_fingerprint(network):
    """Exact structural and metadata fingerprint, excluding transient caches."""
    state = network.__getstate__()
    hashes = {}
    for name, _code in BUFFERS:
        value = state.pop(name)
        digest = hashlib.sha256()
        view = memoryview(value).cast("B")
        for start in range(0, len(view), CHUNK):
            digest.update(view[start:start + CHUNK])
        hashes[name] = digest.hexdigest()
    # State metadata is small compared to packed arcs; pickle preserves the
    # problem frame and typed keys that a JSON dump would lose.
    hashes["metadata"] = hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()
    return hashes


def validate_network(network, identity):
    physics = identity["physics"]
    observed = {
        "trips": list(network.problem.trips), "problem": problem_sha(network.problem),
        "events": canonical_sha(network.events), "prices": canonical_sha(network.prices),
        "grid": network.grid, "g": network.g, "reserve": network.reserve,
        "soc_step": network.soc_step, "block_min": network.block_min,
        "charge_kw": network.charge_kw, "station_charge_kw": network.station_charge_kw,
        "arc_mode": network.arc_mode, "capacity_selector": network.capacity_selector,
        "strict_tariff_coverage": network.strict_tariff_coverage,
    }
    expected = {
        "trips": identity["trip_order"], "problem": identity["problem_sha256"],
        "events": identity["event_lattice_sha256"], "prices": identity["normalized_prices_sha256"],
        "grid": physics["soc_grid"], "g": physics["battery_kwh"], "reserve": physics["reserve_kwh"],
        "soc_step": physics["soc_step_kwh"], "block_min": physics["event_block_min"],
        "charge_kw": physics["non_parx_kw"], "station_charge_kw": physics["station_charge_kw"],
        "arc_mode": physics["arc_mode"], "capacity_selector": physics["capacity_selector"],
        "strict_tariff_coverage": physics["strict_tariff_coverage"],
    }
    if observed != expected:
        raise ValueError("graph object identity mismatch")
    if any(len(getattr(network, name)) != network.n_arcs for name, _ in BUFFERS):
        raise ValueError("graph packed-buffer length mismatch")
    if network.trip_position != {trip: i for i, trip in enumerate(identity["trip_order"])}:
        raise ValueError("graph trip-position mismatch")
    if network.station_position != {station: i for i, station in enumerate(STATIONS)}:
        raise ValueError("graph station-position mismatch")


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _reserve_cache(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.close(lock_fd)
        if path.exists():
            raise FileExistsError(f"graph cache already exists: {path}")
        yield
    finally:
        lock.unlink(missing_ok=True)


def write_graph_cache(path, network, identity, *, build_s, clock=time.perf_counter):
    """Publish one immutable artifact only after its bytes and manifest are synced."""
    path = Path(path)
    with _reserve_cache(path):
        return _write_reserved_cache(path, network, identity, build_s=build_s, clock=clock)


def prepare_graph_cache(path, identity, builder, *, clock=time.perf_counter):
    """Reserve the name before expensive construction; never duplicate a writer."""
    path = Path(path)
    with _reserve_cache(path):
        started = clock()
        network = builder()
        build_s = clock() - started
        manifest = _write_reserved_cache(path, network, identity, build_s=build_s, clock=clock)
        return network, manifest


def _write_reserved_cache(path, network, identity, *, build_s, clock):
    temporary = None
    started = clock()
    try:
        validate_network(network, identity)
        if not math.isfinite(build_s) or build_s < 0:
            raise ValueError("invalid graph construction time")
        state = network.__getstate__()
        buffers = [(name, state.pop(name)) for name, _ in BUFFERS]
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            pickle.dump(state, handle, protocol=5)
            metadata_bytes = handle.tell()
            layout = []
            for name, value in buffers:
                layout.append({"name": name, "typecode": value.typecode,
                               "count": len(value), "bytes": len(value) * value.itemsize})
                view = memoryview(value).cast("B")
                for start in range(0, len(view), CHUNK):
                    handle.write(view[start:start + CHUNK])
            payload_bytes = handle.tell()
            handle.flush()
            handle.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(CHUNK), b""):
                digest.update(chunk)
            manifest = {"schema": SCHEMA, "identity": identity,
                "identity_sha256": canonical_sha(identity), "payload_sha256": digest.hexdigest(),
                "payload_bytes": payload_bytes, "metadata_bytes": metadata_bytes,
                "buffers": layout, "network": network.metrics(), "build_s": build_s,
                "serialization_and_hash_s": clock() - started}
            encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode()
            if len(encoded) > MAX_MANIFEST:
                raise ValueError("graph manifest exceeds size limit")
            handle.write(encoded)
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(MAGIC)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_graph_cache(path, expected):
    """Verify identity and every payload byte before deserializing any metadata."""
    path = Path(path)
    with path.open("rb") as handle:
        size = os.fstat(handle.fileno()).st_size
        trailer = len(MAGIC) + 8
        if size < trailer:
            raise ValueError("truncated graph cache")
        handle.seek(-trailer, os.SEEK_END)
        length = struct.unpack("<Q", handle.read(8))[0]
        if handle.read() != MAGIC or not 0 < length <= min(MAX_MANIFEST, size - trailer):
            raise ValueError("invalid or truncated graph cache trailer")
        payload_bytes = size - trailer - length
        handle.seek(payload_bytes)
        try:
            manifest = json.loads(handle.read(length))
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError("invalid graph cache manifest") from error
        if manifest.get("schema") != SCHEMA:
            raise ValueError("graph cache schema mismatch")
        if (manifest.get("identity") != expected
                or manifest.get("identity_sha256") != canonical_sha(expected)):
            raise ValueError("graph cache identity mismatch (source/input/physics/trip/event/runtime)")
        if manifest.get("payload_bytes") != payload_bytes:
            raise ValueError("graph cache payload size mismatch")
        metadata_bytes = manifest.get("metadata_bytes")
        layout = manifest.get("buffers")
        if not isinstance(metadata_bytes, int) or not 0 < metadata_bytes <= payload_bytes:
            raise ValueError("invalid graph cache metadata length")
        if not isinstance(layout, list) or len(layout) != len(BUFFERS):
            raise ValueError("invalid graph cache buffers")
        for entry, (name, code) in zip(layout, BUFFERS):
            if (entry.get("name") != name or entry.get("typecode") != code
                    or type(entry.get("count")) is not int or entry["count"] < 0
                    or entry.get("bytes") != entry["count"] * array(code).itemsize):
                raise ValueError("invalid graph cache buffer layout")
        if metadata_bytes + sum(entry["bytes"] for entry in layout) != payload_bytes:
            raise ValueError("graph cache buffer size mismatch")
        if not isinstance(manifest.get("build_s"), (int, float)) or not math.isfinite(manifest["build_s"]) or manifest["build_s"] < 0:
            raise ValueError("invalid graph cache construction time")
        digest = hashlib.sha256()
        handle.seek(0)
        remaining = payload_bytes
        while remaining:
            chunk = handle.read(min(CHUNK, remaining))
            if not chunk:
                raise ValueError("truncated graph cache payload")
            digest.update(chunk)
            remaining -= len(chunk)
        if digest.hexdigest() != manifest.get("payload_sha256"):
            raise ValueError("graph cache payload checksum mismatch")
        handle.seek(0)
        state = pickle.load(handle)
        if handle.tell() != metadata_bytes or not isinstance(state, dict):
            raise ValueError("graph cache metadata boundary mismatch")
        for entry in layout:
            value = array(entry["typecode"])
            remaining = entry["bytes"]
            while remaining:
                chunk = handle.read(min(CHUNK, remaining))
                if not chunk:
                    raise ValueError("truncated graph cache buffer")
                value.frombytes(chunk)
                remaining -= len(chunk)
            state[entry["name"]] = value
        network = EventExpandedNetwork.__new__(EventExpandedNetwork)
        network.__setstate__(state)
        validate_network(network, expected)
        if network.metrics() != manifest.get("network"):
            raise ValueError("graph cache network metrics mismatch")
        return network, manifest
