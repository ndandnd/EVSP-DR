"""Durable, opt-in phase telemetry for exact column generation.

The sidecar is operational evidence only: it is not part of resume identity or
model provenance.  Each event is an independently flushed JSONL object, so a
hard preemption can damage at most the final row; reopening repairs only that
tail through the same strict durable reader used by current journals.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

from durable_io import DurableFileError, flush_and_fsync, read_jsonl_records


SCHEMA = "evsp-dr-exact-cg-phase-telemetry-v1"


def peak_rss_bytes() -> int:
    """Return process peak RSS in bytes on Linux/macOS."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


class PhaseTelemetry:
    """Append-only, identity-bound telemetry sidecar."""

    def __init__(self, path: Path, *, identity: dict):
        initialization_started = time.perf_counter()
        self.overhead_s = 0.0
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        encoded = json.dumps(
            self.identity, sort_keys=True, separators=(",", ":")
        ).encode()
        self.identity_sha256 = hashlib.sha256(encoded).hexdigest()
        session_count = 0
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                        if handle.read().strip():
                            raise DurableFileError(
                                f"telemetry has malformed data before EOF at "
                                f"byte {offset}: {self.path}"
                            ) from exc
                        # Do not repair until every complete session identity
                        # has been authenticated below.
                        break
                    if not isinstance(record, dict):
                        raise DurableFileError(
                            f"telemetry row at byte {offset} is not an object"
                        )
                    if record.get("identity_sha256") != self.identity_sha256:
                        raise DurableFileError(
                            "telemetry sidecar belongs to different work: "
                            f"{self.path}"
                        )
                    if record.get("record_type") == "session_start":
                        session_count += 1
            if session_count == 0:
                raise DurableFileError(
                    f"telemetry sidecar has no session identity: {self.path}"
                )
            # Identity is now proven. Normalize only the matching sidecar's
            # interrupted final row/missing final newline.
            read_jsonl_records(
                self.path, repair_trailing=True, collect=False
            )
        self.session = 1 + session_count
        self.started_perf = initialization_started
        self._append({
            "schema": SCHEMA,
            "record_type": "session_start",
            "session": self.session,
            "identity_sha256": self.identity_sha256,
            "identity": self.identity,
            "pid": os.getpid(),
            "host": platform.node(),
            "python": platform.python_version(),
            "epoch_s": time.time(),
            "peak_rss_bytes": peak_rss_bytes(),
        })
        self.overhead_s += time.perf_counter() - initialization_started

    def _append(self, payload: dict) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            flush_and_fsync(handle)

    def phase(
        self,
        name: str,
        duration_s: float,
        *,
        iteration: int | None = None,
        attempt: int | None = None,
        pool_columns: int | None = None,
        incidence_nnz: int | None = None,
        network_nodes: int | None = None,
        network_arcs: int | None = None,
        outcome: str = "ok",
        details: dict | None = None,
    ) -> None:
        started = time.perf_counter()
        try:
            record = {
                "schema": SCHEMA,
                "record_type": "phase",
                "session": self.session,
                "identity_sha256": self.identity_sha256,
                "phase": str(name),
                "duration_s": float(duration_s),
                "elapsed_session_s": (
                    time.perf_counter() - self.started_perf - self.overhead_s
                ),
                "telemetry_overhead_before_s": self.overhead_s,
                "epoch_s": time.time(),
                "iteration": iteration,
                "attempt": attempt,
                "pool_columns": pool_columns,
                "incidence_nnz": incidence_nnz,
                "network_nodes": network_nodes,
                "network_arcs": network_arcs,
                "peak_rss_bytes": peak_rss_bytes(),
                "outcome": outcome,
                "details": dict(details or {}),
            }
            self._append(record)
        finally:
            self.overhead_s += time.perf_counter() - started
