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
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.identity = dict(identity)
        encoded = json.dumps(
            self.identity, sort_keys=True, separators=(",", ":")
        ).encode()
        self.identity_sha256 = hashlib.sha256(encoded).hexdigest()
        existing = []
        if self.path.exists() and self.path.stat().st_size:
            existing = read_jsonl_records(
                self.path, repair_trailing=True
            )
            starts = [
                record for record in existing
                if record.get("record_type") == "session_start"
            ]
            if not starts:
                raise DurableFileError(
                    f"telemetry sidecar has no session identity: {self.path}"
                )
            if any(
                    record.get("identity_sha256") != self.identity_sha256
                    for record in starts):
                raise DurableFileError(
                    f"telemetry sidecar belongs to different work: {self.path}"
                )
        self.session = 1 + sum(
            record.get("record_type") == "session_start"
            for record in existing
        )
        self.started_perf = time.perf_counter()
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
        record = {
            "schema": SCHEMA,
            "record_type": "phase",
            "session": self.session,
            "identity_sha256": self.identity_sha256,
            "phase": str(name),
            "duration_s": float(duration_s),
            "elapsed_session_s": time.perf_counter() - self.started_perf,
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
