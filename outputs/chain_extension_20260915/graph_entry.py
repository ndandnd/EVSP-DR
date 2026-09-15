"""Run unchanged event CG with synchronous, coarse graph-build progress logs."""
import functools
import json
import os
from pathlib import Path
import resource
import sys
import time


def install_progress(exact, network_class, destination, interval=60.0):
    started = time.monotonic()
    progress = {"finished_sources": 0, "last_write": 0.0}
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def emit(phase, event, **details):
        row = {"phase": phase, "event": event, "elapsed_s": time.monotonic()-started,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "pid": os.getpid(), **details}
        with destination.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True)+"\n")
        print("GRAPH_PROGRESS "+json.dumps(row, sort_keys=True), flush=True)

    def wrap(owner, name):
        original = getattr(owner, name)
        @functools.wraps(original)
        def measured(*args, **kwargs):
            emit(name, "start")
            start = time.monotonic()
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                emit(name, "error", error_type=type(exc).__name__)
                raise
            emit(name, "complete", duration_s=time.monotonic()-start)
            return result
        setattr(owner, name, measured)

    for name in ["build_problem", "_write_event_network_cache"]:
        wrap(exact, name)
    for name in ["_split_arcs", "_build_nodes", "_build_arcs"]:
        wrap(network_class, name)
    original = network_class._finalize_source
    @functools.wraps(original)
    def finalize(self, source):
        result = original(self, source)
        progress["finished_sources"] += 1
        now = time.monotonic()
        if now-progress["last_write"] >= interval:
            progress["last_write"] = now
            emit("_build_arcs", "progress", finished_sources=progress["finished_sources"],
                 total_sources=1+len(self.trip_node), source=int(source),
                 stored_arcs=len(self._arc_targets) if self.arc_mode == "lazy" else None)
        return result
    network_class._finalize_source = finalize
    emit("entry", "start")


if __name__ == "__main__":
    code, destination = sys.argv[1:3]
    sys.path.insert(0, str(Path(code)/"src"))
    import exact_pricer_expanded as exact
    from event_pricer_network import EventExpandedNetwork
    install_progress(exact, EventExpandedNetwork, destination)
    raise SystemExit(exact.main(sys.argv[3:]))
