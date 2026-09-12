"""Observational wrapper: retain solver arguments and graph construction order."""
import argparse
import faulthandler
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solver', type=Path, required=True)
    p.add_argument('--progress', type=Path, required=True)
    p.add_argument('--stacks', type=Path, required=True)
    p.add_argument('--progress-seconds', type=float, default=60)
    p.add_argument('--dump-seconds', type=float, default=120)
    p.add_argument('solver_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    argv = a.solver_args[1:] if a.solver_args[:1] == ['--'] else a.solver_args
    a.progress.parent.mkdir(parents=True, exist_ok=True)
    a.stacks.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.solver.resolve().parent))
    import event_pricer_network as network
    original = network.EventExpandedNetwork._finalize_source
    original_build = network.EventExpandedNetwork._build_arcs
    start = time.monotonic()
    last = start
    completed = 0
    with a.progress.open('x') as log, a.stacks.open('x') as stacks:
        def emit(record):
            log.write(json.dumps(record, sort_keys=True) + '\n')
            log.flush()

        def finalize(net, source):
            nonlocal last, completed
            result = original(net, source)
            completed += 1
            current = time.monotonic()
            if completed == 1 or current - last >= a.progress_seconds:
                emit({'phase': 'graph_source_finalized', 'elapsed_s': current-start,
                      'completed_sources': completed, 'source': source,
                      'total_trip_soc_sources': len(net.trip_node),
                      'packed_arcs_so_far': len(getattr(net, '_arc_targets', [])),
                      'window_cache_entries': len(net._window_cache),
                      'maxrss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      'platform': sys.platform})
                last = current
            return result

        def build(net):
            # Arm only inside graph construction, after preflight/provenance
            # subprocesses. Do not introduce a timer thread across their forks.
            faulthandler.dump_traceback_later(a.dump_seconds, repeat=True, file=stacks)
            try:
                return original_build(net)
            finally:
                faulthandler.cancel_dump_traceback_later()

        network.EventExpandedNetwork._finalize_source = finalize
        network.EventExpandedNetwork._build_arcs = build
        emit({'phase': 'diagnostic_start', 'pid': os.getpid(),
              'solver': str(a.solver.resolve()), 'argv': argv})
        try:
            spec = importlib.util.spec_from_file_location('exact_pricer_expanded', a.solver)
            solver = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = solver
            spec.loader.exec_module(solver)
            result = solver.main(argv)
            emit({'phase': 'diagnostic_complete', 'elapsed_s': time.monotonic()-start,
                  'completed_sources': completed, 'returncode': result})
            return result
        finally:
            faulthandler.cancel_dump_traceback_later()
            network.EventExpandedNetwork._finalize_source = original
            network.EventExpandedNetwork._build_arcs = original_build


if __name__ == '__main__':
    raise SystemExit(main())
