"""Validate/repair only a private checkpoint copy, with the pinned native rules.
No CG, pricing, graph expansion or optimization is executed.
"""
from pathlib import Path
import argparse
import json
import sys


def validate(code, argv):
    sys.path.insert(0, str(Path(code) / 'src'))
    import exact_pricer_expanded as native
    # Obtain precisely the native CLI defaults without entering run_cg or locks.
    parse = argparse.ArgumentParser.parse_args
    captured = []
    class Parsed(BaseException):
        pass
    def capture(parser, *args, **kwargs):
        captured.append(parse(parser, *args, **kwargs))
        raise Parsed()
    argparse.ArgumentParser.parse_args = capture
    try:
        native.main(argv)
    except Parsed:
        pass
    finally:
        argparse.ArgumentParser.parse_args = parse
    args = captured[0]
    status_path = Path(args.out)
    status = json.loads(status_path.read_text())
    problem = native.build_problem(native.DATA_DIR, args.csv,
                                   max_station_to_trip_wait_min=native.HORIZON_MIN)
    trips = list(problem.trips)
    issues = native.resume_identity_mismatches(status, args, trips, native._provenance(args))
    if issues:
        raise ValueError('identity: ' + '; '.join(issues))
    journal = Path(str(status_path) + '.columns.jsonl')
    if not journal.is_file():
        raise ValueError('missing column journal')
    records = native.read_jsonl_records(journal, repair_trailing=True)
    pool = native.load_column_pool(records, trips)
    issues = native.resume_pool_mismatches(status, pool)
    if issues:
        raise ValueError('pool: ' + '; '.join(issues))
    iterations = Path(str(status_path) + '.iters.csv')
    rows = native.load_iteration_log(iterations, repair_trailing=True) if iterations.exists() and iterations.stat().st_size else []
    return {'valid': True, 'columns': len(pool), 'journal_records': len(records),
            'iteration_rows': len(rows), 'identity_and_pool_checks': 'pinned native functions',
            'solver_executed': False}

if __name__ == '__main__':
    request = json.loads(sys.stdin.read())
    try:
        result = validate(request['code'], request['argv'])
    except Exception as exc:
        print(json.dumps({'valid': False, 'error': type(exc).__name__ + ': ' + str(exc)}))
        raise SystemExit(1)
    print(json.dumps(result))
