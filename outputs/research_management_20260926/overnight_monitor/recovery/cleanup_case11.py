#!/usr/bin/env python3
"""Execute the frozen cleanup with its sole enumeration guard raised for one source.

No checkout is edited. All imports, frontiers, enumeration and MIP code remain
from execution commit 4a8b497e; this wrapper only admits the audited 11-trip case.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/home/nc437/ladder-lite/spatial_tariff_expansion_20260925')
CELL = 'mix1_two_price_split'
COMMIT = '4a8b497e668be9962bca5906a8b69116fa634882'
CODE = ROOT / 'code'
SOURCE = ROOT / 'results' / CELL / 'fresh_cg/job499021_r0/out/summary.json'
SOURCE_SHA = '684886c6dd791357cab107bc5eb951f951c91b21bf8b53a2171b3073d5551aac'
SELECTED_SHA = 'c771a2c97b8ba3a32af3201f0b7f3d898861e9c19b03d1f0ddb80df908ce5768'
MODULE_SHA = '990a2a5a741d89e921acbf654c07137d4739444294092d3b4210cc7fbb8914db'
OLD = b"assert len(duplicated)<=10,'Require an explicit larger repair design'"
NEW = b"assert len(duplicated)<=11,'Require an explicit larger repair design'"
DUPLICATE_COUNTS = [10, 1, 0, 2, 11]
TRIP_COUNTS = [40, 18, 12, 20, 33]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def patched_source(original):
    """Return a byte-exact one-token patch or fail closed."""
    if hashlib.sha256(original).hexdigest() != MODULE_SHA:
        raise ValueError('frozen cleanup module hash mismatch')
    if original.count(OLD) != 1 or NEW in original:
        raise ValueError('expected exactly one original guard')
    patched = original.replace(OLD, NEW, 1)
    if patched.replace(NEW, OLD, 1) != original:
        raise ValueError('patch changed more than the enumeration guard')
    compile(patched, 'terminal_duplicate_cleanup_case11.py', 'exec')
    return patched


def selection_counts(selected):
    counts = Counter(t for route in selected for t in route['trips'])
    if any(len(route['trips']) != len(set(route['trips'])) for route in selected):
        raise ValueError('unexpected repeated trip within a source route')
    duplicates = [sum(counts[t] > 1 for t in r['trips']) for r in selected]
    lengths = [len(r['trips']) for r in selected]
    if duplicates != DUPLICATE_COUNTS or lengths != TRIP_COUNTS:
        raise ValueError('source selection differs from audited duplicate counts')
    if len(counts) != 111 or sum(c - 1 for c in counts.values()) != 12:
        raise ValueError('source selection differs from audited trip coverage')
    return dict(duplicated_trip_counts_by_route=duplicates,
                route_trip_counts=lengths, unique_trips=len(counts),
                duplicate_excess=12, subsequence_upper_count=sum(2 ** d for d in duplicates))


def validate_source(source, source_sha):
    source = Path(source)
    if source.resolve() != SOURCE.resolve() or source_sha != SOURCE_SHA:
        raise ValueError('this exception is pinned to the exact job499021_r0 source')
    if sha(source) != SOURCE_SHA:
        raise ValueError('source summary hash mismatch')
    selected_path = source.parent / 'selected_routes.json'
    if sha(selected_path) != SELECTED_SHA:
        raise ValueError('selected routes hash mismatch')
    summary = json.loads(source.read_text())
    if summary['fleet_cap'] != 5 or summary['terminal_target_kwh'] != 379.7984451:
        raise ValueError('fleet or terminal target changed')
    for path, expected in summary['input_hashes'].items():
        if sha(path) != expected:
            raise ValueError('input hash mismatch: ' + path)
    receipt = selection_counts(json.loads(selected_path.read_text()))
    receipt.update(source=str(source), source_sha256=SOURCE_SHA,
                   selected_source_sha256=SELECTED_SHA, original_module_sha256=MODULE_SHA,
                   original_guard=10, recovery_guard=11,
                   semantics='identical exhaustive subsequences and charging frontiers; guard only')
    return summary, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seconds', type=float, default=3600)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.seconds != 3600 or args.threads != 8:
        parser.error('this recovery preserves the original 3600-second MIP and 8 threads')
    module = CODE / 'src/terminal_duplicate_cleanup.py'
    patched = patched_source(module.read_bytes())
    _, receipt = validate_source(args.source, args.source_sha256)
    receipt['patched_module_sha256'] = hashlib.sha256(patched).hexdigest()
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if args.check_only:
        return
    if sys.flags.optimize:
        raise ValueError('Python optimization would disable original scientific assertions')
    sys.path.insert(0, str(CODE / 'src'))
    sys.argv = [str(module), '--source', args.source, '--source-sha256', args.source_sha256,
                '--out', args.out, '--seconds', '3600.0', '--threads', '8']
    namespace = dict(__name__='__main__', __file__=str(module), __package__=None)
    exec(compile(patched, str(module) + '[case11 guard]', 'exec'), namespace)


if __name__ == '__main__':
    main()
