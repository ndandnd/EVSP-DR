"""Count which ordered trip sequences occur in an authenticated archived route journal."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def scan(pool, summary, witness, expected_digest):
    native = json.loads(summary.read_bytes())
    sequences = [tuple(route['trips']) for route in json.loads(witness.read_bytes())]
    if len(set(sequences)) != len(sequences):
        raise ValueError('duplicate witness sequences')
    counts = Counter(dict.fromkeys(sequences, 0))
    root = hashlib.sha256()
    full = hashlib.sha256()
    n = 0
    opener = gzip.open if pool.suffix == '.gz' else open
    with opener(pool, 'rb') as source:
        for n, line in enumerate(source, 1):
            if n <= native['columns']:
                root.update(line)
            full.update(line)
            sequence = tuple(json.loads(line)['trips'])
            if sequence in counts:
                counts[sequence] += 1
    if (full.hexdigest() != expected_digest or root.hexdigest() != native['columns_sha256']
            or n != native['dive']['columns_after_dive']):
        raise ValueError('journal authentication failed')
    return dict(column_count=n,present=sum(c>0 for c in counts.values()),matches=sorted(counts.values()),
                root_prefix_sha256=root.hexdigest(),source_pool_sha256=full.hexdigest())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('pool','summary','witness'):
        p.add_argument('--'+arg,type=Path,required=True)
    p.add_argument('--sha256',required=True)
    a = p.parse_args()
    print(json.dumps(scan(a.pool,a.summary,a.witness,a.sha256),indent=2))
