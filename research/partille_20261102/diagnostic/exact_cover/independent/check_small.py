"""Compare single-peak verifier to naive enumeration for small random eight-bus pools."""
from itertools import combinations
import json
from pathlib import Path
from random import Random
from complement import solve
from verify import sha
rng=Random(236042); accepted=rejected=0
for i in range(160):
    size=8+rng.randrange(1,6); anchor=(1<<8)-1; full=(1<<size)-1
    masks=[]
    for trip in range(8):
        options=1+rng.randrange(2);masks.extend((1<<trip)|(rng.randrange(1<<(size-8))<<8) for _ in range(options))
    masks=sorted(set(masks)); exists=False
    for choice in combinations(masks,8):
        union=0
        for route in choice:
            if union & route:break
            union|=route
        else:
            if union==full:exists=True;break
    observed=solve(masks,anchor,size,8)[0]
    if observed!=exists:raise ValueError('disagreement with naive subsets')
    accepted+=int(exists);rejected+=int(not exists)
assert accepted and rejected
p=Path(__file__);r=dict(seed=236042,small_pools=160,accepted=accepted,rejected=rejected,
                      agreement='all',test_sha256=sha(p.read_bytes()),checked_solver_sha256=sha(p.with_name('complement.py').read_bytes()))
p.with_name('small_fixtures.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
