"""Enumerate disjoint trip masks as a relaxation of the fixed-pool eight-bus partition problem."""
import collections
import hashlib
import json
import sys
import time
from pathlib import Path
from original_pool_at_floor import simultaneous, FULL_HASH

def search(instance_path, pool_path, out, seconds):
    groups,floor,ntrips=simultaneous(instance_path)
    by_trip=[0]*ntrips
    by_sequence={}
    tripsets=[]
    digest=hashlib.sha256()
    with pool_path.open('rb') as f:
        for lineno,line in enumerate(f):
            digest.update(line)
            seq=tuple(json.loads(line[line.index(b'['):line.index(b']')+1]))
            if seq in by_sequence: continue
            cover=set(seq)
            if not all(len(cover & g)==1 for g in groups): continue
            n=len(tripsets)
            by_sequence[seq]=n
            tripsets.append(sum(1<<t for t in seq))
            for t in seq: by_trip[t]|=1<<n
    assert digest.hexdigest()==FULL_HASH
    collision=[]
    for seq in by_sequence:
        forbidden=0
        for t in seq: forbidden|=by_trip[t]
        collision.append(forbidden)
    ordered=sorted(range(ntrips),key=lambda t:by_trip[t].bit_count())
    start=time.monotonic(); deadline=start+seconds
    nodes=0; stack=[]; success=False
    def walk(alive,remaining):
        nonlocal nodes,success
        nodes+=1
        if not (nodes % 256) and time.monotonic()>deadline: raise TimeoutError()
        if not remaining:
            success=True
            return True
        minimum=len(tripsets)+1
        for t in ordered:
            if not remaining & 1<<t: continue
            viable=alive & by_trip[t]
            cnt=viable.bit_count()
            if not cnt: return False
            if cnt<minimum: minimum=cnt; branch=viable
        while branch:
            low=branch & -branch
            branch^=low
            r=low.bit_length()-1
            stack.append(r)
            if walk(alive & ~collision[r], remaining & ~tripsets[r]):return True
            stack.pop()
        return False
    exhausted=False
    try:exhausted=not walk((1<<len(tripsets))-1,(1<<ntrips)-1)
    except TimeoutError:pass
    output=dict(arm='partition_relaxation_ignoring_shared_plugs',unique_trip_sequences=len(tripsets),
                nodes=nodes,exhausted=exhausted,found_trip_partition=success,seconds=time.monotonic()-start,
                floor=floor,pool_sha256=FULL_HASH)
    if success:
        assert len(stack)==floor
        seqs=list(by_sequence)
        (out.parent/'private_partition_trip_sequences.json').write_text(json.dumps([seqs[r] for r in stack]))
    out.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__':
    search(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]),float(sys.argv[4]))
