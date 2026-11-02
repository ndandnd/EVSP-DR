"""Exact eight-route disjoint union by complement. Input authenticator is verify.py.

Only the FIRST peak of the timetable is used to prune masks. In the final two
positions directly test complementary pairs using a dictionary of complete masks.
"""
import argparse
import json
import time
from pathlib import Path
from verify import read_inputs, sha, trip_bits

def solve(masks, anchor, n, fleet):
    routes=sorted(m for m in masks if m & anchor)
    universe=(1<<n)-1
    ids={mask:i for i,mask in enumerate(routes)}
    by_trip=[0]*n
    for j,mask in enumerate(routes):
        for t in trip_bits(mask): by_trip[t.bit_length()-1]|=1<<j
    order=sorted(range(n),key=lambda t:by_trip[t].bit_count())
    incompatible=[]
    for mask in routes:
        conflicts=0
        for t in trip_bits(mask): conflicts|=by_trip[t.bit_length()-1]
        incompatible.append(conflicts)
    nodes=[0]*(fleet+1); tested=0
    stack=[(universe,(1<<len(routes))-1,fleet)]
    while stack:
        left,alive,free=stack.pop(); nodes[fleet-free]+=1
        rare_count=len(routes)+1
        for t in order:
            if not (left>>t)&1: continue
            support=by_trip[t]&alive
            size=support.bit_count()
            if size < rare_count:
                rare_count=size; best=support
                if size==0: break
        if rare_count==0: continue
        if free==2:
            for bit in trip_bits(best):
                tested+=1
                other=left ^ routes[bit.bit_length()-1]
                if other in ids:
                    return True,nodes,tested,len(routes)
        else:
            for bit in trip_bits(best):
                j=bit.bit_length()-1
                stack.append((left ^ routes[j],alive & ~incompatible[j],free-1))
    return False,nodes,tested,len(routes)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('instance','journal','summary','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r,masks,anchors,n,fleet=read_inputs(a.instance,a.journal,a.summary)
    start=time.monotonic();found,nodes,tested,count=solve(masks,anchors[0],n,fleet)
    r.update(method='single peak, explicit disjoint-union states, exact complement lookup for final pair',
             filtered_distinct_trip_sets=count,used_peak_sets=1,eight_bus_exact_partition_exists=found,
             exhausted=not found,nodes_by_depth=nodes,complement_tests=tested,elapsed_seconds=time.monotonic()-start,
             verification_source_sha256=sha(Path(__file__).read_bytes()),
             input_adapter_sha256=sha(Path(__file__).with_name('verify.py').read_bytes()))
    a.output.write_text(json.dumps(r,indent=2)+'\n'); print(json.dumps(r))
