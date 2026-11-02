"""Verify the archived eight-bus set-partition claim without disclosing route/trip records.

Inputs: the source CSV in master row order, the owner CG summary and the gzipped
post-dive column journal. Outputs contain hashes, counts and proof results only.
Standard-library search groups columns by which trip they serve in ONE peak
minute and enumerates the eighth route by a dictionary complement lookup. The
optional SciPy LP runs on ALL route masks with only trip equations and fleet size.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

EXPECTED = {
    'instance': 'c1dbe1dd15d20979799c68d01430a5fd0eddc2ca3ced3f46bd8522123e6c480d',
    'owner_summary': '209bde0a89b1bc2be8e0cb7df33280c23aa195b75fb54cd985b65e071e2a5c54',
    'post_dive': '5744645fa8fc62e3c79e3880a12b1388dda8e0342fe46bb6c700d61e98c86d0d',
}

def require(check, message):
    if not check:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def clock_minutes(value):
    hh, mm = map(int, value.split(':'))
    require(hh >= 0 and 0 <= mm < 60, 'bad clock')
    return 60 * hh + mm

def read_inputs(instance, journal, summary):
    raw, meta = instance.read_bytes(), summary.read_bytes()
    require(sha(raw) == EXPECTED['instance'] and sha(meta) == EXPECTED['owner_summary'], 'source hash mismatch')
    meta = json.loads(meta)
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    intervals = [(clock_minutes(r['Start1']),clock_minutes(r['End1'])) for r in rows]
    require(all(a < b for a,b in intervals), 'non-positive trip')
    by_minute = [sum(1 << t for t,(a,b) in enumerate(intervals) if a <= m < b)
                 for m in range(min(a for a,_ in intervals), max(b for _,b in intervals))]
    peak = max(v.bit_count() for v in by_minute)
    anchor_masks = list(dict.fromkeys(v for v in by_minute if v.bit_count() == peak))
    # This check is much stronger than needed to use a peak: check every parsed
    # trip pair consecutive in temporal order, on every archived route.
    masks = set()
    digest, root_digest = hashlib.sha256(), hashlib.sha256()
    count = eligible_records = 0
    with gzip.open(journal, 'rb') as f:
        for count, line in enumerate(f,1):
            digest.update(line)
            if count <= meta['columns']:
                root_digest.update(line)
            r = json.loads(line)
            trips = r['trips']
            require(trips and all(type(t) is int and 0 <= t < len(rows) for t in trips), 'invalid indices')
            ordered = sorted(intervals[t] for t in trips)
            require(all(e <= a for (_,e),(a,_) in zip(ordered, ordered[1:])), 'trip conflict or duplicate')
            mask = sum(1 << t for t in trips)
            masks.add(mask)
            if all(mask & anchor for anchor in anchor_masks):
                eligible_records += 1
    require(count == meta['dive']['columns_after_dive'] and digest.hexdigest() == EXPECTED['post_dive'], 'full pool mismatch')
    require(root_digest.hexdigest() == meta['columns_sha256'], 'root prefix mismatch')
    report = dict(instance_sha256=sha(raw), owner_summary_sha256=EXPECTED['owner_summary'],
                  post_dive_journal_sha256=digest.hexdigest(), root_prefix_sha256=root_digest.hexdigest(),
                  journal_columns=count, root_prefix_columns=meta['columns'], distinct_trip_sets=len(masks),
                  trips=len(rows), maximum_simultaneous_service=peak, distinct_maximum_sets=len(anchor_masks),
                  all_journal_routes_trip_valid_and_temporally_disjoint=True,
                  records_touching_all_maximum_sets=eligible_records)
    return report, sorted(masks), anchor_masks, len(rows), peak

def trip_bits(mask):
    while mask:
        b = mask & -mask
        yield b
        mask ^= b

def linear_relaxation(masks, ntrips, peak):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    import scipy
    import numpy as np
    ii, jj = [], []
    for j,mask in enumerate(masks):
        ii.extend(b.bit_length()-1 for b in trip_bits(mask)); ii.append(ntrips)
        jj.extend([j] * (mask.bit_count()+1))
    matrix = coo_matrix((np.ones(len(ii)),(ii,jj)), shape=(ntrips+1,len(masks))).tocsc()
    answer = linprog(np.zeros(len(masks)), A_eq=matrix, b_eq=[1]*ntrips+[peak],
                     method='highs', options={'time_limit':90., 'disp':False})
    return dict(scipy_version=scipy.__version__,status=int(answer.status),message=answer.message,
                rows=ntrips+1,columns=len(masks),fractional_solution=answer.success)

def enumerate_by_bus(available, eligible, trip_rows, group_ids, full, fleet_size):
    # Variable = one route per anchor trip, not an uncovered service trip. Every
    # feasible fleet induces exactly one choice in each group. At a node mask
    # frequencies can force one bus to supply a trip unavailable from any other.
    # Those requirements can be propagated without removing a valid solution.
    nodes = [0]*(fleet_size+1)
    def walk(group_ids, alive, uncovered):
        nodes[fleet_size-len(group_ids)] += 1
        if len(group_ids)==1:
            return uncovered in eligible
        domains = [(g, alive & available[g]) for g in group_ids]
        if any(d==0 for _,d in domains): return False
        if len(group_ids) <= 5:
            forced = [0]*len(domains)
            for t in trip_bits(uncovered):
                supporters = trip_rows[t.bit_length()-1] & alive
                if not supporters: return False
                # Only force a unique supporting column. The next iteration can
                # expose further forced routes, which are handled by recursion.
                if supporters & (supporters-1): continue
                for ix,(_,d) in enumerate(domains):
                    if d & supporters:
                        if forced[ix] and forced[ix]!=supporters: return False
                        forced[ix]=supporters
                        break
            domains = [(g, f or d) for (g,d),f in zip(domains,forced)]
        g,domain = min(domains,key=lambda kv:kv[1].bit_count())
        rest=tuple(i for i in group_ids if i!=g)
        for c in trip_bits(domain):
            mask = by_col[c.bit_length()-1]
            to_remove=0
            for t in trip_bits(mask): to_remove |= trip_rows[t.bit_length()-1]
            if walk(rest,alive & ~to_remove, uncovered ^ mask): return True
        return False
    by_col = list(eligible)  # eligible is ordered identically to the domain masks.
    found=walk(group_ids,(1<<len(eligible))-1,full)
    return found,nodes

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('instance','journal','summary','output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--lp-only',action='store_true'); a=p.parse_args()
    report,masks,anchors,n,fleet=read_inputs(a.instance,a.journal,a.summary)
    if a.lp_only:
        report['linear_relaxation']=linear_relaxation(masks,n,fleet)
    else:
        eligible=dict.fromkeys(mask for mask in masks if all(mask & anchor for anchor in anchors))
        groups = list(trip_bits(anchors[0]))
        trip_rows=[0]*n
        for col,mask in enumerate(eligible):
            for t in trip_bits(mask): trip_rows[t.bit_length()-1] |= 1<<col
        domains=[trip_rows[g.bit_length()-1] for g in groups]
        start=time.monotonic()
        found,nodes=enumerate_by_bus(domains,eligible,trip_rows,tuple(range(fleet)),(1<<n)-1,fleet)
        report.update(solver='exhaustive anchor-bus domains with final complement lookup',
                      eligible_distinct_trip_sets=len(eligible),nodes_by_depth=nodes,
                      completed=True,eight_bus_partition_exists=found,seconds=time.monotonic()-start)
    report['verification_source_sha256']=sha(Path(__file__).read_bytes())
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
