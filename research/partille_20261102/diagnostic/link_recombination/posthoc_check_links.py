#!/usr/bin/env python3
"""Read-only, stdlib post hoc witness-link audit. Writes only ID-free aggregate.

Regenerate GIRO passenger sequences from original CSV rows. Source route `trips`
are local zero-based instance rows, mapped to unique Ordered_Trip_ID values before
ordered adjacency comparison. Ignore charging nodes, source, and sink: a link is
an adjacent passenger pair even when a route charges between those passengers.
No graph construction, pricing, optimization, replay, or cluster operations.
"""
import argparse, collections, csv, datetime, gzip, hashlib, json, time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def integer(token):
    # CSV numeric IDs may be represented as integral floats; reject truncation.
    from decimal import Decimal
    value = Decimal(str(token).strip())
    assert value.is_finite() and value == value.to_integral_value(), 'noninteger ID'
    return int(value)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    tick = time.monotonic()
    private = args.repo/'outputs/partille_validation_20261102/private/c2'
    data = args.repo/'.codex-work/capacity-packed-20261001/data'
    files = dict(pool=private/'frozen_columns.jsonl.gz', summary=private/'owner_summary.json',
                 membership=private/'pool_membership.json', replay=private/'local_owner_inputs_verified.json',
                 selected=private/'witness_local/selected_routes.json',
                 instance=data/'scale_ladder/instances/nested_probability_k2_15_20260908/Practice_Custom_DutyUnion_k08_p02_20260908.csv',
                 duty_source=data/'Par_VehicleDetails_Updated.csv')
    assert all(p.is_file() for p in files.values()), 'source missing'
    hashes = {k:digest(p) for k,p in files.items()}
    summary, membership, replay, selected = [json.loads(files[k].read_bytes()) for k in ('summary','membership','replay','selected')]
    assert hashes['summary'] == membership['source_summary_sha256'] == replay['summary_sha256']
    assert hashes['instance'] == replay['instance_sha256']
    assert hashes['duty_source'] == replay['duty_source_sha256']
    assert hashes['selected'] == replay['selected_sha256']
    assert replay['validated'] and replay['exact_once'] and replay['fleet']==8 and replay['trip_count']==173
    assert any(Path(p).name == files['instance'].name and h == hashes['instance'] for p,h in summary['input_hashes'].items())
    with files['instance'].open(newline='') as f: instance=list(csv.DictReader(f))
    with files['duty_source'].open(newline='') as f: master=list(csv.DictReader(f))
    ordered=[integer(row['Ordered_Trip_ID']) for row in instance]
    assert len(ordered)==len(set(ordered))==173, 'canonical trip IDs not unique'
    regular=[r for r in master if r['Identifier']=='Regular' and r['Ordered_Trip_ID'].strip()]
    by_ordered={}
    for r in regular:
        trip=integer(r['Ordered_Trip_ID'])
        assert trip not in by_ordered, 'ambiguous master trip ID'
        by_ordered[trip]=r['VehicleTask'].strip()
    # Same first-occurrence duty order as importer; preserve original duty row order.
    duty_order=list(dict.fromkeys(by_ordered[t] for t in ordered))
    witnesses=[tuple(integer(r['Ordered_Trip_ID']) for r in regular if r['VehicleTask'].strip()==d) for d in duty_order]
    assert len(witnesses)==8 and collections.Counter(t for w in witnesses for t in w)==collections.Counter(ordered)
    def minutes(s):
        h,m=map(int,s.split(':')); return 60*h+m
    starts={t:minutes(r['Start1']) for t,r in zip(ordered,instance)}
    assert all([starts[t] for t in w]==sorted(starts[t] for t in w) for w in witnesses)
    def canonical(route):
        seq=route['trips']
        assert all(type(t) is int and 0<=t<len(ordered) for t in seq), 'noncanonical source trip index'
        assert len(seq)==len(set(seq)), 'repeated route trip'
        return tuple(ordered[t] for t in seq)
    assert collections.Counter(canonical(r) for r in selected)==collections.Counter(witnesses), 'independent witness mismatch'
    links=set(); counts=collections.Counter({w:0 for w in witnesses})
    full=hashlib.sha256(); prefix=hashlib.sha256(); n=0
    with gzip.open(files['pool'],'rb') as stream:
        for n,line in enumerate(stream,1):
            full.update(line)
            if n<=summary['columns']: prefix.update(line)
            seq=canonical(json.loads(line))
            links.update(zip(seq,seq[1:]))
            if seq in counts: counts[seq]+=1
    assert n==39390==membership['column_count']==summary['dive']['columns_after_dive']
    assert full.hexdigest()==membership['source_pool_sha256']
    assert prefix.hexdigest()==membership['root_prefix_sha256']==summary['columns_sha256']
    assert sorted(counts.values())==membership['matches'] and sum(c>0 for c in counts.values())==membership['present']==1
    rows=[]; witness_links=set()
    for ordinal,w in enumerate(witnesses,1):
        pairs=set(zip(w,w[1:])); witness_links.update(pairs)
        missing=pairs-links
        rows.append(dict(witness_ordinal=ordinal,passenger_trips=len(w),distinct_links=len(pairs),
                         links_present=len(pairs)-len(missing),links_missing=len(missing),
                         full_sequence_matches=counts[w],full_sequence_present=counts[w]>0,
                         all_links_present=not missing))
    result=dict(schema='evsp-dr-local-witness-passenger-link-audit-v1',started_utc=started,
                inputs_sha256=hashes,uncompressed_pool_sha256=full.hexdigest(),root_prefix_sha256=prefix.hexdigest(),
                archive_routes=n,root_prefix_routes=summary['columns'],passenger_trips=len(ordered),
                independently_regenerated_witnesses=len(witnesses),witness_partition_exact_once=True,
                independently_regenerated_matches_saved_selection=True,
                full_sequences_present=sum(r['full_sequence_present'] for r in rows),
                full_sequences_missing=sum(not r['full_sequence_present'] for r in rows),
                pool_distinct_passenger_links=len(links),witness_distinct_links=len(witness_links),
                witness_links_present=len(witness_links & links),witness_links_missing=len(witness_links-links),
                missing_full_sequences_with_all_links_present=sum(not r['full_sequence_present'] and r['all_links_present'] for r in rows),
                per_witness=rows,elapsed_seconds=time.monotonic()-tick,
                limits=['Passenger adjacency only; charging nodes between passengers are ignored.',
                        'Union-of-links availability does not imply one archived route or compatible charging schedule contains the full witness.',
                        'No physical feasibility, charging compatibility, optimality, pricing, or full-model certificate is established by this scan.'])
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
