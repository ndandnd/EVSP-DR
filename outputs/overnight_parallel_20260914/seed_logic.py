"""Deterministic previous-instance sequence subsets; this does not run CG."""
import math

def choose(parent, mip):
    assert mip['physical_replay_validated'] is True
    integer=mip['selected_routes']
    n=len(integer)
    assert n==mip['buses'] and n>0
    allowed=set(parent['trip_ids'])
    def clean(r):
        ts=r['trips'];cost=float(r['cost'])
        assert ts and len(set(ts))==len(ts) and set(ts)<=allowed and math.isfinite(cost)
        return {'trips':list(ts),'cost':cost}
    integer=[clean(r) for r in integer]
    lp=parent['final_lp'];assert lp['artificial_total']==0
    positives=[r for r in lp['positive_routes'] if float(r['value'])>1e-9]
    # Weight descending; ordered local trip sequence ascending; cost ascending.
    positives=sorted(positives,key=lambda r:(-float(r['value']),tuple(r['trips']),float(r['cost'])))
    selected=[];seen=set()
    for r in positives:
        key=frozenset(r['trips'])
        if key not in seen:
            selected.append(clean(r));seen.add(key)
        if len(selected)==n:break
    assert len(selected)==n
    assert len({frozenset(r['trips']) for r in integer})==n
    assert {tuple(r['trips']) for r in integer}!={tuple(r['trips']) for r in selected},'Identical pair is not an independent treatment'
    return {'integer':integer,'lpweight':selected}

def status(parent,journal,arm,audit):
    return dict(artifact_kind='previous_instance_sequence_subset',optimization_run=False,
        certified_rc_optimal=False,csv=parent['csv'],trip_ids=parent['trip_ids'],
        columns_journal=str(journal),columns=audit['selected_sequence_count'],
        provenance={'instance_sha256':parent['provenance']['instance_sha256']},
        treatment=arm,seed_construction=audit)
