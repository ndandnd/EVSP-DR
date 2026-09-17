"""Read-only local verification of supplied partitions; no solver or cluster."""
import csv, json, sys, hashlib
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent
REVIEW = BASE.parents[1]
ROOT = REVIEW.parents[1]
SOURCE = REVIEW / 'advisor_geography_review_20260917'
sys.path.insert(0, str(REVIEW / 'time_only_vsp_20260916'))
import time_only_vsp as v

refs, scale, direct, closure, resolve = v.travel_data()
rows = v.rows(SOURCE / 'inputs/parent32.csv')
trips = [dict(start=v.minutes(r['Start1']), end=v.minutes(r['End1']),
              start_ref=resolve(r['From1']), end_ref=resolve(r['To1'])) for r in rows]
adj = v.graph(trips, closure, scale)
parent = v.maximum_matching(adj)
assert parent['minimum_path_cover'] == 32
assert sum(map(len, adj)) == 261078
certificates = {'parent': parent}

def check(name, labels, expected):
    parts = []
    for label in sorted(set(labels), key=str):
        members = [i for i, x in enumerate(labels) if x == label]
        local = {x: i for i, x in enumerate(members)}
        sub = [[local[j] for j in adj[i] if j in local] for i in members]
        cert = v.maximum_matching(sub)
        cert['parent_indices'] = members
        parts.append(cert)
    value = sum(p['minimum_path_cover'] for p in parts)
    assert value == expected, (name, value, expected)
    certificates[name] = parts
    return dict(partition=name, component_minima=[p['minimum_path_cover'] for p in parts],
                sum_minima=value, increase=value - 32)

table = []
oids = [int(r['Ordered_Trip_ID']) for r in rows]
for p in range(10):
    mapping = {}
    for g in range(4):
        for r in v.rows(SOURCE / f'inputs/d{p:02d}_g{g}.csv'):
            oid = int(r['Ordered_Trip_ID']); assert oid not in mapping; mapping[oid] = g
    assert set(mapping) == set(oids)
    table.append(check(f'd{p:02d}', [mapping[o] for o in oids], 32))
labels = json.loads((SOURCE / 'partition_labels.json').read_text())['labels']
for name, expected in [('vehicle_group',32),('charger_affinity',34),('spectral_2',40),('spectral_4',68)]:
    table.append(check(name, labels[name], expected))
table.append(check('time_split_AMPM', [int(t['start'] >= 13*60) for t in trips], 64))
cover = json.loads((SOURCE / 'cover_partition.json').read_text())
table.append(check('time_only_cover', cover['labels'], 32))
index = {int(r['count_trip_id']): i for i,r in enumerate(rows)}
paths = [[index[x] for x in p] for p in cover['paths']]
assert len(paths)==32 and sorted(i for p in paths for i in p)==list(range(750))
assert all(b in adj[a] for p in paths for a,b in zip(p,p[1:]))
assert all(len({cover['labels'][i] for i in p})==1 for p in paths)

csv.field_size_limit(100000000)
register = ROOT / 'outputs/research_register/register.csv'
records = [r for r in v.rows(register) if r['campaign_id']=='overnight_extension_20260912' and r['case_id'].startswith('d0')]
cg = [r for r in records if r['stage']=='cg']; mip = [r for r in records if r['stage']=='mip']
summary = dict(verification='PASS', cluster_submissions=0, external_solver_calls=0,
              partitions=table, cg_count=len(cg), cg_stop_counts=dict(Counter(r['stop_reason'] for r in cg)),
              above_eight_route_weight=[dict(case=r['case_id'], weight=r['fractional_fleet']) for r in cg if float(r['fractional_fleet'])>8.000001],
              mip_counts=[dict(incumbent=k[0], pool_bound=k[1], count=n) for k,n in Counter((int(r['mip_incumbent_fleet']),round(float(r['mip_bound_fleet']))) for r in mip).items()],
              sources={str(p.relative_to(ROOT)):v.sha(p) for p in [register, SOURCE/'README.md', SOURCE/'partition_labels.json', SOURCE/'cover_partition.json',SOURCE/'inputs/parent32.csv',Path(__file__)]})
(BASE/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
(BASE/'certificates.json').write_text(json.dumps(certificates,separators=(',',':'))+'\n')
print(json.dumps(summary,indent=2))
