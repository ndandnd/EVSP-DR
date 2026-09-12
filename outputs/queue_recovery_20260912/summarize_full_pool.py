"""Build a compact, dated view of the registered full-pool recovery campaign."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

p = argparse.ArgumentParser()
p.add_argument('snapshot', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
s = json.loads(a.snapshot.read_text())
c = s['campaigns']['full_pool_recovery_20260912']
a.output.mkdir(parents=True, exist_ok=True)

def case(row):
    parts = Path(row['path']).parts
    return parts[parts.index('cases') + 1]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

old = {case(r): r for r in s['campaigns']['overnight_extension_20260912']['mip']}
cg = {case(r): r for r in c['cg']}
rows = []
for m in c['mip']:
    key = case(m)
    g, o = cg[key], old.get(key, {})
    f, audit = g.get('final') or {}, g.get('inherited_event_pool_audit') or {}
    row = dict(case=key, chain=int(key[1]), target_k=int(key[-2:]),
               new_buses=m.get('buses'), old_bounded_buses=o.get('buses'),
               old_fleet_proven=o.get('fleet_proven'), old_fleet_bound=o.get('fleet_bound'),
               cg_certified=g.get('certified_rc_optimal'), cg_minutes=(g.get('wall_s') or 0) / 60,
               cg_iterations=f.get('iter'), weighted_lp_objective=f.get('lp_obj'),
               fractional_route_weight=f.get('route_weight'), minimum_reduced_cost=f.get('min_rc'),
               accepted_inherited_routes=audit.get('accepted_columns'),
               import_minutes=(audit.get('total_import_s') or 0) / 60,
               cg_path=g['path'], cg_payload_sha256=digest(g),
               old_mip_path=o.get('path'), old_mip_sha256=o.get('sha256'))
    for field in ['fleet_bound', 'fleet_proven', 'pool_columns', 'physical_replay_validated',
                  'duplicate_trip_removal_validated', 'cross_route_charger_capacity_validated',
                  'runtime_s', 'status_name', 'path', 'sha256']:
        row['mip_' + field] = m.get(field)
    row['input_sha256'] = (m.get('physical_pool_audit') or {}).get('input_hashes', {}).get('instance_sha256')
    for field in ['stage1_runtime_s', 'stage1_status_name', 'stage2_status_name', 'stage2_fleet_constraint']:
        row[field] = (m.get('two_stage') or {}).get(field)
    rows.append(row)
rows.sort(key=lambda r: (r['chain'], r['target_k']))
if len({r['case'] for r in rows}) != len(rows):
    raise ValueError('Multiple MIP artifacts for a case require an explicit authority decision')
matched = [r for r in rows if r['new_buses'] == r['target_k'] and r['mip_physical_replay_validated']]
highest = [max([r for r in matched if r['chain'] == chain], key=lambda r: r['target_k'])
           for chain in range(1, 7) if any(r['chain'] == chain for r in matched)]
for name, values in [('integer_results', rows), ('highest_matches', highest)]:
    with (a.output / (name + '.csv')).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(values)
payload = dict(source_snapshot=str(a.snapshot), source_snapshot_sha256=hashlib.sha256(a.snapshot.read_bytes()).hexdigest(),
               timestamp_utc=s['timestamp_utc'], cg_code='e091a4dba549510238507ef5e5367abea958bd30',
               mip_code='871d057e1067411f09581e37d78f7c1ca43f68bb', rows=rows)
(a.output / 'results.json').write_text(json.dumps(payload, indent=2) + '\n')
certified = sum(r.get('certified_rc_optimal') is True for r in c['cg'])
lines = ['# Full-pool chain results', '', 'Evidence captured ' + s['timestamp_utc'] + '.', '',
         f'**{len(matched)} of {len(rows)} completed MIPs match their fleet targets.** '
         f'**{certified} of 37 CG cases** have pricing certificates. Unfinished cases are not failed target matches.', '',
         '| Chain | Highest target matched | Full-pool buses | Earlier bounded buses at that target | CG minutes |',
         '|---|---:|---:|---:|---:|']
for r in highest:
    lines.append(f"| {r['chain']} | {r['target_k']} | {r['new_buses']} | {r['old_bounded_buses']} | {r['cg_minutes']:.1f} |")
lines += ['', 'The table shows the highest completed match per chain, not a computational threshold or independent statistical replications. '
          '[All results and source hashes](results.json), [editable CSV](integer_results.csv).', '',
          '## Settings and proof limits', '',
          'Set covering; unlimited inherited sequences checked by the fixed-sequence index; 240 kWh batteries; 240 kW charging; '
          '2.5 kWh / 5-minute event graph; flat prices; no shared-station capacity or return-SOC floor. No GIRO solution columns are injected. '
          'CG uses 100,000 per bus plus electricity and 5 per charge start. MIP stage 1 minimizes fleet for up to 1,800 seconds; '
          'stage 2 imposes fleet ≤ the validated incumbent and minimizes charging with the remainder of the 3,600-second budget. '
          'CG minutes include importing routes and use existing graph caches; original graph construction is excluded.', '',
          'Read the separate certificate, pool proof and physical-check fields in results.json. A fleet proof concerns the accepted column pool. '
          'A CG pricing certificate concerns its represented graph and reduced-cost tolerance. Neither is a full-model integer proof. '
          'Individual-route replay, duplicate-trip removal and shared-capacity checks are separate. '
          'A final MIP TIME_LIMIT can refer to charging optimization after stage 1 has proved the fleet.', '',
          '## Diagnosis', '',
          'C4 k10 previously had a pool proved to require 11 buses; the full-pool run permits 10 at the same certified LP objective. '
          'C1 k8 also improves a proved pool minimum from 9 to 8. Other old incumbents were sometimes unproved, so MIP search difficulty remains relevant. '
          'Do not attribute every change in an inherited chain to one isolated code modification. '
          '[Initial matched-parent audit](../status_20260912T220047Z/README.md).', '',
          '## Execution', '',
          'This summarizer does not submit or retry jobs. Consult the dated collector snapshot and its delta for scheduler transitions. '
          'The separate bounded chain 1 k15 MIP ended with 18 buses and pool bound 15, without a fleet proof; its individual-route replay passed. '
          'It must not be mixed with the new full-pool results. Held historical jobs and V2G work remain protected.']
(a.output / 'README.md').write_text('\n'.join(lines) + '\n')
print(json.dumps(dict(mips=len(rows), target_matches=len(matched), cg_certificates=certified,
                      pool_proofs=sum(r['mip_fleet_proven'] is True for r in rows), output=str(a.output))))
