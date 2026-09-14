"""Report verified strict-pilot endpoints without conflating proof scopes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    pilot = snapshot['campaigns']['strict_capacity_parallel_20260914']
    followup = snapshot['campaigns']['strict_capacity_mip1h_20260914']
    manifest = pilot['workflow']['manifest.json']
    indexed = {}
    for group, campaign in [('pilot', pilot), ('matched', followup)]:
        for record in campaign['records']:
            assert record['stage_completion_verified'] is True
            if group == 'matched':
                assert record['source_cg_binding_verified'] and record['no_new_cg']
            key = (group, record['case_id'], record['phase'])
            assert key not in indexed, key
            indexed[key] = record
    rows = []
    for case in manifest['cases']:
        cid = case['case_id']
        cr = indexed.get(('pilot', cid, 'cg'), {})
        mr = indexed.get(('matched', cid, 'mip'), {})
        cg, mip = cr.get('result', {}), mr.get('result', {})
        result = mip.get('result', {})
        audit = mip.get('physical_station_capacity_audit', {})
        violations = [f"{station}: {value['peak_simultaneous_connections']} connections / {value['documented_chargers']} chargers"
                      for station, value in audit.get('stations', {}).items()
                      if value.get('valid') is False]
        rows.append({
            'case_id': cid, 'target_buses': 1 if case['instance'].startswith('k1') else 2,
            'battery_kwh': case['battery_kwh'], 'minimum_soc_kwh': case['reserve_kwh'],
            'parx_power_kw': 60 if case['arm'] in ('parx60', 'combined') else 240,
            'capacity_enforced': case['arm'] in ('capacity', 'combined'),
            'pricing_selector': case['capacity_selector'], 'tariff': case['prices'],
            'cg_minutes': cg.get('runtime_s', 0) / 60 if cg else None,
            'cg_iterations': cg.get('iteration_count'),
            'cg_pricing_certificate': cg.get('certified_rc_optimal'),
            'cg_stop_reason': cg.get('stop_reason'),
            'weighted_lp_objective': cg.get('final', {}).get('objective'),
            'fractional_buses': cg.get('final', {}).get('route_weight'),
            'integer_buses': result.get('fleet'),
            'fleet_proved_in_pool': result.get('stage1', {}).get('fleet_proven'),
            'charging_related_cost': result.get('stage2', {}).get('charging_cost'),
            'charging_proved_in_pool': result.get('stage2', {}).get('status') == 'OPTIMAL' if result else None,
            'shared_capacity_check': audit.get('valid'),
            'capacity_violations': '; '.join(violations),
            'cg_path': cr.get('path'), 'cg_sha256': cr.get('sha256'),
            'matched_mip_path': mr.get('path'), 'matched_mip_sha256': mr.get('sha256'),
            'cg_allowance_s': case['cg_wall_s'], 'matched_mip_allowance_s': 3600,
        })
    out = Path(__file__).parent / ('status_' + args.snapshot.stem)
    out.mkdir(exist_ok=True)
    with (out / 'results.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Charging-constraint pilot: verified results', '',
        f"Source collection started {snapshot['timestamp_utc']}. Missing results are pending, not infeasible.", '',
        '| Case | CG minutes | Pricing certified | Matched MIP buses | Fleet proved in pool | Charging-related cost | Shared capacity check |',
        '|---|---:|---|---:|---|---:|---|',
    ]
    def show(value):
        return 'pending' if value is None else ('yes' if value is True else 'no' if value is False else str(value))
    for row in rows:
        minutes = round(row['cg_minutes'], 1) if row['cg_minutes'] is not None else None
        cost = round(row['charging_related_cost'], 3) if row['charging_related_cost'] is not None else None
        lines.append('| ' + ' | '.join(show(v) for v in [row['case_id'], minutes,
            row['cg_pricing_certificate'], row['integer_buses'], row['fleet_proved_in_pool'],
            cost, row['shared_capacity_check']]) + ' |')
    lines += ['',
        'All four completed k2 controls match two buses and have pricing certificates. Their matched one-hour MIPs prove fleet and charging objectives within their respective saved pools. Shared capacity was disabled: each selected solution has two simultaneous connections at station 2190L, where the documented limit is one. These results do not establish feasibility with station capacity enforced. Those four k2 treatments remain separate.', '',
        'The 236.44-kWh treatment also applies a 35.466-kWh (15%) reserve. It changes battery and reserve together. All cases use constant charging power and no 65% terminal target. Individual route feasibility in this dedicated solver is by construction; it is not a separate continuous replay audit.', '',
        'Capacity/combined k2 cases receive 220 minutes of CG versus 110 for baseline/PARX-only. The matched MIPs equalize only final integer-search allowances. These physics cells are feasibility pilots, not isolated runtime-causal estimates. Reference-versus-cached-pricing pairs have matching settings; these are single runs, not controlled runtime repetitions.', '',
        'Charging-related cost includes electricity and the modeled charging-start fee. A pool proof is distinct from the CG pricing certificate and from shared-capacity validation.', '',
        '[Editable result table and source hashes](results.csv).',
    ]
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    (out / 'validation.json').write_text(json.dumps({
        'snapshot': str(args.snapshot), 'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
        'expected_cases': len(rows), 'verified_cg_endpoints': sum(r['cg_path'] is not None for r in rows),
        'verified_matched_mips': sum(r['matched_mip_path'] is not None for r in rows),
        'source_completion_and_parent_bindings_checked': True,
    }, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
