"""Validate and summarize published evening experiments from one snapshot."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    source = json.loads(raw)
    project = Path(__file__).resolve().parents[2]
    out = Path(__file__).parent / ('status_' + args.snapshot.stem)
    out.mkdir(exist_ok=True)
    register = json.loads((project / 'outputs/research_register/register.json').read_text())
    assert register['source_snapshot']['sha256'] == hashlib.sha256(raw).hexdigest()
    normalized = {(r['campaign_id'], r['stage'], r['source_path']): r
                  for r in register['rows']}
    checks = []
    for name in ['compact_seed_support_20260914', 'remaining_chain_gaps_20260914',
                 'chain_extension_20260914', 'overnight_parallel_20260914']:
        for stage in ['cg', 'mip']:
            for item in source['campaigns'][name].get(stage, []):
                row = normalized[name, stage, item['path']]
                if item.get('sha256'):
                    assert row['source_sha256'] == item['sha256']
                if stage == 'mip':
                    expected = dict(mip_incumbent_fleet=item['buses'],
                                    mip_bound_fleet=item['fleet_bound'],
                                    fleet_proven=item['fleet_proven'],
                                    physical_selected_validated=item['physical_replay_validated'])
                else:
                    lp, final = item.get('final_lp') or {}, item.get('final') or {}
                    expected = dict(full_model_lp_certified=item['certified_rc_optimal'],
                                    stop_reason=item['stop_reason'],
                                    weighted_lp_objective=lp.get('objective', final.get('lp_obj')),
                                    fractional_fleet=lp.get('route_weight', final.get('route_weight')))
                assert all(row.get(k) == v for k, v in expected.items()), (name, item['path'])
                checks.append(dict(campaign=name, stage=stage, path=item['path'], exact_match=True))

    campaign = source['campaigns']['compact_seed_support_20260914']
    manifest = campaign['workflow']['manifest.json']
    manifest_sha = hashlib.sha256((Path(__file__).parent.parent /
                                  'compact_seed_support_20260914/manifest.json').read_bytes()).hexdigest()
    cgs = {Path(r['path']).parent.name: r for r in campaign['cg']}
    mips = {Path(r['path']).parent.name.removesuffix('_mip'): r for r in campaign['mip']}
    rows = []
    for cid, case in manifest['cases'].items():
        if case['kind'] != 'cg':
            continue
        cg, mip = cgs.get(cid), mips.get(cid)
        if cg:
            assert cg['completion_marker_matches']
            assert cg['publication_manifest_sha256'] == manifest_sha
            assert cg['provenance']['instance_sha256'] == case['input_sha256']
            assert cg['provenance']['git_commit'] == case['execution_commit']
            audit = cg['inherited_event_pool_audit']
            assert audit['source_status_sha256'] == case['seed_status_sha256']
            assert audit['source_journal_sha256'] == case['seed_journal_sha256']
            assert audit['selected_for_replay'] == case['seed']['selected_sequence_count']
            assert audit['unprocessed_selected'] == audit['rejected_columns'] == 0
            assert not audit['inherited_lp_certificate']
        if mip:
            assert cg is not None
            assert mip['completion_marker_matches']
            assert mip['publication_manifest_sha256'] == manifest_sha
            assert mip['source_status_sha256'] == cg['sha256']
            assert mip['source_journal_sha256'] == cg['columns_journal_sha256']
            audit = mip['physical_pool_audit']
            assert audit['input_hashes']['instance_sha256'] == case['input_sha256']
            assert audit['rejected_columns'] == audit['deterministically_repaired'] == 0
            assert mip['physical_replay_validated']
        final = (cg or {}).get('final') or {}
        lp = (cg or {}).get('final_lp') or {}
        rows.append(dict(case_id=cid, pair_id=case['pair_id'], chain=case['chain'],
                         target_k=case['target_k'], treatment=case['treatment'],
                         inherited_sequences=case['seed']['selected_sequence_count'],
                         cg_minutes=cg['wall_s'] / 60 if cg else None,
                         cg_certified=cg['certified_rc_optimal'] if cg else None,
                         stopping_reason=cg['stop_reason'] if cg else None,
                         weighted_lp_objective=lp.get('objective', final.get('lp_obj')),
                         fractional_route_weight=lp.get('route_weight', final.get('route_weight')),
                         minimum_reduced_cost=final.get('min_rc'),
                         buses=mip['buses'] if mip else None,
                         pool_fleet_bound=mip['fleet_bound'] if mip else None,
                         fleet_proved_in_pool=mip['fleet_proven'] if mip else None,
                         individual_route_replay=mip['physical_replay_validated'] if mip else None,
                         mip_minutes=mip['runtime_s'] / 60 if mip else None,
                         input_sha256=case['input_sha256'],
                         cg_path=cg['path'] if cg else None,
                         cg_sha256=cg['sha256'] if cg else None,
                         mip_path=mip['path'] if mip else None,
                         mip_sha256=mip['sha256'] if mip else None))
    rows.sort(key=lambda r: (r['target_k'], r['chain'], r['inherited_sequences']))
    with (out / 'compact_seed_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    groups = collections.defaultdict(dict)
    for row in rows:
        groups[row['pair_id']]['core512' if row['inherited_sequences'] == 512 else 'core'] = row
    def value(v):
        return 'pending' if v is None else str(v)
    def minutes(v):
        return 'pending' if v is None else f'{v:.1f}'
    lines = ['# Evening experiments — verified results', '',
             f'Snapshot: {source["timestamp_utc"]}. Counts include only published endpoints.', '',
             f'**Compact seeds: {len(cgs)}/36 CG endpoints, '
             f'{sum(bool(r["certified_rc_optimal"]) for r in cgs.values())} certified; '
             f'{len(mips)}/36 MIP endpoints, '
             f'{sum(r["buses"] == r["target_k"] for r in rows if r["buses"] is not None)} matching target.**', '',
             'The core preserves the previous integer solution and every positive-weight LP route. '
             'The other treatment keeps that core and fills to 512 distinct trip sets. '
             'Both use the same input, code and budgets. Prior computation is recorded separately.', '',
             '| Case | Core routes | Buses: core | Buses: 512 | CG minutes: core | CG minutes: 512 |',
             '|---|---:|---:|---:|---:|---:|']
    for pair, arms in sorted(groups.items(), key=lambda p: (p[1]['core']['target_k'], p[1]['core']['chain'])):
        a, b = arms['core'], arms['core512']
        lines.append(f'| C{a["chain"]}, k={a["target_k"]} | {a["inherited_sequences"]} | '
                     f'{value(a["buses"])} | {value(b["buses"])} | '
                     f'{minutes(a["cg_minutes"])} | {minutes(b["cg_minutes"])} |')
    lines += ['', f'{sum(bool(r["fleet_proven"]) for r in mips.values())}/{len(mips)} available MIPs '
              'have a fleet proof within their own pools; all pass individual-route replay. '
              f'At k=15, {sum(r["target_k"] == 15 and r["buses"] is not None for r in rows)}/12 MIPs are published. This is a '
              'completion-selected subset; unfinished cases prevent an overall success-rate or timing claim. '
              'A CG certificate applies to the recorded discretized weighted model and tolerance. '
              'Fractional route weight is not a fleet-only lower bound.', '',
              'Baseline physics: covering, 240 kWh / 240 kW, charging-start fee 5, '
              'without shared charger capacity or a terminal-SOC floor. CG allows four hours; '
              'MIP allows three hours of fleet search within 3.5 hours total. Stage two constrains '
              'fleet to be no greater than the incumbent and minimizes charging-related cost.', '',
              '[All values, exact stopping reasons and source hashes](compact_seed_results.csv).', '',
              f'The nine longer searches on unchanged large-chain pools have '
              f'{len(source["campaigns"]["remaining_chain_gaps_20260914"]["mip"])} published endpoints. '
              'The fixed-state capacity calls are recorded as single-call diagnostics, never automatically '
              'as CG or MIP results. The two earlier wrapper failures and corrected attempts remain separate.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    queue = [line.split('|', 3) for line in source['squeue']['stdout'].splitlines()]
    counts = collections.Counter(r[1] for r in queue if not r[0].startswith('537227'))
    validation = dict(snapshot=str(args.snapshot), snapshot_sha256=hashlib.sha256(raw).hexdigest(),
                      source_timestamp=source['timestamp_utc'],
                      normalized_endpoints_verified=len(checks), checks=checks,
                      compact_cg=len(cgs), compact_mip=len(mips),
                      queue_counts=dict(counts), errors=[])
    dump(out / 'validation.json', validation)
    print(json.dumps({k: v for k, v in validation.items() if k != 'checks'}))


if __name__ == '__main__':
    main()
