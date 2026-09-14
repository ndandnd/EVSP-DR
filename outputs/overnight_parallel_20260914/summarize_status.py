"""Report verified seed experiments and existing-pool unions from a snapshot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def case_id(row):
    return Path(row['path']).parent.name


def write_csv(path, rows):
    if rows:
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    out = Path(__file__).parent / ('status_' + args.snapshot.stem)
    out.mkdir(exist_ok=True)
    all_mips = [r for c in snapshot['campaigns'].values() for r in c.get('mip', [])]
    union = snapshot['campaigns']['parallel_pool_unions_20260914']
    union_cases = union['workflow']['manifest.json']['cases']
    constructions = union['workflow']['pool_constructions']
    union_rows = []
    for result in union['mip']:
        cid = case_id(result)
        case = union_cases[cid]
        construction = constructions[case['source_case']]
        assert result['completion_marker_matches'] is True
        assert result['source_status_sha256'] == construction['result_sha256']
        assert result['source_journal_sha256'] == construction['journal_sha256']
        assert construction['optimization_run'] is False
        assert construction['full_model_lp_certified'] is False
        audit = result['physical_pool_audit']
        assert audit['input_hashes']['instance_sha256'] == case['input_sha256']
        assert audit['rejected_columns'] == audit['deterministically_repaired'] == 0
        assert result['physical_replay_validated'] is True
        source_best = {}
        for source in construction['construction_summary']['sources']:
            matches = [r for r in all_mips
                       if (r.get('physical_pool_audit') or {}).get('mip_ordered_pool_sha256')
                       == source['source_mip_ordered_pool_sha256']
                       and (r.get('physical_pool_audit') or {}).get('input_hashes', {}).get('instance_sha256')
                       == case['input_sha256'] and r.get('partitioning') is False]
            assert matches, (cid, source['arm'])
            best = min(r['buses'] for r in matches)
            source_best[source['arm']] = dict(buses=best,
                proved=any(r['buses'] == best and r['fleet_proven'] for r in matches),
                evidence=[{'path':r['path'], 'sha256':r['sha256'],
                    'runtime_s':r.get('runtime_s'),
                    'stage1_runtime_s':(r.get('two_stage') or {}).get('stage1_runtime_s'),
                    'stage1_time_limit_s':(r.get('two_stage') or {}).get('stage1_time_limit_s'),
                    'stage2_available_time_s':(r.get('two_stage') or {}).get('stage2_available_time_s')}
                    for r in matches])
        union_rows.append(dict(case_id=cid, target_buses=case['target_k'],
            best_original_pool_buses=source_best['original']['buses'],
            best_c200_pool_buses=source_best['c200']['buses'],
            best_complementary_pool_buses=source_best['complementary']['buses'],
            union_buses=result['buses'], union_fleet_bound=result['fleet_bound'],
            fleet_proved_in_union=result['fleet_proven'],
            fleet_stage_minutes=result['two_stage']['stage1_runtime_s']/60,
            total_mip_minutes=result['runtime_s']/60, pool_columns=result['pool_columns'],
            physical_replay=True, input_sha256=case['input_sha256'],
            result_path=result['path'], result_sha256=result['sha256'],
            union_status_sha256=result['source_status_sha256'],
            union_journal_sha256=result['source_journal_sha256'],
            source_pool_evidence=json.dumps(source_best, sort_keys=True)))

    seeds = snapshot['campaigns']['overnight_parallel_20260914']
    seed_cases = seeds['workflow']['manifest.json']['cases']
    cg_by_case = {case_id(r):r for r in seeds['cg']}
    mip_by_case = {case_id(r):r for r in seeds['mip']}
    seed_rows = []
    for cid, case in seed_cases.items():
        if case['kind'] != 'cg':
            continue
        cg = cg_by_case.get(cid, {})
        mip = mip_by_case.get(cid+'_mip', {})
        audit = cg.get('inherited_event_pool_audit') or {}
        if cg:
            assert cg['completion_marker_matches'] is True
            assert audit['source_status_sha256'] == case['seed_status_sha256']
            assert audit['source_journal_sha256'] == case['seed_journal_sha256']
            assert audit['selected_for_replay'] == case['seed']['selected_sequence_count']
            assert not any(audit[k] for k in ('inherited_duals','inherited_basis','inherited_lp_certificate'))
        if mip:
            assert cg and mip['completion_marker_matches'] is True
            assert mip['source_status_sha256'] == cg['sha256']
            assert mip['source_journal_sha256'] == cg['columns_journal_sha256']
            assert mip['physical_replay_validated'] is True
        seed_rows.append(dict(case_id=cid, pair_id=case['pair_id'], target_buses=case['target_k'],
            method=case['treatment'], selected_parent_sequences=case['seed']['selected_sequence_count'],
            completed_replays=audit.get('replay_completed'), accepted_sequences=audit.get('accepted_columns'),
            rejected_sequences=audit.get('rejected_columns'), added_columns=audit.get('added_to_child_pool'),
            import_seconds=audit.get('import_runtime_s'), cg_minutes=cg.get('wall_s',0)/60 if cg else None,
            cg_certified=cg.get('certified_rc_optimal'), cg_stop=cg.get('stop_reason'),
            cg_weighted_objective=(cg.get('final_lp') or {}).get('objective'),
            fractional_route_count=(cg.get('final_lp') or {}).get('route_weight'),
            mip_buses=mip.get('buses'), pool_fleet_bound=mip.get('fleet_bound'),
            fleet_proved_in_pool=mip.get('fleet_proven'),
            charging_proved_in_pool=(mip.get('two_stage') or {}).get('stage2_status_name') == 'OPTIMAL' if mip else None,
            mip_minutes=mip.get('runtime_s',0)/60 if mip else None,
            cg_path=cg.get('path'), cg_sha256=cg.get('sha256'),
            mip_path=mip.get('path'), mip_sha256=mip.get('sha256'),
            input_sha256=case['input_sha256'], seed_status_sha256=case['seed_status_sha256'],
            seed_journal_sha256=case['seed_journal_sha256']))

    # Study paths use /home symlinks, while publications resolve under /share.
    published = {(case_id(r), r['sha256']) for r in seeds['mip'] + union['mip']}
    late = [r for r in snapshot.get('mip_preemption_study', {}).get('attempts', [])
            if r.get('result_exists') and any('/'+root+'/' in (r.get('result_path') or '')
                for root in ('overnight_parallel_20260914','parallel_pool_unions_20260914'))
            and (r.get('case_id'), r.get('result_sha256')) not in published]
    write_csv(out/'union_results.csv', union_rows)
    write_csv(out/'seed_results.csv', seed_rows)
    lines = ['# Combined pools and small previous-k warm starts', '',
        f"Verified source collection: {snapshot['timestamp_utc']}.", '',
        '## Combining three saved pools', '',
        '| Case | Target | Best original / 200-column / complementary fleets | Union buses | Union fleet bound | Fleet proved in union |',
        '|---|---:|---|---:|---:|---|']
    for r in union_rows:
        lines.append(f"| {r['case_id']} | {r['target_buses']} | {r['best_original_pool_buses']} / {r['best_c200_pool_buses']} / {r['best_complementary_pool_buses']} | {r['union_buses']} | {r['union_fleet_bound']:.6g} | {'yes' if r['fleet_proved_in_union'] else 'no'} |")
    lines += ['', 'Every union preserves its three frozen source pools and adds no new pricing run. Result, input and construction hashes are verified; selected routes pass individual replay. Fleet proofs apply only to the combined pool. Open bounds do not prove the target absent. Because some individual source searches still have open gaps, a union recovery alone does not prove that combining columns was necessary.', '',
        'All three sources and the union retain their actual search budgets and timing in the CSV. These baseline experiments omit shared charger capacity and a terminal-SOC floor. No new full-model pricing certificate comes from constructing a union.', '',
        '## Small warm starts with completed CG', '',
        '| Case | Previous sequences selected / added | Import seconds | CG minutes | CG certified | Integer buses | Fleet proved in pool |',
        '|---|---|---:|---:|---|---:|---|']
    for r in seed_rows:
        if r['cg_path']:
            lines.append(f"| {r['case_id']} | {r['selected_parent_sequences']} / {r['added_columns']} | {r['import_seconds']:.2f} | {r['cg_minutes']:.1f} | {'yes' if r['cg_certified'] else 'no'} | {r['mip_buses'] if r['mip_buses'] is not None else 'pending'} | {'yes' if r['fleet_proved_in_pool'] else 'pending' if r['fleet_proved_in_pool'] is None else 'no'} |")
    lines += ['', 'These are paired treatments on 18 selected inputs, not 36 independent datasets. Compare previous integer-selected sequences with an equal number chosen by LP weight. Selected sequence counts are matched; coverage and actual child additions can differ. Prior CG and MIP costs remain in the manifest and must be included in end-to-end comparisons. Historical fresh/full-pool runs are context, not automatically matched timing controls.', '',
        'Only completed publications enter this table; running attempts and late scheduler-only outputs remain separate. A CG certificate does not replace the integer search. No overall winner is inferred from the first completed pairs.', '',
        '[All seed cases, values and hashes](seed_results.csv); [union results and source-pool evidence](union_results.csv).']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    (out/'late_scheduler_results.json').write_text(json.dumps(late,indent=2)+'\n')
    validation = dict(snapshot=str(args.snapshot), snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        union_mips_verified=len(union_rows), seed_cgs_verified=len(cg_by_case),
        seed_mips_verified=len(mip_by_case), late_scheduler_results=len(late), errors=[])
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(json.dumps(validation))


if __name__ == '__main__':
    main()
