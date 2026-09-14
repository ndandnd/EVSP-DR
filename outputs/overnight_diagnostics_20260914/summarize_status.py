"""Create a dated diagnostic comparison from verified collector publications."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    source = json.loads(raw)
    campaign = source['campaigns']['overnight_diagnostics_20260914']
    cases = campaign['workflow']['manifest.json']['cases']
    root = Path(__file__).resolve().parent / ('status_' + args.snapshot.stem)
    root.mkdir(exist_ok=True)
    snapshot_sha = hashlib.sha256(raw).hexdigest()
    endpoints = {}
    records = []
    for stage in ['cg', 'mip']:
        for result in campaign[stage]:
            assert result['completion_marker_matches'] is True
            cid = next(cid for cid in cases if cid in Path(result['path']).parts)
            case = cases[cid]
            endpoints[cid] = result
            final = result.get('final') or {}
            records.append(dict(case_id=cid, stage=stage, chain=case['chain'],
                target_k=case['target_k'], treatment=case['treatment'],
                cg_minutes=result.get('wall_s', 0)/60 if stage == 'cg' else None,
                cg_certified=result.get('certified_rc_optimal'),
                weighted_lp_objective=final.get('lp_obj'),
                fractional_route_count=final.get('route_weight'),
                integer_buses=result.get('buses'), pool_fleet_bound=result.get('fleet_bound'),
                fleet_proven_in_pool=result.get('fleet_proven'),
                individual_route_replay=result.get('physical_replay_validated'),
                source_path=result['path'], source_sha256=result['sha256'],
                input_sha256=case['input_sha256'], execution_commit=case['execution_commit'],
                snapshot_sha256=snapshot_sha))
    if records:
        write_csv(root/'endpoints.csv', records)

    reference = source['campaigns']['cumulative_budget_20260913']
    cg0 = next(r for r in reference['cg'] if r['case_id'] == 'c5_k05' and r['budget_arm'] == 'base')
    mip0 = next(r for r in reference['mip'] if r['case_id'] == 'c5_k05' and r['budget_arm'] == 'base')
    rows = []
    for label, cg, mip in [('Original: 30 by reduced cost', cg0, mip0),
            ('200 by reduced cost', endpoints.get('c5_k05_c200'), endpoints.get('c5_k05_c200_mip')),
            ('30 complementary columns', endpoints.get('c5_k05_complementary'), endpoints.get('c5_k05_complementary_mip'))]:
        if not cg or not mip:
            continue
        assert cg['certified_rc_optimal'] and mip['physical_replay_validated']
        rows.append(dict(treatment=label, target_k=5, cg_minutes=cg['wall_s']/60,
            cg_certified=True, weighted_lp_objective=cg['final']['lp_obj'],
            fractional_route_count=cg['final']['route_weight'],
            columns_in_mip_pool=mip['pool_columns'], integer_buses=mip['buses'],
            pool_fleet_bound=mip['fleet_bound'], fleet_proven_in_pool=mip['fleet_proven'],
            individual_route_replay=True, cg_source_path=cg['path'], cg_sha256=cg['sha256'],
            mip_source_path=mip['path'], mip_sha256=mip['sha256'],
            snapshot_path=str(args.snapshot), snapshot_sha256=snapshot_sha))
    write_csv(root/'chain5_k5_comparison.csv', rows)
    assert all(abs(r['weighted_lp_objective']-rows[0]['weighted_lp_objective']) < 1e-6 for r in rows)
    label = datetime.fromisoformat(reference['collected_utc']).astimezone(ZoneInfo('America/New_York')).strftime('%d %B, %H:%M %Z').lstrip('0')
    text = [f'# Overnight diagnostic results — {label}', '',
        '## First completed comparison: chain 5, target five buses', '',
        '| Column selection | CG minutes | Certified weighted LP objective | Columns in MIP pool | Integer buses | Fleet proved in pool? |',
        '|---|---:|---:|---:|---:|---|']
    text += [f"| {r['treatment']} | {r['cg_minutes']:.1f} | {r['weighted_lp_objective']:,.3f} | {r['columns_in_mip_pool']:,} | {r['integer_buses']} | {'yes' if r['fleet_proven_in_pool'] else 'no'} |" for r in rows]
    text += ['', 'All three runs reach the same weighted LP objective to numerical precision, with fractional route count five. Their integer pools differ. In these runs, the complementary selection supports five buses while the other two pools provably require six. These are different column sets, not nested pools: 45,105 columns need not contain the useful routes in the 12,029-column pool.', '',
        'This is one selected difficult case. It demonstrates that an LP pricing certificate and a large column count do not guarantee an integer target solution in the saved pool. It does not establish a general success rate or full-model integer optimality. Both new treatments took longer CG time than the original in this case; hardware variation limits direct timing attribution.', '',
        'The input, baseline physics, objective, execution code, cumulative CG allowance and final one-hour MIP allowance are held fixed for these treatment comparisons. The methods use covering, 240 kWh / 240 kW, flat prices and a charging-start fee of five. Shared charger capacity and a terminal-SOC floor are absent. Individual-route replay passes; other physical checks remain separate.', '',
        '[Exact results and source hashes](chain5_k5_comparison.csv).']
    # Report every observed comparison, including misses and worse incumbents.
    paired = []
    for base_id in sorted({cid.rsplit('_', 1)[0] for cid in cases
                           if cid.endswith(('_c200', '_complementary'))}):
        treatment_mips = {arm: endpoints.get(base_id+'_'+arm+'_mip')
                          for arm in ('c200', 'complementary')}
        if not any(treatment_mips.values()):
            continue
        control = next(r for r in reference['mip']
                       if r['case_id'] == base_id and r['budget_arm'] == 'base')
        control_cg = next(r for r in reference['cg']
                          if r['case_id'] == base_id and r['budget_arm'] == 'base')
        for arm, mip in [('original', control), *treatment_mips.items()]:
            cg = control_cg if arm == 'original' else endpoints.get(base_id+'_'+arm)
            case = cases[base_id+'_c200']
            paired.append(dict(case_id=base_id, chain=case['chain'], target_k=case['target_k'],
                treatment=arm, integer_buses=mip['buses'] if mip else None,
                pool_fleet_bound=mip['fleet_bound'] if mip else None,
                fleet_proven_in_pool=mip['fleet_proven'] if mip else None,
                individual_route_replay=mip.get('physical_replay_validated') if mip else None,
                mip_source_path=mip['path'] if mip else None,
                mip_sha256=mip.get('sha256') if mip else None,
                cg_minutes=cg['wall_s']/60 if cg else None,
                cg_certified=cg.get('certified_rc_optimal') if cg else None,
                weighted_lp_objective=(cg.get('final_lp') or {}).get('objective',
                    (cg.get('final') or {}).get('lp_obj')) if cg else None,
                cg_source_path=cg['path'] if cg else None,
                cg_sha256=cg.get('sha256') if cg else None,
                snapshot_sha256=snapshot_sha))
    if paired:
        write_csv(root/'column_selection_comparison.csv', paired)
        text += ['', '## All cases with a completed treatment MIP', '',
            '| Chain / target | Original buses | 200-column buses | Complementary buses |',
            '|---|---:|---:|---:|']
        for cid in sorted({r['case_id'] for r in paired}):
            group = [r for r in paired if r['case_id'] == cid]
            vals = [str(r['integer_buses']) if r['integer_buses'] is not None else 'pending'
                    for r in group]
            text.append(f"| C{group[0]['chain']} / {group[0]['target_k']} | {' | '.join(vals)} |")
        text += ['', 'Numbers are integer incumbents, not all optimal fleets. Pending means no published MIP result in this collection. See the CSV for each fleet bound, proof, timing and source hash. Results arriving early are not a random sample of the batch. A worse incumbent does not prove that the new pool lacks the earlier fleet.', '',
            '[Complete comparison with proof scopes](column_selection_comparison.csv).']
    continuation = endpoints.get('w4_k19_resume8h')
    extra_min = None
    if continuation:
        old = next(r for r in source['campaigns']['chain_extension_20260913']['cg'] if '/w4_k19/' in r['path'])
        extra_min = (continuation['wall_s']-old['wall_s'])/60
        drop = (old.get('final_lp') or {}).get('objective', old['final']['lp_obj'])-continuation['final_lp']['objective']
        text += ['', '## Chain 4 k=19 continuation', '',
            f"The separate continuation reached its pricing certificate after {continuation['wall_s']/60:.1f} cumulative CG minutes: {extra_min:.1f} beyond the original run. Its weighted LP objective improved by only {drop:.6f}. The original four-hour endpoint remains uncertified; this later certificate belongs to a distinct longer-budget treatment. Its final MIP is {'collected separately' if 'w4_k19_resume8h_mip' in endpoints else 'pending'}. This is a certificate at the stated reduced-cost tolerance within the tested graph."]
        mip = endpoints.get('w4_k19_resume8h_mip')
        if mip:
            start = mip.get('mip_start') or {}
            text += ['', f"The continuation MIP found {mip['buses']} buses with pool fleet bound {mip['fleet_bound']:.0f}; fleet proved: {mip['fleet_proven']}. The original pool's separate MIP found 19 and proved it. The new MIP used its ordinary {start.get('kind')} initializer with {start.get('validated_bus_count')} buses, accepted by Gurobi; it did not inherit the previous 19-bus integer incumbent. This new timed incumbent is not evidence that the continued pool cannot support 19. Membership of the earlier selected routes in the continued pool has not been audited. No full-model integer conclusion follows."]
    c5 = endpoints.get('w5_k19_resume8h')
    if c5:
        old = next(r for r in source['campaigns']['chain_extension_20260913']['cg'] if '/w5_k19/' in r['path'])
        extra = (c5['wall_s']-old['wall_s'])/60
        drop = (old.get('final_lp') or {}).get('objective', old['final']['lp_obj'])-c5['final_lp']['objective']
        text += ['', '## Chain 5 k=19 continuation', '',
            f"Certified after {c5['wall_s']/60:.1f} cumulative CG minutes, {extra:.1f} additional minutes. The weighted LP objective improved by {drop:.6f}. Its final MIP is {'collected separately' if 'w5_k19_resume8h_mip' in endpoints else 'pending'}. The original four-hour endpoint remains uncertified."]
    summary = dict(snapshot=str(args.snapshot), snapshot_sha256=snapshot_sha,
        diagnostic_cg=len(campaign['cg']), diagnostic_cg_certified=sum(bool(r.get('certified_rc_optimal')) for r in campaign['cg']),
        diagnostic_mip=len(campaign['mip']), comparison_rows=len(rows),
        c4_k19_additional_cg_minutes=extra_min,
        source_hashes_verified_by_collector=True, full_model_integer_proof_claimed=False)
    text += ['', f"Collected diagnostic endpoints: {summary['diagnostic_cg']} CG runs ({summary['diagnostic_cg_certified']} certified) and {summary['diagnostic_mip']} MIPs. Other cases remain pending or running; missing results are not failures.", '', '[All collected endpoints](endpoints.csv).']
    (root/'README.md').write_text('\n'.join(text)+'\n')
    (root/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
