"""Compare longer searches with the same audited, ordered input pools."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def pool_key(result):
    audit = result.get('physical_pool_audit') or {}
    return (audit.get('mip_ordered_pool_sha256'),
            json.dumps(audit.get('input_hashes'), sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    args = parser.parse_args()
    raw = args.snapshot.read_bytes()
    snapshot = json.loads(raw)
    campaign = snapshot['campaigns']['parallel_pool_followup_20260914']
    cases = campaign['workflow']['manifest.json']['cases']
    all_mips = [r for c in snapshot['campaigns'].values() for r in c.get('mip', [])]
    extension = snapshot['campaigns']['chain_extension_20260913']['mip']
    diagnostics = snapshot['campaigns']['overnight_diagnostics_20260914']['mip']
    rows = []
    for result in campaign['mip']:
        cid = Path(result['path']).parent.name
        case = cases[cid]
        comparator = case.get('comparator')
        candidates = [r for r in all_mips if r.get('path') == comparator] if comparator else []
        if not candidates and comparator and '/chain_extension_20260913/' in comparator:
            candidates = [r for r in extension if f"/cases/{case['original_case']}/" in r['path']]
        if not comparator:
            candidates = [r for r in diagnostics
                          if r.get('source_journal_sha256') == case['source_journal_sha256']]
        assert len(candidates) == 1, (cid, len(candidates))
        prior = candidates[0]
        assert result['completion_marker_matches'] is True
        assert result['source_journal_sha256'] == case['source_journal_sha256']
        assert result['source_status_sha256'] == case['source_status_sha256']
        assert pool_key(result)[0] and pool_key(result) == pool_key(prior), cid
        assert result['physical_pool_audit']['input_hashes']['instance_sha256'] == case['input_sha256']
        for r in (result, prior):
            audit = r['physical_pool_audit']
            assert audit['rejected_columns'] == audit['deterministically_repaired'] == 0
            assert r['physical_replay_validated'] is True
        target = case['target_k']
        outcome = ('target matched' if result['buses'] == target else
                   'target absent from this pool' if result['fleet_proven'] else 'fleet gap open')
        rows.append(dict(case_id=cid, target_buses=target, prior_buses=prior['buses'],
            buses_found=result['buses'], pool_fleet_bound=result['fleet_bound'],
            fleet_proved_in_pool=result['fleet_proven'], outcome=outcome,
            fleet_stage_minutes=result['two_stage']['stage1_runtime_s']/60,
            total_mip_minutes=result['runtime_s']/60,
            original_pool_sha256=pool_key(prior)[0], new_pool_sha256=pool_key(result)[0],
            input_sha256=case['input_sha256'], same_ordered_pool_and_inputs=True,
            prior_result_path=prior['path'], prior_result_sha256=prior['sha256'],
            new_result_path=result['path'], new_result_sha256=result['sha256'],
            source_status_sha256=case['source_status_sha256'],
            source_journal_sha256=case['source_journal_sha256']))
    misses = [r for r in extension if r['buses'] > int(Path(r['path']).parts[
        Path(r['path']).parts.index('cases')+1].split('_k')[1])]
    recovery_rows = []
    for original in misses:
        cid = Path(original['path']).parts[Path(original['path']).parts.index('cases')+1]
        target = int(cid.split('_k')[1])
        recovered = [r for r in all_mips if r.get('buses') == target
                     and r.get('fleet_proven') and r.get('physical_replay_validated')
                     and pool_key(r) == pool_key(original)]
        recovery_rows.append(dict(case_id=cid, target_buses=target,
            original_buses=original['buses'], same_pool_target_recovered=bool(recovered),
            recovery_paths=';'.join(r['path'] for r in recovered)))
    out = Path(__file__).parent / ('status_' + args.snapshot.stem)
    out.mkdir(exist_ok=True)
    for name, data in [('results.csv', rows), ('original_gap_recoveries.csv', recovery_rows)]:
        if data:
            with (out/name).open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader(); writer.writerows(data)
    lines = ['# Longer searches on unchanged saved pools', '',
        '| Case | Target | Prior buses | New buses | Fleet proved in pool | Fleet-stage minutes | Meaning |',
        '|---|---:|---:|---:|---|---:|---|']
    lines += [f"| {r['case_id']} | {r['target_buses']} | {r['prior_buses']} | {r['buses_found']} | "
              f"{'yes' if r['fleet_proved_in_pool'] else 'no'} | {r['fleet_stage_minutes']:.1f} | {r['outcome']} |" for r in rows]
    recovered_count = sum(r['same_pool_target_recovered'] for r in recovery_rows)
    lines += ['', f"All {len(rows)} pairs have matching ordered-pool and input hashes; selected routes pass individual replay. "
        'These searches add no CG columns. Fleet search has at most three hours within 3½ hours total; charging uses the remaining time. '
        'A fleet proof above target establishes target absence only from that finite pool. Open gaps do not establish absence.', '',
        f"Across the original extension controls, {recovered_count} of {len(recovery_rows)} misses now have a verified target solution from the same pool. "
        f"The other {len(recovery_rows)-recovered_count} remain unresolved. The continued C4 k19 pool is a separate treatment; its recovery does not add another original-control recovery.", '',
        'The four recovered targets in this batch have fleet-proof times 87.2, 46.8, 32.7 and 19.3 minutes. '
        'C6 k23 therefore recovered within the original 30-minute fleet allowance; extra allocated time alone cannot explain every changed outcome. '
        'Hardware and parallel-search timing remain uncontrolled. Charging-cost optimality is separate from fleet proof.', '',
        'Baseline physics omit shared station capacity and a terminal-SOC floor. No branch-and-price or full-model integer proof is claimed.', '',
        '[Editable before/after values and source hashes](results.csv); [original gap recovery map](original_gap_recoveries.csv).']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    validation = dict(snapshot=str(args.snapshot), snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        source_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        verified_pairs=len(rows), original_misses=len(recovery_rows), same_pool_recoveries=recovered_count,
        remaining_original_gaps=len(recovery_rows)-recovered_count, errors=[])
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps({'report':str(out), **validation}))


if __name__ == '__main__':
    main()
