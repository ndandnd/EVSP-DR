"""Publish a dated, reproducible view of the two active comparison campaigns."""
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
    source = args.snapshot
    raw = source.read_bytes()
    snapshot = json.loads(raw)
    c = snapshot['campaigns']['cumulative_budget_20260913']
    e = snapshot['campaigns']['chain_extension_20260913']
    assert c['schema'] == 'evsp-cumulative-budget-collection-v1'
    root = Path(__file__).resolve().parent / ('status_' + source.stem)
    root.mkdir(exist_ok=True)
    digest = hashlib.sha256(raw).hexdigest()
    timestamp = datetime.fromisoformat(c['collected_utc']).astimezone(ZoneInfo('America/New_York'))
    label = timestamp.strftime('%d %B, %H:%M %Z').lstrip('0')
    cg = {r['case_id']: r for r in c['cg'] if r['budget_arm'] == 'base'}
    mip = {(r['case_id'], r['budget_arm']): r for r in c['mip']}
    assert len(mip) == len(c['mip']), 'duplicate scientific MIP endpoints'
    warm = [r for r in c['mip'] if r['budget_arm'] == 'warm']
    assert len(warm) == 24 and all(r['buses'] == r['target_k'] and
        r['fleet_proven'] and r['physical_replay_validated'] for r in warm)
    rows = []
    for case in c['records']:
        cid, k = case['case_id'], case['target_k']
        x, f, w = cg.get(cid, {}), mip.get((cid, 'base'), {}), mip.get((cid, 'warm'), {})
        if not f:
            meaning = 'MIP pending' if x.get('usable') else 'CG stopped without usable pool' if x else 'CG running or waiting'
        elif not f.get('physical_replay_validated'):
            meaning = 'physical validation incomplete'
        elif f['buses'] == k:
            meaning = 'target matched'
        elif f['buses'] < k:
            meaning = 'fewer buses than target'
        elif f.get('fleet_proven'):
            meaning = 'proved pool limit above target'
        else:
            meaning = 'fleet gap open'
        rows.append(dict(case_id=cid, chain=case['chain'], target_k=k,
            fresh_cg_allowance_min=case['budgets']['fresh_primary_budget_s']/60,
            fresh_cg_minutes=x.get('wall_s', 0)/60 if x else None,
            fresh_cg_certified=x.get('certified_rc_optimal'), fresh_cg_stop=x.get('stop_reason'),
            fresh_weighted_lp_objective=x.get('final', {}).get('lp_obj'),
            fresh_fractional_route_weight=x.get('final', {}).get('route_weight'),
            fresh_buses=f.get('buses'), fresh_pool_fleet_bound=f.get('fleet_bound'),
            fresh_fleet_proven=f.get('fleet_proven'), fresh_mip_status=f.get('status_name'),
            fresh_meaning=meaning, warm_buses=w.get('buses'), warm_fleet_proven=w.get('fleet_proven'),
            fresh_cg_source_path=x.get('path'), fresh_cg_source_sha256=x.get('sha256'),
            fresh_mip_source_path=f.get('path'), fresh_mip_source_sha256=f.get('sha256'),
            warm_mip_source_path=w.get('path'), warm_mip_source_sha256=w.get('sha256'),
            snapshot_path=str(source), snapshot_sha256=digest))
    assert len(rows) == 24
    write_csv(root/'comparison.csv', rows)

    # The matched k15 warm-reference MIPs certify the preceding completed baseline.
    reach = {}
    for chain in range(1, 7):
        r = mip[(f'c{chain}_k15', 'warm')]
        assert r['buses'] == 15 and r['fleet_proven'] and r['physical_replay_validated']
        reach[chain] = dict(chain=chain, target_k=15, buses=15,
            cg_minutes_at_this_k=r['source_cg_wall_s']/60, source_path=r['path'],
            source_sha256=r['sha256'], fleet_proven_in_pool=True,
            physical_route_replay=True, source_campaign='matched warm k15 reference')
    manifest = e['workflow']['manifest.json']['cases']
    for r in e['mip']:
        parts = Path(r['path']).parts
        matches = [cid for cid in manifest if cid in parts]
        assert len(matches) == 1, r['path']
        cid = matches[0]
        chain = int(cid.split('_')[0][1:])
        k = int(cid.split('_')[1][1:])
        if r['buses'] == k and r.get('fleet_proven') and r.get('physical_replay_validated') and k > reach[chain]['target_k']:
            reach[chain] = dict(chain=chain, target_k=k, buses=r['buses'],
                cg_minutes_at_this_k=r['source_cg_wall_s']/60, source_path=r['path'],
                source_sha256=r.get('sha256'), fleet_proven_in_pool=True,
                physical_route_replay=True, source_campaign='chain extension')
    write_csv(root/'chain_reach.csv', list(reach.values()))
    counts = {name: sum(r['fresh_meaning'] == name for r in rows) for name in
        ['target matched', 'proved pool limit above target', 'fleet gap open']}
    fresh_done = sum(r['budget_arm'] == 'base' for r in c['mip'])
    cert = sum(bool(r.get('certified_rc_optimal')) for r in cg.values())
    summary = [f'# Research status — {label}', '',
        f"Largest verified integer match: **{max(r['target_k'] for r in reach.values())} buses**. These are baseline covering runs with inherited columns, 240 kWh batteries, 240 kW charging and a fee of 5 per charging start. Shared station capacity and a terminal-SOC floor are absent. Fleet proofs apply to the saved pools; physical replay checks individual routes. Charging optimality is a separate question.", '',
        '| Chain | Largest target matched | Integer buses | CG minutes at this k |',
        '|---|---:|---:|---:|']
    summary += [f"| {r['chain']} | {r['target_k']} | {r['buses']} | {r['cg_minutes_at_this_k']:.1f} |" for r in reach.values()]
    summary += ['', 'CG minutes include this k’s route import and CG. Earlier k values, original graph construction and MIP are separate. The source of each row is in [chain_reach.csv](chain_reach.csv). A CG certificate at a larger k is not an integer result.', '',
        '## Fresh runs given the accumulated warm-chain time', '',
        f"{cert}/24 fresh CG runs with the primary allowance have pricing certificates. {fresh_done} corresponding fresh MIPs have finished: **{counts['target matched']} target matches, {counts['proved pool limit above target']} proved pool limits above target, and {counts['fleet gap open']} unresolved fleet gaps**. All 24 matched warm-reference MIPs reach their targets with finite-pool fleet proofs. The register retains any distinct larger-allowance continuations separately; this table never mixes them with primary results.", '',
        '| Case | Fresh CG min | Fresh buses | Fresh pool fleet bound | Fleet proved? | Warm buses | Meaning |',
        '|---|---:|---:|---:|---|---:|---|']
    for r in rows:
        values = [f"C{r['chain']}, k={r['target_k']}",
            f"{r['fresh_cg_minutes']:.1f}" if r['fresh_cg_minutes'] is not None else 'pending',
            str(r['fresh_buses']) if r['fresh_buses'] is not None else 'pending',
            f"{r['fresh_pool_fleet_bound']:.0f}" if r['fresh_pool_fleet_bound'] is not None else 'pending',
            'yes' if r['fresh_fleet_proven'] else 'no' if r['fresh_fleet_proven'] is False else 'pending',
            str(r['warm_buses']), r['fresh_meaning']]
        summary.append('| '+' | '.join(values)+' |')
    summary += ['', '**How to interpret this:** a proved pool limit means longer MIP search on the unchanged pool cannot meet the target. An unresolved gap does not prove that the target is absent from the pool. Overall MIP TIME_LIMIT can occur after fleet optimality was proved, during charging optimization. The certificate, fleet proof, target attainment and physical checks remain separate.', '',
        'Pricing certificates concern the tested graph and reduced-cost tolerance. A weighted LP objective and total fractional route weight are different quantities; the latter is not automatically a fleet-only lower bound. Historical code revisions and hardware varied, so the time comparison is retrospective. Shared endpoints for the larger allowance are aliases, not independent searches.', '',
        f"[Exact comparison, budgets and source hashes](comparison.csv). Collector snapshot: `{source}`; SHA-256 `{digest}`. The original snapshot and remote artifacts remain authoritative."]
    (root/'README.md').write_text('\n'.join(summary)+'\n')
    validation = dict(snapshot=str(source), snapshot_sha256=digest,
        collected_utc=c['collected_utc'], cumulative_cg=len(c['cg']),
        cumulative_mip=len(c['mip']), fresh_mip=fresh_done, fresh_outcomes=counts,
        extension_cg=len(e['cg']), extension_mip=len(e['mip']), cases=len(rows),
        errors=c['errors'], source_builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    print(json.dumps({'report': str(root), **validation}))


if __name__ == '__main__':
    main()
