"""Read-only incidence and reduced-cost audit of the known C1 k15 pool gap."""
import hashlib
import json
from pathlib import Path

FULL = Path('/home/nc437/ladder-lite/full_pool_recovery_20260912/cases/w1_k15/mip/41592_r0/result.json')
FULL_HASH = '09872dfb4720d683e77b1110a6baad67047af9abe2255394380cdd3f0dee0a7d'
COMPACT = {
    'core': 'dc2620dab854cf18f4a62f1b67e92fc58cd1ce466a1c0ce2bd9d86012d1b7545',
    'core512': '42c2a38c4deb00a55cc86ab935199df3c90885cf678fe8a5bcc5925d8a6913e9',
}


def read_bound(path, expected):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected, str(path)
    return json.loads(raw)


def main():
    full = read_bound(FULL, FULL_HASH)
    assert full['buses'] == 15 and full['physical_replay_validated']
    assert full['partitioning'] is False
    donors = full['selected_routes']
    assert len(donors) == 15
    keys = [tuple(sorted(route['trips'])) for route in donors]
    assert all(len(key) == len(set(key)) for key in keys)
    results = []
    for treatment, digest in COMPACT.items():
        path = Path('/home/nc437/ladder-lite/compact_seed_support_20260914/cases') / ('c1_k15_' + treatment + '_mip') / 'mip_result.json'
        mip = read_bound(path, digest)
        assert mip['buses'] == 16 and mip['fleet_proven'] and mip['fleet_bound'] == 16
        assert mip['partitioning'] is False and mip['physics'] == full['physics']
        audit = mip['physical_pool_audit']
        assert audit['input_hashes'] == full['physical_pool_audit']['input_hashes']
        assert audit['rejected_columns'] == audit['deterministically_repaired'] == audit['added_giro_route_count'] == 0
        cg = read_bound(mip['source_result'], mip['source_result_sha256'])
        assert cg['certified_rc_optimal']
        duals = {int(k): v for k, v in cg['final_lp']['trip_duals'].items()}
        counts = {key: 0 for key in keys}
        journal_hash = hashlib.sha256()
        journal_rows = 0
        with Path(mip['source_journal']).open('rb') as stream:
            for line in stream:
                journal_hash.update(line)
                if not line.strip():
                    continue
                column = json.loads(line)
                journal_rows += 1
                key = tuple(sorted(column['trips']))
                if key in counts:
                    counts[key] += 1
        assert journal_hash.hexdigest() == mip['source_journal_sha256']
        rows = []
        for index, (route, key) in enumerate(zip(donors, keys), start=1):
            cost = route['expanded_grid_cost']
            assert cost == route['cost']
            dual_sum = sum(duals[t] for t in key)
            rows.append(dict(donor_route=index, trips=list(key), trip_count=len(key),
                             matching_coverage_columns=counts[key],
                             weighted_route_cost=cost, trip_dual_sum=dual_sum,
                             reduced_cost=cost-dual_sum))
        missing = sum(row['matching_coverage_columns'] == 0 for row in rows)
        assert missing > 0, 'All 15 donor incidences present: investigate apparent pool-proof contradiction.'
        results.append(dict(treatment=treatment, compact_mip_path=str(path),
                            compact_mip_sha256=digest, compact_cg_path=mip['source_result'],
                            compact_cg_sha256=mip['source_result_sha256'],
                            compact_journal_path=mip['source_journal'],
                            compact_journal_sha256=journal_hash.hexdigest(),
                            journal_rows=journal_rows, mip_columns=mip['pool_columns'],
                            donor_routes_missing=missing, donor_routes=rows))
    print(json.dumps(dict(full_mip_path=str(FULL), full_mip_sha256=FULL_HASH,
                          input_hashes=full['physical_pool_audit']['input_hashes'],
                          scope='Trip-incidence membership and weighted reduced costs at each compact final dual. Individual route replay is inherited from recorded audits. No shared capacity, no new optimization, no duplicate-removal validation.',
                          results=results), indent=2))


if __name__ == '__main__':
    main()
