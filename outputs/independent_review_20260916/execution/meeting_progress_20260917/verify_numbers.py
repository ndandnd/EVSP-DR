from pathlib import Path
import csv, json, hashlib
R=Path(__file__).resolve().parents[4]
B=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):
    with p.open() as f: return list(csv.DictReader(f))

def main():
    snap=R/'outputs/independent_review_20260916/execution/monitor/20260917T034416Z/snapshot.json'
    original=R/'outputs/independent_review_20260916/execution/audited_chain_results.csv'
    longer=R/'outputs/overnight_next_20260914/status_20260916T194843Z/longer_gap_results.csv'
    fresh=R/'outputs/cumulative_budget_20260913/status_20260916T194843Z/comparison.csv'
    d=json.loads(snap.read_text())['campaigns']['p1']['data']
    checks={r['case_id']:r for r in d['endpoint_audit']['checks']}
    done=[r for r in d['rows'] if r['status'] in ('finished','reused_finished')]
    assert len(done)==30
    for row in done:
        a=checks[row['case_id']]
        assert a['status']=='verified_control_identity' and all(a['tests'].values())
        assert a['result_sha256']==row['result_sha256'] and a['buses']==row['buses']
    trials=[r for r in done if r['item']==7]
    assert len(trials)==18 and not any(r['target_matched'] for r in trials)
    assert [min(r['buses'] for r in trials if r['chain']==c) for c in range(1,7)]==[18,17,17,18,16,18]
    orig=csvrows(original)
    maxima=[max(int(r['target_buses']) for r in orig if int(r['chain'])==c and int(r['integer_buses'])==int(r['target_buses'])) for c in range(1,7)]
    assert maxima==[26,28,31,29,26,28]
    older=csvrows(longer)
    at32={c:[] for c in range(1,7)}
    for row in older:
        if int(row['target'])==32 and int(row['buses'])==32:
            assert row['individual_route_replay']=='True' and row['unchanged_pool_verified']=='True'
            at32[int(row['chain'])].append({'source':'older longer search','case_id':row['case_id'],'sha256':row['mip_sha256']})
    for row in done:
        if row['item']==9 and row['buses']==32:
            at32[row['chain']].append({'source':'P1 seed repeat','case_id':row['case_id'],'sha256':row['result_sha256']})
    assert all(at32.values())
    pair=csvrows(fresh); assert len(pair)==24
    counts={k:sum(int(r['fresh_buses'])==k for r in pair if int(r['target_k'])==k) for k in [5,8,10,15]}
    assert list(counts.values())==[5,1,0,0]
    assert all(int(r['warm_buses'])==int(r['target_k']) for r in pair)
    assert all(r['fresh_cg_certified']=='True' for r in pair)
    audit=[json.loads(line) for line in (B/'constrained_k5_source_check.jsonl').read_text().splitlines()]
    assert len(audit)==4 and all(x['sha_verified'] and x['audit']['physical_routes_validated'] for x in audit)
    assert all(x['audit']['buses']==5 and x['audit']['reserve_kwh']==36 and x['audit']['minimum_active_minutes']==3 for x in audit)
    result={'status':'PASS','snapshot_utc':'2026-09-17T03:44:16Z','source_sha256':{str(p.relative_to(R)):sha(p) for p in [snap,original,longer,fresh]},'original_maxima':maxima,'matched_32_evidence':at32,'fresh_original_matches':counts,'fresh_k15_repeats':{'hits':0,'trials':18,'best_by_chain':[18,17,17,18,16,18]},'source_and_control_checks_verified':30,'code_review_sha256':sha(R/'outputs/independent_review_20260916/CODE_REVIEW.md'),'limits':'No new solver calls, exact arithmetic pricing, or duplicate-trip conversion audit for the new seed outcomes.'}
    (B/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (B/'p1_selected_results.json').write_text(json.dumps(done,indent=2)+'\n')
    print('Verified original maxima, 6/6 best k32 hits, 0/18 fresh k15 repeats, and original fresh counts.')
if __name__=='__main__': main()
