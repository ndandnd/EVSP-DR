"""Read-only consistency checks; archive supplied reviewer reports without editing them."""
from pathlib import Path
import csv, hashlib, json, shutil
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def rows(path):
    with path.open() as f:
        return {r['case_id']: r for r in csv.DictReader(f)}

def main():
    supplied = ['advisor_time_only_audit_20260917/README.md', 'advisor_seg_lp_20260916/README.md']
    receipts = []
    for rel in supplied:
        source = ROOT / rel
        dest = OUT / 'reviewer_source' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        assert digest(source) == digest(dest)
        receipts.append({'source': rel, 'archive': str(dest.relative_to(ROOT)), 'sha256': digest(source)})
    bounds_path = ROOT / 'time_only_vsp_20260916/per_case.csv'
    chain_path = ROOT / 'execution/audited_chain_results.csv'
    bounds, chain = rows(bounds_path), rows(chain_path)
    assert len(bounds) == len(chain) == 102 and bounds.keys() == chain.keys()
    assert all(r['input_sha256'] == chain[c]['input_sha256'] for c, r in bounds.items())
    assert all(int(r['segregated_closure_lp_fleet_lower_bound']) == int(r['target_buses']) for r in bounds.values())
    for group in ['18E1', '18E2']:
        assert all(r[group + '_closure_relaxation_minimum'] == r[group + '_giro_duties'] for r in bounds.values())
    deficits = {c: int(r['target_buses']) - int(r['mixed_closure_relaxation_minimum']) for c, r in bounds.items()}
    assert list(deficits.values()).count(0) == 93 and list(deficits.values()).count(1) == 9
    assert all(abs(float(r['fractional_route_weight']) - int(r['mixed_closure_relaxation_minimum'])) < 1e-6 for r in bounds.values())
    result = {
        'status': 'PASS', 'sources': receipts,
        'data_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [bounds_path, chain_path]},
        'input_hash_matches': 102, 'separated_bound_equals_k': 102,
        'mixed_bound_equals_k': 93, 'mixed_bound_equals_k_minus_one': 9,
        'external_recomputation_scope': 'Reviewer reports independent matching recomputation on 12 instances / 36 group cells; code and transcript not supplied as files.',
        'scope_of_this_check': 'CSV identities/counts and verbatim source preservation. No optimizer, certificate recomputation, physical replay or cluster job.',
        'numeric_headlines_changed': False,
    }
    (OUT / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['sources', 'data_sha256']}))

if __name__ == '__main__':
    main()
