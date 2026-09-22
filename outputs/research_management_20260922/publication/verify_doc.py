"""Verify exported document claims against immutable native result receipts."""
from pathlib import Path
import csv, hashlib, json, re

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
b = (P/'current_before.md').read_text()
a = (P/'current_after.md').read_text()
d = (P/'route_columns_tab.md').read_text()
checks = {
    'current_before_is_previous_verified_export': b == (ROOT/'outputs/research_followup_20260921/presentation_update_20260922/current_after.md').read_text(),
    'three_targeted_lines_changed': sum(x != y for x, y in zip(b.splitlines(), a.splitlines())) == 3 and len(b.splitlines()) == len(a.splitlines()),
    'current_images_preserved': re.findall(r'!\[[^\]]*\]\[[^\]]*\]', b) == re.findall(r'!\[[^\]]*\]\[[^\]]*\]', a),
    'existing_matrix_tab_unchanged': (P/'matrix_tab_before.md').read_bytes() == (P/'matrix_tab_after.md').read_bytes(),
    'matrix_original_dimensions': '| Minute capacity | 1,789 | 321 | 0 | 44,844 |' in d,
    'matrix_merged_dimensions': '| Merge repeated rows | 303 | 321 | 0 | 10,232 |' in d,
    'matrix_endpoint_dimensions': '| Start/end equations | 307 | 321 | 272 | 7,714 |' in d,
    'capacity_equality_scope': 'within this saved pool' in d,
    'start_diagnostic_scope': 'excludes its earlier acquisition cost' in d,
    'compression_not_new_physics': 'not new physical validation or faster pricing' in d,
    'not_general_speedup': 'does not establish a general speedup' in d,
    'existing_sparsity_tab_linked': 't.lt33xg84cn65' in d,
    'new_tab_linked_from_front': 't.ikbgt85cdszz' in a,
    'toy_overlap_shown': '| S, 08:10–08:20 | 1 | 1 | ≤ 1 |' in d,
    'capacity_diff_equation': 'uₑ − uₑ₋₁' in d,
    'taper_and_connection_limits': 'arrival SOC' in d and 'setup/disconnection' in d,
    'gurobi_primary_sources': 'docs.gurobi.com' in d and 'support.gurobi.com' in d,
    'scheduler_vs_outcome': 'Pending trials are not results' in d,
    'salvages_correct': 'C1 finished at 18 buses versus control 19; C5 at 17 versus control 16' in a,
    'strict_graph_stop_correct': 'before any pricing iteration' in a and 'route weight 64' in a,
}
summary = json.loads((P.parent/'charging_column_structure/pilot/collections/20260922T053756Z/attempts/729675_r0/summary.json').read_text())
for v, label in zip(summary['variants'], ['Original minute rows', 'Merged rows', 'Start/end equations']):
    row = f"| {label} | {v['build_update_wall_s']:.3f} | {v['mip']['Runtime']:.3f} | "
    checks['native_runtime_'+v['variant']] = row in d
    checks['original_matrix_'+v['variant']] = v['original_matrix_validation']['valid'] and v['finite_pool_fleet_proven'] and abs(v['lp']['objective']-3) < 1e-10
prep = list(csv.DictReader((P.parent/'mip_structure/preparation_results.csv').open()))
first_four = [r for r in prep if r['case'] != 'c3_k15_fresh']
checks['first_four_structure'] = len(first_four) == 4 and all(r['connected_components'] == '1' and r['identical_columns'] == r['redundant_rows'] == '0' for r in first_four)
checks['fresh_fleet_screening'] = all(r['fleet_safe_zero_fix_count'] == '0' for r in first_four if r['case'].endswith('fresh'))
checks['sequential_screening'] = next(r for r in first_four if r['case'] == 'c1_k15_sequential')['fleet_safe_zero_fix_count'] == '8458' and '8,458/130,468' in d
endpoints = list(csv.DictReader((P.parent/'mip_structure/endpoint_results_partial.csv').open()))
c4 = [r for r in endpoints if r['case'] == 'c4_k08_fresh']
checks['c4_five_pool_proofs'] = len(c4) == 5 and all(r['fleet'] == '9' and r['finite_pool_fleet_proven'] == 'True' for r in c4) and 'All five C4 k8 arms prove nine buses' in d
strong = next(r for r in endpoints if r['case'] == 'c1_k15_sequential' and r['arm'] == 'strong_start')
checks['saved_start_runtime_scope'] = abs(float(strong['actual_optimize_wall_s']) - 11.15) < .005 and strong['fleet'] == '15' and strong['offline_start_acquisition_excluded'] == 'True' and 'excluding its acquisition time' in d
checks['zero_initial_occupancy'] = 'zero occupancy before the first event' in d
result = dict(checks=checks, all_passed=all(checks.values()), count=len(checks), new_doc_tab='t.ikbgt85cdszz', notes='Native editable tables. Build is wall seconds; MIP table uses Gurobi Runtime. Historical figures and links retained. No Slides changed.')
(P/'doc_verification.json').write_text(json.dumps(result, indent=2)+'\n')
# The caller may redirect this process's stdout to verification_run.log. Hash
# that completed log in the publication step, not while this process writes it.
hashes = {f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.is_file() and f.name not in {'artifact_hashes.json', 'verification_run.log', 'publish_allowlist.json'}}
(P/'artifact_hashes.json').write_text(json.dumps(hashes, indent=2)+'\n')
print(json.dumps(result, indent=2))
assert result['all_passed']
