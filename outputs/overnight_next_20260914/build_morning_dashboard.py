"""Build the current Doc tab from verified, dated summary tables.

This is a reading view. Original and separate-search outcomes stay distinct;
the input summaries retain source hashes and the complete proof information.
"""
import argparse
import csv
import html
import json
from pathlib import Path
from summarize_pool_experiments import verify_mip


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def table(headers, rows):
    def cell(value, tag):
        return f'<{tag} style="padding:6px;vertical-align:top">{html.escape(str(value))}</{tag}>'
    return '<table border="1" style="border-collapse:collapse;width:100%"><tr>' + ''.join(cell(x, 'th') for x in headers) + '</tr>' + ''.join('<tr>' + ''.join(cell(x, 'td') for x in r) + '</tr>' for r in rows) + '</table>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--status', required=True, type=Path)
    parser.add_argument('--time', required=True)
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args()
    folder = args.status
    all_rows = read(folder/'all_chain_extension_results.csv')
    by_case = {r['case_id']: r for r in all_rows}
    longer = read(folder/'longer_gap_results.csv')
    gaps = read(folder/'remaining_gap_results.csv')
    best_repeat = {r['case_id']: r['rerun_buses'] for r in gaps}
    for row in longer:
        if row.get('buses'):
            best_repeat[row['case_id'].replace('_longmip', '')] = row['buses']
    # C2 k25 finished in an earlier follow-up campaign, rather than the final
    # gap cohorts. Verify its current source record instead of hiding that result.
    snapshot = json.loads(args.snapshot.read_bytes())
    followup = snapshot['campaigns']['parallel_pool_followup_20260914']
    case = followup['workflow']['manifest.json']['cases']['w2_k25_longmip']
    item = next(r for r in followup['mip'] if '/w2_k25_longmip/' in r['path'])
    checked = verify_mip(item, case,
                         followup['workflow']['validation.json']['manifest_sha256'],
                         case['source_status_sha256'], case['source_journal_sha256'])
    assert case['input_sha256'] == by_case['w2_k25']['input_sha256']
    best_repeat['w2_k25'] = str(checked['buses'])
    (folder/'c2_k25_repeat_verification.json').write_text(json.dumps(checked, indent=2)+'\n')
    def buses(row):
        return row['integer_buses'] or 'Pending'
    def best(cid):
        return best_repeat.get(cid, '—')
    fixed = []
    for chain in range(1, 7):
        a, b = by_case[f'w{chain}_k25'], by_case[f'w{chain}_k28']
        fixed.append([chain, buses(a), best(a['case_id']), buses(b), best(b['case_id'])])
    cg = []
    for chain in range(1, 7):
        row = by_case[f'w{chain}_k28']
        if row['cg_result_collected'] != 'True':
            cg.append([chain, 'Running', 'Pending', 'Pending', 'Pending'])
        else:
            stop = 'No improving route remains (certified)' if row['cg_pricing_certificate'] == 'True' else 'Four-hour limit'
            cg.append([chain, f"{float(row['cg_minutes']):.1f}", f"{float(row['fractional_route_weight']):.4f}", f"{float(row['weighted_lp_objective']):,.4f}", stop])
    large = read(folder/'compact_large_results.csv')
    compact = {(int(r['chain']), int(r['target']), r['treatment']): r for r in large}
    compact_table = []
    for chain in range(1, 7):
        compact_table.append([chain] + [compact[chain, k, treatment]['buses'] for k in [20, 25] for treatment in ['previous_k_core', 'previous_k_core512']])
    queue = json.loads((folder/'validation.json').read_text())['queue']
    base = 'https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab='
    source = 'https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/' + folder.name + '/'
    parts = [
        '<html><body style="font-family:Arial;font-size:11pt;color:#202124">',
        '<h1>EVSP–DR: current results</h1>',
        f'<p><b>Verified 15 September, {html.escape(args.time)} EDT.</b> Read this tab for current conclusions. Figures and the dated research history remain in the other tabs.</p>',
        '<h2>The main result</h2>',
        '<p><b>All six baseline chains have a 25-bus solution for target 25; several now reach target 28.</b> Some matches require a separate MIP search on the same columns. These larger results do <b>not</b> include shared charger capacity or an ending-SOC requirement.</p>',
        '<h2>Actual buses found</h2>',
        '<p>Target k means the trips came from k GIRO bus duties. “Original” is the one-hour MIP. “Separate search” uses the unchanged saved columns with a larger allowance; it starts a new search tree. A dash means no separate result is needed or available. Pending is not a failed solve.</p>',
        table(['Chain', 'Target 25: original', 'Target 25: separate search', 'Target 28: original', 'Target 28: separate search'], fixed),
        '<p>Every target match shown has its fleet minimum proved within its own saved pool. Charging optimality is a separate question.</p>',
        '<p>The original k16–25 batch matched 35 of 60 targets. Separate searches recovered all 25 misses. Those pools already contained adequate routes. This does not isolate an effect of extra time: CPU hardware and search trajectories can differ.</p>',
        '<h2>What CG achieved at target 28</h2>',
        table(['Chain', 'CG minutes', 'Fractional route weight', 'Weighted LP objective', 'Why CG stopped'], cg),
        '<p>CG minutes include this k’s route import and CG, but exclude earlier k values, graph preparation and MIP. Fractional route weight is the sum of route variables, not the weighted objective. A time-limited result is not a certified full-model lower bound.</p>',
        '<p><b>Three distinct checks:</b> CG convergence proves that pricing found no improving route within its modeled representation and tolerance. A saved-pool fleet proof establishes the fewest buses using only the generated columns. Individual-route replay checks each route’s modeled feasibility. None alone proves full operational optimality.</p>',
        '<h2>Why some integer solutions are worse</h2>',
        table(['Obstacle', 'Evidence', 'What can help'], [
            ['Search has not found the available combination', 'All 25 original k16–25 misses were recovered without adding columns.', 'More effective MIP search; record hardware and search work.'],
            ['Useful integer routes are missing', 'At C1 target 15, both smaller pools prove 16 buses are necessary; the full pool supports 15. Their LP objectives agree.', 'Add different columns; more MIP time alone cannot repair these pools.'],
        ]),
        '<p>In that C1 example, 12 of the known 15-bus solution’s trip patterns are absent from the core pool, and 11 from the expanded pool. All 15 witness routes have positive reduced cost at the smaller pools’ final dual prices. They would not improve the LP, so negative-reduced-cost pricing need not generate them.</p>',
        '<h2>Does keeping fewer starting routes help?</h2>',
        '<p><b>Core:</b> keep earlier integer-solution routes and routes with positive LP weight. <b>Expanded:</b> fill that core to 512 distinct trip sequences. These are starting sequences; CG can add more. All 24 tests at targets 8 and 10 match. At target 15, 9 of 12 match.</p>',
        table(['Chain', 'Target 20: core', 'Target 20: expanded', 'Target 25: core', 'Target 25: expanded'], compact_table),
        '<p>Of these 24 larger results, eight match, eight pools prove the target impossible, and eight misses remain unresolved. Only three CG runs converge; 21 hit four hours. MIPs allow three hours for fleet search, 3.5 hours total. Smaller starts therefore do not reliably preserve the full pool’s integer quality.</p>',
        '<p>In the separate accumulated-CG-time comparison, warm starts match 24 of 24 targets; fresh starts match 6 of 24. All 24 fresh CGs converge. Extra CG time alone does not reproduce the warm results.</p>',
        '<h2>Which GIRO settings are included?</h2>',
        table(['Model or test', 'Settings and result'], [
            ['Large baseline chains', '240 kWh battery; 240 kW charging; no reserve, shared charger capacity or ending-SOC floor. Set covering; inherited columns. Route cost = 100,000 + electricity + 5 per charging start.'],
            ['Reserve and selected speed/capacity tests', '236.44 kWh battery with 15% reserve. Eight of ten one-duty tests use one bus; duty 13405 uses two in both variants. All ten CGs converge. No 65% ending-SOC floor or nonlinear charging.'],
            ['Duty 13408: reserve, shared capacity, PARX at 60 kW', 'One bus recovered; CG takes 77 minutes versus 2.2 minutes with reserve alone. A one-bus test does not establish multi-bus charging feasibility.'],
            ['Harder capacity pricing', 'Two reference pricing calls finish in 3.58 and 3.19 hours. The cached version does not finish either within about four hours. No speedup demonstrated.'],
        ]),
        '<p>For covering results, individual routes were replayed, but removing duplicate trip assignments has not been separately validated. The large-chain table must not be presented as a match to every GIRO constraint.</p>',
        '<h2>Work in progress</h2>',
        f'<p>Queue in this collection: <b>{queue["RUNNING"]} running</b>, {queue["PENDING"]} waiting for required inputs; held historical jobs excluded. No array throttle is blocking this batch.</p>',
        table(['Work', 'Question'], [
            ['Six chains through targets 29–30', 'Graph preparation runs in parallel. Each CG then waits for its own graph and the preceding k’s columns; its MIP follows.'],
            ['Eight pool unions and four unchanged-pool controls', 'Can combining core and expanded columns restore a target fleet? No new CG or GIRO routes are added.'],
            ['Longer MIPs: C1 and C5 at target 27', 'Originals found 28/bound 27 and 29/bound 26. Are target fleets already present in these unchanged pools?'],
        ]),
        '<p>Three sampled target-29 graph builders processed roughly one-third of their source states after 4.9 hours. This preprocessing is separate from CG time; progress is not a reliable completion-time estimate.</p>',
        '<p>The monitor continues hourly, repairs demonstrated execution problems, and reports verified changes or access loss. The next decision is whether pool combinations repair the integer gaps; further algorithm changes should be tested with matched settings.</p>',
        '<h2>Figures and sources</h2>',
        f'<p><a href="{base}t.ts4vwph3s99i">Comparison figures and explanations</a> · <a href="{base}t.h5h2ivyiprly">CG curves and Gantt plots</a>. Existing figures are preserved; check their dates and model settings.</p>',
        f'<p><a href="{source}CHAIN_TABLES.md">Every original chain count, exact LP objective and CG stopping reason</a> · <a href="{source}README.md">Verified source report</a> · <a href="{source}doc_before.md">Detailed dashboard before this consolidation</a> · <a href="{base}t.lumf8xm66fow">Dated research history</a>.</p>',
        '</body></html>',
    ]
    output = '\n'.join(parts)
    (folder/'dashboard.html').write_text(output)
    (folder/'dashboard_tables.json').write_text(json.dumps({'fixed_targets': fixed, 'cg_at_28': cg, 'compact_large': compact_table}, indent=2)+'\n')
    print(json.dumps({'html': str(folder/'dashboard.html'), 'tables': 6}))


if __name__ == '__main__':
    main()
