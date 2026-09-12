"""Reproduce baseline phase shares and conditional Amdahl calculations; no benchmarks."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'outputs/algorithm_review_20260912/runtime_evidence.json'
EXTRACTOR = SOURCE.with_name('extract_runtime_evidence.py')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(xs):
    return {'median': median(xs), 'min': min(xs), 'max': max(xs), 'n': len(xs)}

def main():
    evidence = json.loads(SOURCE.read_text())
    rows = evidence['records']
    assert len(rows) == len({r['path'] for r in rows}) == 98
    snapshot = Path(evidence['snapshot_path'])
    assert sha(snapshot) == evidence['snapshot_sha256']
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('extractor', EXTRACTOR)
    extractor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extractor)
    assert extractor.extract(json.loads(snapshot.read_text())) == rows
    fields = {
        'incidence': 'incidence_s',
        'pricing_inclusive': 'pricing_batch_inclusive_s',
        'enrichment_exclusive': 'pricing_enrichment_exclusive_s',
        'shortest_path': 'pricing_shortest_path_s',
        'master': 'master_s',
        'graph_load_or_build': 'network_build_or_load_s',
        'io_fsync': 'iteration_and_journal_fsync_s',
    }
    checks = []
    for r in rows:
        assert r['wall_s'] > 0
        assert all(r[f] >= 0 for f in fields.values())
        assert abs(r['pricing_batch_inclusive_s'] - r['pricing_shortest_path_s'] - r['pricing_enrichment_exclusive_s']) < 1e-8
        measured = sum(r[fields[k]] for k in ['incidence','pricing_inclusive','master','graph_load_or_build','io_fsync'])
        residual = r['wall_s'] - measured
        assert residual >= -1e-6, (r['path'], residual)
        checks.append(residual)
    fresh = [r for r in rows if r['campaign'] in ('covering_complement75','covering_rerun9')]
    assert len(fresh) == 84 and all(r['network_cache_hit'] for r in fresh)
    by_k = {}
    cases = []
    for k in (5,10,15):
        group = [r for r in fresh if r['k'] == k]
        assert len(group) == 6
        measures = []
        for r in group:
            wall = r['wall_s']
            shares = {label:r[field]/wall for label,field in fields.items()}
            shares['unaccounted'] = 1-sum(shares[label] for label in ['incidence','pricing_inclusive','master','graph_load_or_build','io_fsync'])
            measures.append({
                'source_path':r['path'], 'k':k,'chain':r['chain'], 'wall_s':wall,
                'phase_shares':shares,
                'incidence_zero_cost_ceiling':wall/(wall-r['incidence_s']),
                'incidence_zero_cost_saved_minutes':r['incidence_s']/60,
                'pricing_2x_hypothetical_speedup':1/(1-shares['pricing_inclusive']/2),
                'pricing_5x_hypothetical_speedup':1/(1-shares['pricing_inclusive']*0.8),
            })
        cases.extend(measures)
        by_k[str(k)] = {
            'wall_minutes':stats([x['wall_s']/60 for x in measures]),
            'phase_shares':{label:stats([x['phase_shares'][label] for x in measures]) for label in measures[0]['phase_shares']},
            **{label:stats([x[label] for x in measures]) for label in ['incidence_zero_cost_ceiling','incidence_zero_cost_saved_minutes','pricing_2x_hypothetical_speedup','pricing_5x_hypothetical_speedup']},
        }
    result = {
        'kind':'measured_baseline_profile_and_conditional_arithmetic_only',
        'source':str(SOURCE), 'source_sha256':sha(SOURCE),
        'extractor_sha256':sha(EXTRACTOR), 'script_sha256':sha(Path(__file__)),
        'snapshot_sha256':sha(snapshot), 'baseline_execution_commits':sorted({r['source_provenance']['git_commit'] for r in fresh}),
        'validation':{'records':98,'fresh_cached_records':84,'source_snapshot_matched':True,'negative_exclusive_count':0,'overcount_count':0,'residual_seconds_all_98':stats(checks)},
        'cohort':'Fresh covering; singleton initialization; cached event graphs; six certified cases per k. Pricing certificates apply to recorded conservative expanded-grid model.',
        'aggregation':'Each share or speedup is computed per case first; summary median/min/max are over case ratios. No ratio of median phase time to median wall time.',
        'assumptions':['Timer intervals are comparable elapsed wall times and nonoverlapping except shortest-path/enrichment within inclusive pricing. Arithmetic checks cannot prove timer placement correctness.','Incidence ceiling removes ALL measured incidence time with zero replacement cost and unchanged CG trajectory; actual redundant-removal savings can be smaller.','Pricing scenarios accelerate the ENTIRE inclusive pricing batch by 2x/5x, leave every other phase unchanged, and preserve CG trajectory. They are hypothetical, not measured improvements.','Unaccounted residual overhead remains unchanged; scheduler wait and cold graph build excluded from fresh cached cohort.'],
        'by_k':by_k,'cases':cases,
    }
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    def fmt(v,mult=1,digits=2):
        return f"{v['median']*mult:.{digits}f} [{v['min']*mult:.{digits}f}, {v['max']*mult:.{digits}f}]"
    lines = ['# Baseline profile and conditional ceilings','',
        'These are measured baseline phase shares and arithmetic scenarios, **not measured improvements from implemented changes**. Proposed changes were not implemented in this frozen baseline. Current local prototypes require their own paired measurements.', '',
        'Source: `../../algorithm_review_20260912/runtime_evidence.json`, reconciled exactly against its frozen collector snapshot (98 records; 84 fresh cache-hit cases). The following rows use six fresh covering/singleton cases per k. Entries are **median [minimum, maximum] of per-case ratios**, never ratios of medians.', '',
        '| k | Wall minutes | Incidence % | Pricing batch % | Enrichment only % | Master % | Graph load % |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for k,b in by_k.items():
        lines.append('| '+' | '.join([k,fmt(b['wall_minutes'])]+[fmt(b['phase_shares'][f],100) for f in ['incidence','pricing_inclusive','enrichment_exclusive','master','graph_load_or_build']])+' |')
    lines += ['', '`pricing_extra_columns` is inclusive of shortest-path work. Exclusive enrichment is batch minus shortest path; it is already included in pricing and must not be added again. Graph timing includes cache loading on hits.', '',
        '| k | Remove all incidence: ceiling × | Maximum minutes saved | Hypothetical entire pricing 2×: overall × | Hypothetical entire pricing 5×: overall × |',
        '|---:|---:|---:|---:|---:|']
    for k,b in by_k.items():
        lines.append('| '+' | '.join([k]+[fmt(b[f],digits=3 if f!='incidence_zero_cost_saved_minutes' else 2) for f in ['incidence_zero_cost_ceiling','incidence_zero_cost_saved_minutes','pricing_2x_hypothetical_speedup','pricing_5x_hypothetical_speedup']])+' |')
    lines += ['',
        'Incidence ceiling: `T / (T − I)`, minutes saved `I / 60`. This assumes **all** incidence time disappears, zero replacement cost, identical iteration count/columns/solver trajectory, and unchanged other costs. Removing only redundant work cannot be credited with this entire budget without measurement.', '',
        'Pricing scenarios: `1 / (1 − p + p/s)` where `p` is each case’s inclusive pricing share and `s` is 2 or 5. Neither scenario asserts that these phase accelerations are feasible. Improvements to one subroutine cannot automatically receive the entire pricing budget.', '',
        'Validation: all 98 records have nonnegative exclusive enrichment and no overcount when summing graph + inclusive pricing + incidence + master + fsync. This is arithmetic consistency, not independent proof of instrumentation boundaries. Unaccounted time is retained in the denominator and is not credited as savings. Selected-case residual shares:', '',
        '| k | Unaccounted % |', '|---:|---:|']
    for k,b in by_k.items():
        lines.append(f"| {k} | {fmt(b['phase_shares']['unaccounted'],100)} |")
    lines += ['',
        'Cold graph construction belongs to a separate cohort: seven cold and seven cache-hit overnight records are excluded here. The review’s cold `d00_g0` example is not a paired cache speedup. Scheduler queue time is outside these process wall times. A separate seven-hour capacity call cannot be extrapolated into savings for these baseline cases without matching input, invocation count, timer scope, and solver trajectory.', '',
        'The source records retain input hashes, execution commit, objective, physics and certificate scope. This analysis makes no new feasibility, finite-pool MIP, full-model lower-bound, or GIRO-attainment claim. It does not run production code, submit jobs, or alter experiment status.', '',
        'Reproduce from the repository root: `python3 outputs/algorithm_benchmarks_20260912/profile/quantify_profile.py`. Full per-case ratios and provenance hashes are in `summary.json`; source raw records are referenced rather than duplicated.', '']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'validated':result['validation'],'by_k':by_k},indent=2))

if __name__ == '__main__':
    main()
