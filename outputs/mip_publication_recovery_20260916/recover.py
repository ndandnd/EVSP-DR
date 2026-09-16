"""Recover a completed result after wrapper watchdog failure; preserve failure evidence."""
from pathlib import Path
import argparse, collections, datetime, hashlib, json, os, subprocess

H = Path('/home/nc437/ladder-lite')
B = H / 'chain_extension_31_32_20260915'
C = B / 'cases/w2_k31'
A = C / 'mip/228599_r0'
S = '/usr/local/slurm/slurm-25.05.5/bin/'
def read(p): return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v, separators=(',', ':')).encode()).hexdigest()
def put(p, content):
    if p.exists():
        assert p.read_bytes() == content, 'Refuse to replace existing evidence'
        return
    tmp = p.with_suffix(p.suffix + '.recovery-tmp')
    with tmp.open('xb') as f:
        f.write(content); f.flush(); os.fsync(f.fileno())
    os.link(tmp, p); tmp.unlink()
def encoded(v): return (json.dumps(v, indent=2) + '\n').encode()
def main(publish):
    assert '228599' not in subprocess.check_output([S+'squeue','-u','nc437','-h','-o','%i'], text=True).split()
    acct = subprocess.check_output([S+'sacct','-u','nc437','-S','2026-09-16','-j','228599','--format=JobID,State,ExitCode,Submit','-P'], text=True)
    assert '228599|FAILED|124:0|2026-09-15T10:23:13' in acct
    e, r, m = read(A/'execution.json'), read(A/'result.json'), read(B/'manifest.json')
    assert e['watchdog_time_limit'] and e['returncode'] == -15 and not e['signals']
    assert sha(B/'manifest.json') == e['manifest_sha256']
    assert sha(C/'cg.json') == e['source_status_sha256'] == r['source_result_sha256']
    assert sha(r['source_journal']) == e['source_journal_sha256'] == r['source_journal_sha256']
    assert r['physical_replay_validated'] and r['stage1_incumbent_validated']
    assert r['two_stage']['stage2_incumbent_validated'] and r['status_name'] == 'TIME_LIMIT'
    assert r['two_stage']['stage1_time_limit_s'] == 1800
    assert r['two_stage']['stage2_fleet_constraint'] == 'at_most'
    prov = r['mip_provenance']
    assert prov['git_commit'] == prov['final_observed_git_commit'] == m['mip_execution_commit']
    assert prov['tracked_clean_at_end'] and not prov['git_dirty']
    case = m['cases']['w2_k31']
    assert sha(B/'code/data'/case['csv']) == case['input_sha256'] == e['input_sha256']
    assert r['physical_pool_audit']['input_hashes']['instance_sha256'] == case['input_sha256']
    routes = r['selected_routes']; assert len(routes) == r['buses'] == 32
    coverage = collections.Counter(t for route in routes for t in route['trips'])
    assert set(coverage) == set(range(case['trip_count']))
    hashes = [hashlib.sha256(json.dumps({k:x.get(k) for k in ('trips','route_nodes','charging_stops','cost')}, sort_keys=True, separators=(',', ':')).encode()).hexdigest() for x in routes]
    assert hashes == r['selected_route_hashes'] and digest(hashes) == r['selected_route_set_sha256']
    assert A.joinpath('result.json').stat().st_mtime < e['ended_epoch']
    assert r['end_to_end_before_publication_s'] < 6300
    checks = dict(status='passed', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), scheduler=acct,
                  result_sha256=sha(A/'result.json'), execution_sha256=sha(A/'execution.json'),
                  manifest_sha256=sha(B/'manifest.json'), buses=32, covered_trips=len(coverage),
                  duplicate_trips=sum(v>1 for v in coverage.values()), individual_replay_source_flag=True,
                  independent_physical_replay=False, scientific_values_changed=False,
                  scheduler_failure_preserved=True, root_cause='Process had not exited after result publication; exact shutdown cause undetermined.')
    if publish:
        put(C/'mip_result.json', (A/'result.json').read_bytes())
        put(C/'mip_provenance.json', encoded(dict(result_path=str(A/'result.json'), result_sha256=sha(A/'result.json'),
            source_status_sha256=e['source_status_sha256'], execution_commit=m['mip_execution_commit'],
            recovery_record=str(C/'mip_publication_recovery.json'))))
        put(C/'mip_publication_recovery.json', encoded(checks))
    print(json.dumps(checks))
if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--publish', action='store_true'); main(p.parse_args().publish)
