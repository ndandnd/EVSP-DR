"""Scientific release rules. Missing results and ordinary covering hits cannot trigger scaling."""
import argparse
import math
import os
from pathlib import Path
import sys
from campaign import read, write, sha, HERE


def valid_at_floor(audit, expected, arm):
    return (audit.get('schema') == 'partille-selection-replay-v1'
            and audit.get('arm') == arm
            and audit.get('instance_sha256') == expected['instance_sha256']
            and audit.get('validated') is True and audit.get('exact_once') is True
            and (arm != 'blind' or audit.get('pricing_executed') is True)
            and audit.get('cover_valid') is True and audit.get('duplicate_service') == 0
            and audit.get('grid_plug_overload_minutes') == 0
            and audit.get('shared_fleet_cap') == expected['time_only_floor'] + 4
            and audit.get('fleet') == expected['time_only_floor']
            and audit.get('trip_count') == expected['n_trips']
            and math.isfinite(float(audit['objective_nonfleet'])))


def latest_attempt(folder, job):
    runs = sorted(folder.glob(f'job{job}_r*'), key=lambda x:int(x.name.rsplit('_r',1)[1]))
    if not runs:
        raise ValueError(f'no attempt for {folder.name}, job {job}')
    return runs[-1]


def bind(audit, selected, summary):
    if audit.get('summary_sha256') != sha(summary) or audit.get('selected_sha256') != sha(selected):
        raise ValueError('native receipt and actual saved selection disagree')


def pilot(root, case, jobs):
    verified = {}
    folder = root/'c2_k08'
    for arm, stem, directory, job in [('blind','blind','blind',jobs['blind']),
                                      ('fixed_giro_diagnostic','witness','fixed_diagnostic',jobs['witness'])]:
        audit = read(folder/f'{stem}_verified.json')
        if not valid_at_floor(audit, case, arm):
            verified[arm] = False
            continue
        attempt = latest_attempt(folder/directory, job)
        if arm == 'blind':
            selection, summary = attempt/'selected_routes.json',attempt/'summary.json'
            info=read(summary).get('integer_pricing') or {}
            if info.get('error') or (info.get('stats') or info).get('pricing_calls',0) <= 0:
                raise ValueError('integer pricing execution is unverified')
        else:
            selection = attempt/'witness/selected_routes.json'
            summary = Path(os.environ['C2_BASELINE'])/'summary.json'
        bind(audit, selection, summary)
        verified[arm] = True
    return all(verified.values()),verified


def paired_outcome(root, cell, arm, expected, job_family):
    attempts=list((root/'paired'/cell/arm).glob('job*_r*/receipt.json'))
    if not attempts:
        raise ValueError(f'missing {cell} {arm}')
    candidates=[p for p in attempts if str(read(p).get('job_family','')) == str(job_family)]
    if not candidates:
        raise ValueError(f'no receipt for the registered job family {job_family}')
    path=max(candidates, key=lambda p:read(p)['started_unix'])
    r=read(path)
    if 'ended_unix' not in r or r['hashes']['instance'] != expected['instance_sha256']:
        raise ValueError(f'incomplete attempt or mismatched data in {cell}/{arm}')
    if r['exit_code']==3 and read(path.parent/'solve/summary.json').get('selection_state') == 'no_selection':
        return False
    if r['exit_code'] != 0:
        raise ValueError(f'execution error in {cell}/{arm}')
    a=read(path.parent/'verification.json')
    bind(a,path.parent/'solve/selected_routes.json',path.parent/'solve/summary.json')
    expected_arm = 'blind' if arm == 'repair' else 'baseline'
    if a.get('instance_sha256')!=expected['instance_sha256'] or a.get('arm')!=expected_arm:
        raise ValueError('audit belongs to another case or arm')
    if a.get('validated') is not True and not (a.get('reason')=='cover is feasible but has duplicate service'
                                               and a.get('cover_valid') is True):
        raise ValueError('validation unresolved; check before comparing')
    return valid_at_floor(a,expected,expected_arm)


def run(stage, root):
    jobs=read(root/'paired_jobs.json')
    cases=read(HERE/'design.json')['cases']
    pilot_hit, evidence=pilot(root,cases['c2_k08'],jobs)
    if stage=='eight' or not pilot_hit:
        return pilot_hit,evidence
    results={}
    for c in range(1,7):
        cell=f'c{c}_k08'
        control=paired_outcome(root,cell,'simple',cases[cell],jobs['control'] if c==2 else jobs['eight'])
        repair=True if c==2 else paired_outcome(root,cell,'repair',cases[cell],jobs['eight'])
        results[cell]={'control':control,'repair':repair}
    controls=sum(r['control'] for r in results.values())
    hits=sum(r['repair'] for r in results.values())
    # A prospectively specified operational expansion rule, not a significance test.
    release=hits>=4 and hits>controls and all(r['repair'] for r in results.values() if r['control'])
    return release,results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['eight','ten'])
    args=p.parse_args()
    root=Path(os.environ['CAMPAIGN_ROOT'])
    try:
        ok, evidence=run(args.stage,root)
        verdict='release' if ok else 'scientific_hold'
        result={'stage':args.stage,'status':verdict,'detail':evidence}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ok=False
        result={'stage':args.stage,'status':'incomplete_evidence','reason':str(exc)}
    write(root/'gates'/(args.stage+'.json'),result)
    print(result)
    sys.exit(0 if ok else 42)
