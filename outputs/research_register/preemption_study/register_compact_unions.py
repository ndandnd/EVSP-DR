"""Register this campaign's exact restart paths under the established lock.

Run on Unicorn before collection. This changes metadata only, never jobs.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import subprocess


def main():
    home = Path('/home/nc437/ladder-lite')
    campaign = home / 'compact_pool_union_20260915'
    if not (campaign / 'jobs.json').exists():
        print(json.dumps({'added': 0, 'reason': 'production not registered'}))
        return
    subprocess.run(['/home/nc437/evsp_env/bin/python', str(campaign / 'export_registry.py')],
                   check=True, capture_output=True, text=True)
    proposed = json.loads((campaign / 'mip_registry_additions.json').read_bytes())['cases']
    path = home / 'mip_preemption_study_20260911' / 'registry.json'
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        raw = path.read_bytes()
        data = json.loads(raw)
        known = {(str(r['job_id']), r.get('result_path')): r for r in data['cases']}
        added = []
        for row in proposed:
            assert row['campaign'] == 'compact_pool_union_20260915'
            assert Path(row['result_path']).is_relative_to(campaign)
            key = str(row['job_id']), row['result_path']
            if key in known:
                for field in ('case_id', 'cohort', 'restart_count', 'solver_budget_s'):
                    assert str(known[key].get(field)) == str(row[field]), (key, field)
                continue
            data['cases'].append(row)
            known[key] = row
            added.append(row)
        if added:
            temp = path.with_name(path.name + '.compact-union.' + str(os.getpid()))
            with temp.open('w') as stream:
                json.dump(data, stream, indent=2, allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(path)
        receipt = dict(utc=now, added=len(added), registered_cases=len(data['cases']),
                       before_sha256=hashlib.sha256(raw).hexdigest(),
                       after_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       additions=added, no_scheduler_changes=True)
        if added:
            dest = campaign / 'registry_receipts'
            dest.mkdir(exist_ok=True)
            (dest / (now.replace(':', '') + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
        (campaign / 'latest_registry_merge.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'additions'}))


if __name__ == '__main__':
    main()
