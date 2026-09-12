"""One immutable CG retry with provenance checks and an outer process-group limit."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def main():
    manifest_path = Path(sys.argv[1]).resolve()
    manifest = json.loads(manifest_path.read_text())
    code = Path(manifest['code'])
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == manifest['execution_commit']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    for item in manifest['required_files']:
        assert sha(item['path']) == item['sha256'], item['path']
    out = Path(manifest['output'])
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive claim prevents accidental duplicate invocation or overwrite.
    with (out.parent / 'attempt_claim.json').open('x') as stream:
        json.dump({'job_id': os.environ.get('SLURM_JOB_ID'), 'started_epoch': time.time(),
                   'manifest_sha256': sha(manifest_path)}, stream)
    assert not out.exists()
    started = time.monotonic()
    save(out.parent / 'start.json', {'manifest': manifest, 'manifest_sha256': sha(manifest_path),
                                   'started_epoch': time.time(),
                                   'resources': {k: os.environ.get(k) for k in
                                     ['SLURM_JOB_ID','SLURM_JOB_PARTITION','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE']}})
    print('RUN', json.dumps(manifest['argv']), flush=True)
    process = subprocess.Popen(manifest['argv'], cwd=code, start_new_session=True)
    watchdog = False
    try:
        returncode = process.wait(timeout=manifest['watchdog_seconds'])
    except subprocess.TimeoutExpired:
        watchdog = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=15)
    finally:
        # The process group is private to this attempt. Remove any orphan workers.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    artifacts = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in out.parent.iterdir() if p.is_file()]
    save(out.parent / 'end.json', {'returncode': returncode, 'watchdog_fired': watchdog,
                                 'elapsed_seconds': time.monotonic() - started,
                                 'ended_epoch': time.time(), 'artifacts': artifacts})
    raise SystemExit(returncode if returncode else 124 if watchdog else 0)


if __name__ == '__main__':
    main()
