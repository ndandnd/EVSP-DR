"""Read-only dependency check for the two explicitly scoped cold roots."""
import datetime
import hashlib
import json
import pathlib
import re
import subprocess

SLURM = pathlib.Path('/usr/local/slurm/slurm-25.05.5/bin')
ROOTS = [pathlib.Path('/home/nc437/ladder-lite/phys240kw'),
         pathlib.Path('/home/nc437/ladder-lite/cg_acceleration_20260903')]

def run(args):
    return subprocess.check_output(args, text=True, timeout=30)

queue = run([str(SLURM/'squeue'), '--me', '-h', '-o', '%F|%A|%j|%T|%Z'])
ids = sorted({line.split('|')[0] for line in queue.splitlines()})
records, errors, references = [], [], []
for job in ids:
    try:
        raw = run([str(SLURM/'scontrol'), 'show', 'job', job, '-o'])
    except Exception as exc:
        errors.append({'job': job, 'error': str(exc)})
        continue
    fields = dict(re.findall(r'(\w+)=([^\s]*)', raw))
    record = {'job': job, **{k: fields.get(k) for k in
              ('JobName', 'JobState', 'Reason', 'Command', 'WorkDir', 'StdOut',
               'StdErr', 'Dependency', 'ExcNodeList')}}
    cmd = pathlib.Path(fields.get('Command', '/nonexistent'))
    own_worker = (fields.get('JobName') == 'storage-archive-0912' and
                  str(cmd) == '/home/nc437/ladder-lite/storage_cleanup_20260912_archive/archive.sbatch')
    body = ''
    if cmd.is_file():
        if cmd.stat().st_size > 2_000_000:
            errors.append({'job': job, 'error': 'command too large to inspect'})
        else:
            data = cmd.read_bytes()
            record['command_sha256'] = hashlib.sha256(data).hexdigest()
            body = data.decode('utf-8', errors='replace')
    for root in ROOTS:
        if not own_worker and (str(root) in raw or root.name in body):
            references.append({'job': job, 'root': str(root)})
    records.append(record)
files = []
for root in ROOTS:
    for path in sorted(root.rglob('*')):
        if not (path.name.endswith('.columns.jsonl') or
                (path.parent.name == 'network_cache' and path.suffix == '.pkl')):
            continue
        st = path.lstat()
        files.append({'path': str(path), 'size': st.st_size,
                      'mtime_ns': st.st_mtime_ns, 'inode': st.st_ino,
                      'device': st.st_dev, 'nlink': st.st_nlink,
                      'mode': st.st_mode, 'symlink': path.is_symlink()})
print(json.dumps({'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'queue': queue, 'jobs': records, 'errors': errors,
                  'references': references, 'files': files}, indent=2))
