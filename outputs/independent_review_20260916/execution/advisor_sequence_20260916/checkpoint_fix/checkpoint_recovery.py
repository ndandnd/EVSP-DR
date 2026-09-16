"""Crash-safe staging and explicit fallback for immutable previous CG attempts."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile

SUFFIXES = ('', '.columns.jsonl', '.iters.csv')

class RecoveryError(RuntimeError):
    pass

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

def publish_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()))
    with temporary.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)

def recover(history, attempt, validate, *, copy_file=shutil.copy2):
    """Return a published output stem, or None only for a genuinely first run.

    Validation can repair a torn tail in the private copy, never its source.
    Exceptions from validation reject that candidate; all failures are recorded.
    KeyboardInterrupt/SystemExit propagate. An OS kill leaves a hidden stage,
    never an exposed partial checkpoint. Copy/source mutation is detected.
    """
    history, attempt = Path(history), Path(attempt)
    report = {'schema': 'evsp-checkpoint-selection-v1', 'candidates': [], 'selected': None}
    previous_dirs = [p for p in history.iterdir() if p.is_dir() and p != attempt and not p.name.startswith('.')]
    candidates = []
    for directory in previous_dirs:
        # Original layout and this version's atomically published layout.
        candidates.extend(p for p in (directory/'cg.json', directory/'resume/cg.json') if p.is_file())
    candidates.sort(key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True)
    for status in candidates:
        stage = Path(tempfile.mkdtemp(prefix='.resume-stage-', dir=attempt))
        copied = stage/'cg.json'
        row = {'source_status': str(status), 'accepted': False}
        try:
            source_paths = {suffix: Path(str(status)+suffix) for suffix in SUFFIXES if Path(str(status)+suffix).exists()}
            if '.columns.jsonl' not in source_paths:
                raise RecoveryError('missing source journal')
            before = {suffix: sha(p) for suffix, p in source_paths.items()}
            row['source_sha256'] = before
            for suffix, source in source_paths.items():
                target = Path(str(copied)+suffix)
                copy_file(source, target)
                with target.open('rb') as f:
                    os.fsync(f.fileno())
                if sha(target) != before[suffix]:
                    raise RecoveryError('copy hash mismatch: '+suffix)
            if before != {suffix: sha(p) for suffix, p in source_paths.items()}:
                raise RecoveryError('source checkpoint changed while being copied')
            row['validation'] = validate(copied)
            if row['validation'].get('valid') is not True:
                raise RecoveryError('validator did not confirm valid checkpoint')
            row['published_sha256'] = {suffix: sha(Path(str(copied)+suffix)) for suffix in source_paths}
            row['accepted'] = True
            publish_json(stage/'ready.json', row)
            fsync_dir(stage)
            published = attempt/'resume'
            if published.exists():
                raise RecoveryError('refusing to overwrite an existing published checkpoint')
            os.replace(stage, published)
            fsync_dir(attempt)
            report['candidates'].append(row)
            report['selected'] = str(published/'cg.json')
            publish_json(attempt/'resume_selection.json', report)
            return published/'cg.json'
        except Exception as exc:
            row['accepted'] = False
            row['error'] = type(exc).__name__+': '+str(exc)
            report['candidates'].append(row)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    report['prior_attempts'] = [str(p) for p in previous_dirs]
    publish_json(attempt/'resume_selection.json', report)
    if previous_dirs:
        raise RecoveryError('Prior attempts exist but no valid checkpoint remains; refusing an empty restart. See resume_selection.json.')
    return None
