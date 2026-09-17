"""Publish a complete, hash-matched private copy of one atomic CG pool file."""
from pathlib import Path
import hashlib
import os
import tempfile


def _file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_pool_copy(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('refusing to overwrite destination pool: ' + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + destination.name + '.copy-', dir=destination.parent)
    temporary = Path(name)
    try:
        copied = hashlib.sha256()
        length = 0
        with os.fdopen(fd, 'wb') as output, source.open('rb') as input_file:
            for block in iter(lambda: input_file.read(1024 * 1024), b''):
                output.write(block)
                copied.update(block)
                length += len(block)
            output.flush()
            os.fsync(output.fileno())
        digest = copied.hexdigest()
        if _file_sha256(temporary) != digest:
            raise ValueError('private pool copy hash mismatch')
        if _file_sha256(source) != digest:
            raise ValueError('source pool changed during copying')
        # Each worker owns a unique attempt directory. Never expose a partial
        # pool under the filename searched by the next resume attempt.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError('destination pool appeared during copying')
        os.replace(temporary, destination)
        directory_fd = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {'source': str(source), 'source_sha256': digest,
                'destination': str(destination), 'destination_sha256': digest,
                'bytes': length, 'atomic_replace': True}
    finally:
        if temporary.exists():
            temporary.unlink()
