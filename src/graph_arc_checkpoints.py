"""Atomic, hash-checked completed-source shards for packed event graph builds.

These checkpoints contain raw numeric buffers and JSON metadata, never executable
pickle payloads. A manifest replacement is the only commit point. One writer owns
an advisory lock for the entire graph build; unreferenced files are ignored.
"""
from bisect import bisect_left
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

SCHEMA = 'evsp-packed-source-shards-v1'
BUFFERS = (('_arc_targets', 'I', 4), ('_arc_costs', 'd', 8), ('_arc_recipes', 'I', 4))


def digest(path, prefix=b''):
    h = hashlib.sha256(prefix)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ArcCheckpoints:
    def __init__(self, directory, identity, source_order, *,
                 shard_bytes=256 * 1024 * 1024, interval_s=300.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / 'writer.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        self.identity = identity
        self.source_order = source_order
        self.shard_bytes = shard_bytes
        self.interval_s = interval_s
        self.manifest_path = self.directory / 'manifest.json'
        self.manifest = {'schema': SCHEMA, 'identity': identity, 'shards': []}
        self.finished = self.edge_start = 0
        self.last_commit = time.monotonic()
        self.io_s = 0.0
        self.resumed_sources = 0
        self.bytes_written = 0
        try:
            if self.manifest_path.exists():
                self.manifest = json.loads(self.manifest_path.read_text())
                if (self.manifest.get('schema') != SCHEMA
                        or self.manifest.get('identity') != identity):
                    raise ValueError('graph checkpoint identity mismatch')
        except BaseException:
            self.close()
            raise

    def close(self):
        self.lock.close()

    def restore(self, network):
        started = time.monotonic()
        for number, shard in enumerate(self.manifest['shards']):
            name = f'shard-{number:06d}.bin'
            if shard['file'] != name:
                raise ValueError('graph checkpoint shard name/order mismatch')
            path = self.directory / name
            count = shard['edge_count']
            rows = shard['rows']
            expected_sources = self.source_order[self.finished:self.finished + len(rows)]
            if (not rows or [row[0] for row in rows] != expected_sources
                    or shard['edge_start'] != self.edge_start or count < 0):
                raise ValueError('graph checkpoint source/edge continuity mismatch')
            cursor = self.edge_start
            for source, begin, end in rows:
                if begin != cursor or end < begin:
                    raise ValueError('graph checkpoint row offsets mismatch')
                cursor = end
            if cursor != self.edge_start + count:
                raise ValueError('graph checkpoint edge count mismatch')
            if path.stat().st_size != count * 16 or digest(path, json.dumps(rows, separators=(',', ':')).encode()) != shard['sha256']:
                raise ValueError('graph checkpoint shard hash/size mismatch')
            with path.open('rb') as stream:
                for attribute, _kind, _width in BUFFERS:
                    getattr(network, attribute).fromfile(stream, count)
            for source, begin, end in rows:
                network._arc_slices[source] = (begin, end)
            # Reconstruct sink entries from verified arrays, not duplicate JSON.
            for source, begin, end in rows:
                index = bisect_left(network._arc_targets, network.SINK, begin, end)
                if index < end and network._arc_targets[index] == network.SINK:
                    network.sink_arcs.append((source, network._arc_costs[index], None))
            self.finished += len(rows)
            self.edge_start += count
        self.resumed_sources = self.finished
        self.io_s += time.monotonic() - started
        return self.finished

    def maybe_commit(self, network, finished, *, force=False):
        if finished == self.finished:
            return
        count = len(network._arc_targets) - self.edge_start
        if (not force and count * 16 < self.shard_bytes
                and time.monotonic() - self.last_commit < self.interval_s):
            return
        started = time.monotonic()
        sources = self.source_order[self.finished:finished]
        rows = [(source, *network._arc_slices[source]) for source in sources]
        number = len(self.manifest['shards'])
        name = f'shard-{number:06d}.bin'
        temporary = self.directory / ('.' + name + '.' + uuid.uuid4().hex)
        h = hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode())
        try:
            with temporary.open('wb') as stream:
                for attribute, _kind, _width in BUFFERS:
                    view = memoryview(getattr(network, attribute))[self.edge_start:]
                    try:
                        stream.write(view)
                        h.update(view)
                    finally:
                        view.release()
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.directory / name)
            fsync_directory(self.directory)
            shard = {'file': name, 'sha256': h.hexdigest(),
                     'edge_start': self.edge_start, 'edge_count': count, 'rows': rows}
            updated = dict(self.manifest, shards=self.manifest['shards'] + [shard])
            manifest_tmp = self.directory / ('.manifest.' + uuid.uuid4().hex)
            try:
                with manifest_tmp.open('w') as stream:
                    json.dump(updated, stream, separators=(',', ':'), allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(manifest_tmp, self.manifest_path)
                fsync_directory(self.directory)
            finally:
                manifest_tmp.unlink(missing_ok=True)
            self.manifest = updated
            self.finished = finished
            self.edge_start = len(network._arc_targets)
            self.bytes_written += count * 16
            self.last_commit = time.monotonic()
        finally:
            temporary.unlink(missing_ok=True)
            self.io_s += time.monotonic() - started
