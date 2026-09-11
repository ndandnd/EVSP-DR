# Archived pool streams

The storage archive worker handles only the remaining `.json.columns.jsonl` streams under the two explicitly listed source roots:

- `/home/nc437/ladder-lite/finetime`
- `/home/nc437/ladder-lite/factfill`

The immutable path and pre-job (size, mtime) manifest is `remaining_manifest.tsv`. The corrected v3 worker records per-file source SHA-256, gzip archive SHA-256, sizes, timestamps and status in `ARCHIVED_POOLS_STATUS.tsv`. It refuses an archive collision, rechecks source metadata after compression, verifies the decompressed SHA-256, fsyncs the archive, writes and fsyncs a `prepared` record before unlinking the source, and appends `success` only after unlink. On restart it reconciles a durable prepared record by rechecking the archive, decompressed hash and current source SHA before removing a still-present source; later failure rows cannot hide a prepared row. `.json`, `.iters.csv`, logs, snapshots, checkpoints, network caches and every other directory are outside the manifest.

The worker is one serial default-partition job with one CPU and 2 GiB RAM. It excludes `scaglione-compute-01`, uses no automatic requeue, and records a queue/reference audit at start. Serial execution limits NFS I/O contention; it is an archival workload and has no scientific dependencies. The source roots are cold and were checked against all active, pending and held job command paths before submission.

Before running an old experiment that names an archived source path, restore it with `restore_archive.sh` (or `gzip -cd ARCHIVE.gz > ORIGINAL`), then verify the source SHA-256 from `ARCHIVED_POOLS_STATUS.tsv`. The restore utility refuses to overwrite an existing destination and refuses a hash mismatch. The superseded v1 and v2 workers are retained as `archive_remaining_v1_superseded.sbatch` and `archive_remaining_v2.sbatch` for audit only.
