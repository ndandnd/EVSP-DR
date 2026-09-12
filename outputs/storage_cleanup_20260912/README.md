# Unicorn storage cleanup — 12 September 2026 UTC

User-authorized space reclamation with preserved research evidence. Initial scan at 04:11 UTC: approximately 889 GiB under `/home/nc437`, of which 845 GiB is `ladder-lite`. GNU `du -h` uses binary units; totals are live observations, not a quota. Shared filesystem: 130 TiB total, 31% used. Earlier September 11 reclamation of 74,782,010,325 bytes is separate and must not be counted again.

## Protected work and coordination

Both active Astra tasks were consulted before selecting files. The DR task (`01a07ecc-9b77-79b3-9782-e4308a80ba07`) confirmed no new overnight dependency on `phys240kw` or `cg_acceleration_20260903`. The V2G task (`01a08d89-2673-7ff1-968a-714f62d1a10e`) requires its stochastic-review root intact. Protected: the held job537227 threshold root, every nested_warm root, the fresh84 campaign including its cache, overnight_extension_20260912, all execution/source repositories, V2G roots, environments, and all current/pending/held job inputs. No research jobs are canceled, released, requeued, or reconfigured by this cleanup. The shared SSH master is preserved.

Current Google Doc status was read through its authenticated browser export. It now distinguishes fresh covering, inherited covering, station-capacity, and matched-return-energy experiments. The other DR task owns concurrent document updates; a precise archival result will be sent to it for inclusion. Google Slides are untouched.

## Exact scope

`manifest.json` and the four disjoint TSV manifests identify 156 regular files, each with one hardlink at audit time:

| Category | Files | Original bytes |
|---|---:|---:|
| phys240kw column journals, last written August20–21 | 72 | 34,513,843,153 |
| cg_acceleration_20260903 column journals, last written by September5 | 72 | 44,927,269,098 |
| cg_acceleration_20260903 graph-cache pickles | 12 | 69,880,810,653 |
| Total | 156 | 149,321,922,904 |

Result/status JSON, iteration CSVs, logs, timing evidence, cache manifests, source code, and all other files remain at their original paths. No unique evidence is discarded. Missing original large-file paths after successful archival are intentional; restore before rerunning a legacy command that expects them.

`dependency_audit.json` records116 queued job groups with zero detected references to the two candidate roots and zero inspection errors. The audit examines job paths and command scripts, complemented by explicit owner confirmation about indirect dependencies. Each worker repeats the dependency check at startup. No automatic expansion to other directories is authorized by these manifests.

## Execution and integrity

Remote archive root: `/home/nc437/ladder-lite/storage_cleanup_20260912_archive`.

Four balanced, single-CPU maintenance workers, each with2GiB RAM, run on default_partition, excluding scaglione-compute-01, no automatic requeue,24h scheduler limit. This is a bounded archival task, not a CG experiment or a change to the default50 CG concurrency policy. Four streams bound maintenance CPU/I/O exposure while current research runs continue. Each manifest is approximately37.3GB. Exact scheduler IDs and effective exclusions will be recorded in submission evidence.

The worker is derived from the previously completed September11 v3 archival worker. It verifies immutable package checksums, uses a per-group lock, enforces exact source/archive allowlists, rejects symlinks and multiply-linked files, hashes the source, writes gzip level1, fsyncs it, verifies gzip integrity and the decompressed SHA-256, rechecks exact size, nanosecond mtime, inode, mode and link count against the original JSON manifest (device identity is checked within each worker, because NFS device numbers differ across clients), and durably records both hashes before removal. A final source hash/identity check precedes unlink; the source directory is fsynced. Per-file prepared/success records support safe interruption recovery. An archive published before its prepared record is deliberately a fail-closed orphan: the original is retained and manual reconciliation is required. No file is removed merely because its run is old or its scheduler state is COMPLETED.

Twelve remote temporary-file fixtures passed: normal archive+restore; interrupted prepared state with source present; prepared state after unlink; source mutation; invalid archive mapping; symlink; hardlink; orphan archive; failure to durably record preparation; same-size/in-place-identity replacement; and dangling symlink during prepared recovery; and cross-client NFS device numbers. Restore also refuses an existing destination. `bash -n` passed. This tests archival integrity, not solver correctness.

## Restoration

Locate a source in `status_0.tsv` through `status_3.tsv` after completion. Use its archive path and source_sha256:

```bash
bash /home/nc437/ladder-lite/storage_cleanup_20260912_archive/restore_archive.sh \
  ARCHIVE.gz ORIGINAL_PATH SOURCE_SHA256
```

Restore verifies the full decompressed content hash, uses exclusive publication, and fsyncs the restored file/directory. Content is byte-identical; restored inode and timestamps need not equal the original. Do not restore into a path an active writer is using. Archives and their hashes remain after restoration. Git stores this procedure, code, manifests and audit records; Git does not contain the149GB source pools or replace their verified compressed archives.

## Status

Initial array950484 failed closed on client-local NFS device-number differences before any file removal. Probe950543 established the cause; archive_v2.sbatch corrects only the cross-client device comparison. Twelve fixtures pass. See INITIAL_ATTEMPT.md. No successful archival or space savings claimed until completion evidence is collected.
