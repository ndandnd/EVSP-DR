# Checkpoint audit — full40 CG job 341405

**Column recovery is implemented and passed native tests. Automatic recovery under every interruption is not guaranteed.** Two concrete robustness gaps were found; no production code, job, checkpoint or partition was changed.

At the read-only snapshot, job **341405** was pending on graph job **341404_0**. It requests **50 hours / 8 CPUs / 128 GB** on the default partition with requeue enabled and `scaglione-compute-01` excluded. Its application budget is **48 hours**. No full40 CG checkpoint existed yet; the tests below used a separate four-trip fixture and the exact pinned code.

## What survives

| Claim | Verdict | Evidence |
|---|---|---|
| Status JSON is published atomically. | **VERIFIED for ordinary process interruption** | File is flushed/fsynced, then replaced atomically. Injecting failure before replacement leaves the previous complete JSON intact. Parent-directory fsync is absent, so this is not a universal host/NFS-server-crash guarantee. |
| Columns are saved only every 25 iterations. | **REFUTED** | Newly added column batches are flushed/fsynced each iteration. The status JSON is checkpointed every 25 iterations, before that iteration's solve. Iteration CSV is separately flushed/fsynced each completed pricing iteration. |
| An interrupted final journal record can be repaired. | **VERIFIED** | Native resume discarded a truncated final JSON object and recovered all 13 complete fixture columns. Interior corruption was rejected without changing status/journal/iteration files. |
| A journal newer than the status is resumable. | **VERIFIED** | Native test resumed a valid journal paired with an older initializing status and certified the fixture. A status claiming more columns than the journal held was rejected. |
| Resume checks input and model identity. | **VERIFIED** | Code/input/reference/deadhead/tariff hashes, trip IDs, physics, master sense/backend and initialization controls are checked. Changing charging power from 240 to 241 rejected resume before repairing the deliberately torn journal. |
| Resume retains the solved LP basis or an in-progress pricing calculation. | **REFUTED** | A new restricted-master object is built from saved columns. No LP-basis persistence is present. Current LP/pricing work must be repeated. |
| Every requeue receives a new 48-hour application budget. | **REFUTED** | Saved elapsed time is restored from the maximum of the iteration log and status. A synthetic saved 200 seconds with a 120-second cap stopped immediately with no new iteration. |
| Newest corrupt restart copy automatically falls back to an older good checkpoint. | **REFUTED — reproduced gap** | The wrapper reads the newest JSON without catching parse failure. A truncated newest copy raises `JSONDecodeError` before the older valid checkpoint is considered. |
| A termination signal guarantees a quick final save and exit. | **REFUTED — reproduced gap** | Real SIGTERM and SIGUSR1 were handled in the tiny native run, but both were followed by another master LP solve before final publication. On a large pool that solve may outlast the scheduler's grace period. |

## Recovery sequence and limits

The wrapper uses a new `job_r<restart>` attempt directory, selects the newest earlier CG status, checks code/input identity, then copies status, journal and iteration CSV before invoking `--resume`. The original attempt remains untouched. Native resume validates the wider identity, repairs only safe tail damage, reconstructs the pool, restores recorded iteration/time offsets and rebuilds the master. Output locking prevents two writers from sharing one output stem.

The files are **not one atomic transaction**. Journal-ahead-of-status is intentionally supported. The private-attempt copy uses ordinary `shutil.copy2`; interruption during that copy can leave the newest status or journal incomplete. An older good attempt survives, but the current automatic selector may fail rather than fall back. Even a readable newest status with an incomplete journal may fail pool validation without trying an older candidate. This is a recoverability gap, not evidence that previous columns have been destroyed.

Hard-kill losses are not simply “25 iterations.” Most completed column batches survive independently of the status checkpoint. The in-flight LP/pricing work and any unflushed current batch can be lost; that can represent substantial time. Conversely, progress reported in a durable iteration row can precede the corresponding column insertion, so restored iteration count is not a promise that all work from that numbered iteration survived.

The application clock restores **recorded** elapsed time and excludes telemetry overhead. A hard kill during a long iteration can leave elapsed time since the last durable log unrecorded. Actual total cluster time across attempts can therefore exceed the nominal 48-hour application budget; scheduler/attempt accounting must include this lost work when making computational comparisons. Wall/iteration limits are operational controls, not immutable model-identity fields; max iterations is applied per attempt.

Graph preparation is different: a completed, published graph cache is reused and hash-checked. An interrupted unpublished graph build starts again; its internal build progress is not checkpointed. MIP search trees are also not restored.

## Why the current job cannot simply move to Scaglione

The read-only cluster snapshot at **22:46:51 UTC on September 16** showed 52 idle CPUs across the five non-GPU Scaglione nodes (8 on `scaglione-cpu-01`, 44 on `scaglione-cpu-03`). However, every `scaglione-cpu-01` through `-05` advertises **128,350 MB** of Slurm memory, while this CG job requests **128G = 131,072 MiB**. Thus none of those nodes can satisfy its unchanged memory request, even if all their CPUs become idle. The GPU node `scaglione-compute-01` remains excluded by the standing resource policy. CPU availability alone is insufficient to move this job; reducing memory would require a separate, justified resource decision. No job was moved and no request was changed. Source: `../cluster_load.json` (hash retained in `provenance.json`).

## Held recommendations — not applied

1. **Validate restart candidates before selection.** Publish copied checkpoint sets only after all files have been copied and validated; use an atomic staging-directory rename or ready marker. Catch incomplete/corrupt candidates and try older validated attempts, recording the rejection. Never silently restart empty if all checkpoints fail.
2. **Make termination a fast-save path.** Flush the durable files and publish the last valid state, then skip final LP polishing/diversification on shutdown. Check cancellation inside long pricing/master work where practical. Merely increasing the signal grace period does not guarantee completion of a large LP.
3. **Account for all attempts externally.** Use Slurm elapsed/attempt records to include work lost before the next durable timestamp. If a hard total computation cap is required, enforce it across attempts rather than relying solely on saved application time.
4. **For stronger storage-crash durability**, consider fsync of containing directories after rename, with the cluster filesystem's guarantees explicitly understood. Current tests establish process-level behavior, not resilience to a server/storage outage.

## Source trace and tests

Pinned execution: `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`. Source copies and SHA-256 hashes are retained in `sources/` and `provenance.json`.

- `durable_io.py`: `flush_and_fsync`, `atomic_write_json`, `read_jsonl_records`, `exclusive_output_lock`.
- `exact_pricer_expanded.py`: identity checks 1362–1498; repaired-pool validation 1551–1588; signal handling 1619–1629; resume and cumulative offsets 1633–1743; initial/live status 2057–2154; partial status 2431–2478; loop checks 2548–2564; per-batch journal fsync 2869; final LP attempt 3082 onward; final atomic write 3530.
- `production_campaign.py`: attempt directory 156 onward; checkpoint copying 184–196; signal forwarding 214 onward. `production_manifest.json` and `production_readonly_snapshot.json` bind resource requests, tool hashes and current scheduler state.

`native_tests.json` records eight native scenarios plus the reproduced candidate-selection failure. `signal_tests.json` records real OS signals using test-only instrumentation to observe subsequent master solves. `durability_tests.json` records atomic-write fault injection and lock behavior; `iteration_tests.json` checks CSV-tail repair and malformed-interior rejection. `native_test_artifacts.tar.gz` preserves the tiny inputs, native statuses, journals and logs. No Slurm job was submitted for this audit.

A separate descriptive FDL-setting error was discovered while reading the same code: production CG/native MIP use a 1,560-minute station-to-trip window, while the independent input/route checks used the stricter 220-minute default. `../../p3_frolunda/manifest_erratum.json` records the correction without changing its frozen manifest or validated feasible schedules.
