# Partial baseline graph checkpoints — isolated implementation

This patch adds opt-in resumable construction of packed baseline event graphs. Branch `codex/graph-checkpoints-20260921`, commit `71f3acb8f59501f60adfe49fcbfbb85fa8a6fe2a`. [Exact patch](implementation.patch). It changes no running job, existing campaign pin, scientific model, pricing routine, CG checkpoint or completed-cache schema. No cluster submission, cancellation, restart or remote source edit was made.

## Why this is useful

**This is the representation already used by the 44 baseline jobs.** The patch is based on their exact baseline commit `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`, whose graph invocation uses `--event-arc-mode lazy`. An audit of all50 recorded attempts across44 graph owners confirms that mode and source pin. No representation migration is needed. This is not limited to the separate strict-physics packed recovery. [Current-campaign representation evidence](baseline_representation_audit.json).

The current 44 baseline graph builds take an estimated 15–25 hours each. Six confirmed preemptions have already discarded 84.6 minutes of construction. Their progress files record source counts, arc counts and memory, but **not the arc buffers**. Therefore neither this patch nor those logs can recover their interrupted work. The current jobs continue unchanged.

The builder already finalizes sources in deterministic order. Each completed source appends sorted target/cost/reconstruction-recipe arrays and records its contiguous slice. This is a natural checkpoint boundary: no partly constructed row is published, and later rows do not change earlier rows. The transient charging-window cache can be rebuilt without changing the graph.

## Implementation

- `src/graph_arc_checkpoints.py`: immutable raw numeric shards, per-shard hashes covering both row metadata and binary data, continuity checks, exclusive writer lock, fsync and atomic manifest replacement.
- `src/event_pricer_network.py`: optional checkpoint directory, source/input/physics identity, buffer reconstruction, skipping completed sources and per-attempt checkpoint telemetry. Normal construction remains the default.
- `src/exact_pricer_expanded.py`: `--event-graph-checkpoints PATH`, accepted only for event/lazy graphs. It passes the existing content-hash provenance and reports checkpoint telemetry in network metrics for a build/resume attempt.
- `tests/test_graph_arc_checkpoints.py`: forced process kills, orphan handling, exact buffer/pricing/replay comparisons, changed-input rejection, corrupt-data rejection, writer exclusion and completed-cache round-trip.

The directory is stable across attempts and specific to one graph identity. For a future audited worker, pass an owner-case path such as `cases/<graph-owner>/graph-partial/`; keep ordinary attempt logs and final output paths separate. This option was not added to the currently running workers.

Identity contains the existing cache provenance (execution commit and CSV/tariff/reference/deadhead hashes), hashes of every local source module, the prepared graph's full mathematical data and parameters, Python/NumPy versions, byte order and numeric widths. Prepared identity includes all trip times/energies, adjacency, event lattice, SOC grid, normalized prices and derived graph structures. Changes fail closed. This deliberately prefers a conservative mismatch over silent reuse. No cross-commit compatibility exception is introduced.

## Commit and recovery protocol

1. Finish a complete source row and append its packed data in memory.
2. At 256 MiB of new packed data or 300 seconds since the last commit, write the new target/cost/recipe ranges to an unreferenced temporary shard. These thresholds are checked at row boundaries; a slow or unusually large single row can exceed them.
3. Flush/fsync the shard, rename it to its numbered name, and fsync the directory.
4. Write/fsync a new JSON manifest, atomically replace the old manifest and fsync the directory. **Manifest replacement is the commit point.**
5. On restart, lock the directory, validate identity, source order, row continuity, sizes and hashes; reconstruct arrays/slices/sink entries; build only subsequent rows. Unreferenced shards and temporary files are ignored.

A crash after step3 but before step4 leaves an orphan shard. Resume uses the previous committed manifest and safely overwrites that unreferenced shard number. A crash after step4 resumes after the newly committed rows. Referenced shards are not overwritten. One writer owns the lock until graph construction finishes; a second writer fails immediately.

Checkpoint data are raw numeric arrays and JSON. Resume does not deserialize executable pickle data. The prepared-state pickle is hashed in memory only and is never loaded from a checkpoint. The existing completed graph cache continues using its established authenticated pickle format.

## Verification

Ten new checkpoint tests and eighteen existing event/pricing regression tests pass. Forced **SIGKILL after the third manifest commit** resumes after exactly three completed sources. Forced **SIGKILL after the third shard is durable but before its manifest commit** resumes after two sources and ignores the orphan. Resumed packed arrays are byte-identical to uninterrupted arrays; row slices, sink entries, graph metrics, fixed-sequence replay and reduced-cost pricing outputs also agree. A real CSV case independently rebuilt in a fresh process passes identity and resume checks. Completed checkpoints skip every source and round-trip through the existing completed-cache writer/loader.

[Final checkpoint tests](tests_checkpoints_final.txt) · [Final event regression tests](tests_event_regression_final.txt) · [Local overhead sample](toy_overhead.json).

The initial test attempt had four failures in the **test comparison helper** because NumPy arrays cannot be compared with dictionary equality. Converting comparison values to plain lists fixed that helper; it did not require an algorithm correction. The initial log is retained as `tests_initial.txt`.

## Overhead and limits

The local synthetic sample has 32 trips, 3,073 source rows and 803,584 packed bytes. Three descriptive repetitions measured median fresh construction 57.38 ms, checkpointed construction 81.37 ms and completed-checkpoint loading 13.56 ms. Source hashing, fsync and metadata dominate this very small graph. These are local observations, not a production or cluster speedup estimate; ordering and hardware caches are uncontrolled.

A production attempt writes each newly committed packed edge once, plus JSON row metadata; earlier edge buffers are not rewritten at each checkpoint. Shard writes use memory views rather than copying the accumulated graph. Resume reads each shard once to validate its hash and again to reconstruct arrays. Transient loading memory is bounded by a shard buffer plus metadata, apart from the normal graph allocation; a single oversized row can enlarge a shard. Sink entries are reconstructed by binary search in sorted source rows, not a Python loop over every edge.

Partial shards remain after a completed full graph cache is written. **Retaining both requires roughly another full graph's packed storage**, plus metadata and temporary files. No automatic deletion or storage-policy change is included. A future deployment must check available shared storage and validate advisory locking/fsync behavior on the actual cluster filesystem. Local durability tests do not establish those filesystem guarantees. Automatic partial cleanup should happen only after a completed cache and its hash have been independently verified.

The initial problem/event/node preparation is rebuilt on resume; only the expensive arc-construction phase is checkpointed. Warm construction caches are not persisted. Completed-cache identity still includes source commit, so old-pin caches are not silently accepted by this new commit. This patch does not change scheduling, graph admission, the 44 current jobs, or held work.

Independent read-only code review is in progress; its findings will be preserved alongside this handoff.
