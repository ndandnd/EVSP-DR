# Preserve separate result paths for restarted MIPs

Job 189220, chain 3 k=8 with integer-route seeds, was preempted after 7,451 seconds and automatically restarted three minutes later. The first attempt saved no final result. The registry already retained both attempt directories, but the report selected the last registration by job ID alone. That could attach a retry's result to the interrupted attempt.

The reporting collector now matches job ID and explicit restart count. Unknown restarts have no result binding. Six focused checks passed, including both real attempts and legacy records. The fresh 16:05–16:09 collection verifies distinct r0/r1 paths and preserves their actual scheduler states. No solver, allocation, retry policy or registry was changed.

The original 15:56 collection remains preserved. Its r0 result_path is superseded by this correction; its preemption and timing evidence remains valid. collect_attempts_before.py preserves the original reporting code. audit.json, validation.json and fresh_collection_verification.json contain source hashes and exact scope.
