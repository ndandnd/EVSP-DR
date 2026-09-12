# w2_k14 cache compatibility audit

The original producer `21fbecba826824c44f897feef038fcf51c532582` and baseline `a29992196acb74d02b8c7891be4061718889999f` have compatible graph construction and serialization for this cache. Every tracked `src` file except `exact_pricer_expanded.py` is byte-identical. This covers all repository-local dependencies, including the event graph, input builder, configuration, utilities, realization, optimizer, validation and durable I/O modules. In `exact_pricer_expanded.py`, ASTs of the expanded network class, cache identity/read/write/hash functions, input-builder calls, and the entire price-load through graph-build/cache-load/metrics block are identical. The changed functions add inheritance, provenance and CLI/status handling; they do not change the graph-construction block.

The instance, tariff, reference and deadhead files match all four original manifest SHA-256 values at producer, baseline and current target. Cache physics are event/lazy, SOC step 2.5 kWh, event support 5 minutes, battery 240 kWh, default charger 240 kW, reserve 0 and strict tariff coverage false.

`evidence.json` records full source blob inventory, critical SHA-256 values, AST checks, exact target source hash and target working-tree status. `audit.py` reproduces this bounded local comparison. The updated snapshot includes the actual uncommitted import fix: only `exact_pricer_expanded.py` and `durable_io.py` differ from baseline among tracked source files. Graph construction, input-builder calls, the expanded class and cache identity/load/write ASTs remain literally identical. `_file_sha256` is **not** literally AST-identical: it adds keyword-only `checkpoint=None` and calls it only under `if checkpoint is not None`. Both cache call sites omit that keyword. Removing exactly that default-None argument and its exact guarded callback produces the original AST, with no other transformations. Six isolated old/new helper executions (empty, one byte, either side of the 1-MiB chunk boundary, and multiple chunks) also produce identical SHA-256 digests.

The entire `durable_io.py` module becomes AST-identical after the same narrowly defined default-None normalization of `read_jsonl_records`; every other function is literally AST-identical, with no added or removed functions. This module change affects optional inherited-journal checkpoints and does not alter graph construction or cache serialization. The current graph reuse conclusion therefore extends to the audited actual fix snapshot. The final commit still needs its hashes pinned by rerunning `audit.py`; unexplained later source changes would invalidate this snapshot conclusion.

## Required explicit reuse attestation

Do not treat the previously relabeled manifest as original build provenance. Preserve the immutable source manifest (`source_manifest.json`, SHA-256 `10c47518d9cc5bdfb908b7eed5176351764e7bffc618fb6d693702fe199e162a`). After the final fix review, create a separate derived manifest for the new consumer with:

- Existing `identity` fields, setting only its execution/consumer `git_commit` to the exact final fix commit so the current loader accepts it.
- `build_identity` equal to the complete original producer `identity`, unchanged.
- Original `pickle_sha256`, `pickle_bytes`, `network_metrics` and `original_build_s`, unchanged.
- An explicit compatibility-attestation object naming producer, baseline, final consumer commit, source-manifest SHA-256, this evidence SHA-256, and the audited graph-code equivalence scope.

Keep both source and derived manifests. Before use, authenticate the actual remote pickle against original SHA-256 `273fb5a6c2c9331200bc652a72f0504021623a0a33757af61d65300c3d7c2385` and size 4,471,034,154 bytes. The existing loader rechecks the hash and exact network metrics (30,347 nodes; 279,321,120 arcs; lattice hash `35d9427e4b2753ea706491bfde1724d060e621e1648c633e73a1088516e9c9f5`). Preserve the underlying pickle without rewriting it.

This audit did not access Unicorn, hash/load the remote pickle, or establish a CG certificate. Its conclusion is source compatibility subject to final-fix confirmation and artifact authentication, not proof based on the overnight run's successful load or its relabeled commit. The evidence supports deliberate authenticated reuse; it does not justify an estimated 9,129-second rebuild by default.
