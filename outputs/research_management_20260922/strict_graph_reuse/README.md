# Strict packed graph reuse: local implementation and bounded native gate

Implementation is complete and independently reviewed at **fedf421461f94727e6b1292a0e7789ab76ed8587**, branch `codex/strict-graph-reuse-20260922`, based on `35770aae2c08e7d5a356cc3b673e67608e5b1036`. Worktree: `.codex-work/strict-graph-reuse-20260922`. `implementation.diff` is the exact four-file code/test patch. No production solve, cluster submission, source push, existing runner-pin change, or full k19 rebuild was performed by this implementation task.

## Behavior and accounting

- New `--mode prepare-graph --graph-cache PATH` builds and atomically publishes a strict packed graph; `--mode verify-graph` validates and loads it without a master or route pool. Requires `--arc-mode lazy` and a no-shared-capacity arm.
- New CG `--graph-cache PATH` loads only. A missing, corrupt, truncated, stale-source, wrong-input or wrong-physics artifact is an error; there is no automatic fallback rebuild.
- Without the flag, CG retains its original graph-inclusive allowance. With a verified cache, the solver allowance begins after validation/load. Status records build/load/solver durations, total routine `runtime_s`, cache identity and payload hashes. The preparation report records its own full routine runtime; campaign accounting must add preparation runtime to the solver-attempt runtime. Do not count the manifest's serialization-and-hashing timer as the entire preparation/export runtime.
- Publication reserves the output name before construction, streams packed arrays in 8 MiB chunks, flushes/fsyncs, atomically renames, then fsyncs the directory. The single file includes bounded manifest/footer metadata, payload length/layout and SHA256. Load verifies identity and the entire payload on the same descriptor before deserializing metadata; packed views are reconstructed without another complete arc-array pickle copy. Existing artifacts are immutable. A process killed before rename can leave a temporary file and reservation lock, but no loadable partial final artifact; remove a stale lock only after confirming no writer remains.
- Identity covers cache schema, Git commit and every Python source file's bytes, runtime/array representation, instance/tariff/reference/deadhead hashes, actual problem adjacency/times/energy, ordered trips/stations, normalized tariffs, event lattice, SOC grid, battery/initial/reserve/terminal rules, charge power, wait/horizon, arc mode, capacity semantics and objective constants. These are trusted local artifacts; a checksum is corruption detection, not authentication of hostile pickle metadata.

## Route-pool lineage

Same-instance old k19 `--resume` remains rejected by the unchanged checkpoint identity function, now before graph work. Do not replace the saved commit, checkpoint ID or original pool to bypass it.

The existing explicit cross-instance gate now additionally permits only exact parent commit `35770aae2c08e7d5a356cc3b673e67608e5b1036`, selected with `--inherit-compatible-commit`. This supports the genuine k17→new k19 path while preserving parent status/pool/instance hashes, unchanged shared-input hashes and physics, source-trip remapping, per-route checkpoint checks and physical replay of every inherited route. Cross-commit shared capacity remains forbidden. The pre-existing audited `50ceb6...` option is retained.

The saved k19 evidence records 8,343 inherited k17 routes and an initial 8,397-column pool, with zero pricing. Reusing the same verified k17 artifacts plus deterministic singletons is the intended new initialization; equality of the complete production 8,397-column initial pool still needs an explicit native preflight comparison before a solve. This local task tested the gate with a synthetic k17-style parent, not the full production pool.

`source_compatibility_audit.json` establishes byte identity to the pinned parent for event graph construction/pricing, configuration/objective constants, problem construction, tariff/realization helpers and physical replay. AST comparisons establish unchanged checkpoint identity, route keys/capacity rows, station power, network builder, master class, MIP functions and the CG pricing loop. Only cache I/O/timing/preflight/reporting and the explicit compatibility allowlist changed.

## Local validation

- New cache/lineage suite: **13 tests passed**, 0.444 seconds test runtime, **1.40 seconds total process wall**, **109,314,048 bytes maximum RSS (104.25 MiB)** on macOS ARM Python 3.12. No solver model is constructed by this suite; Gurobi's Python package is imported by the runner but no license-backed solve is requested.
- Strict/inheritance regression discovery: **28 checks passed** (includes inherited fixture checks); capacity runner: **9 passed**; existing event-network suite: **14 passed**. The adjacent capacity tests use only their existing tiny local solver models.
- New tests cover exact metadata/packed-buffer fingerprints, 16 fixed-dual/objective route and reduced-cost comparisons plus physical replay, a required-charging fixed sequence, DataFrame-backed ProblemData, input/source/physics/order/event mismatch, corruption/truncation/layout rejection before deserialization, interrupted publication, immutable existing artifacts, competing preparation, path aliases, no-solver preparation, missing-cache failure, clock separation and pinned parent acceptance/rejection gates.
- The measured two-trip graph has 60 nodes, 756 arcs, 12,096 packed arc bytes, and a 38,396-byte artifact. Local write/load are approximately 1.12/0.45 ms; these are tiny-fixture measurements, not full-graph forecasts. See `small_graph_measurements.json`, `runtime.json` and test logs.
- `py_compile` and `git diff --check` passed. Independent reviewer confirmed the final implementation and narrow lineage gate without remaining blockers.

## Bounded native preflight plan

Root may deploy a source archive pinned and hash-bound to the implementation commit above, then run one **1 CPU / 4 GiB / 15 minute** default-partition job on the intended NFS artifact filesystem. Exclude `scaglione-compute-01`. This resource limit is for a 104 MiB measured two-trip validation, not an arbitrary throttle on an experiment array. No full k19 build, CG or MIP belongs in this gate.

Dependencies: Python, NumPy, pandas, SciPy and importable `gurobipy`; versions are recorded in `runtime.json`. Run from the clean source root with a unique writable NFS scratch directory supplied as `STRICT_GRAPH_TMP`:

```sh
mkdir -p "$STRICT_GRAPH_TMP"
TMPDIR="$STRICT_GRAPH_TMP" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /usr/bin/time -v python3 -m unittest discover -s tests -p test_strict_event_graph_cache.py -v
```

Record archive/source hashes, execution commit, package versions, filesystem free space, wall time and peak RSS. The suite's corruption/interrupted-write tests must pass on NFS; confirm no published partial cache or unexplained lock remains. Root owns deployment/job artifacts in `native_preflight/`.

Before any later full preparation, retain the prior k19 result/budget; use precisely the original instance/tariff/reference hashes and 239.01 kWh battery, 35.8515 kWh reserve, 2.5 kWh SOC grid, 5-minute event block, 240/60 kW non-PARX/PARX power, 1,560-minute wait bound, no shared capacity and packed arcs. Measure native cold/load peak RSS and storage headroom; the old graph had 3,243,291,120 packed bytes and a 6.01 GiB measured build MaxRSS, but this patch makes no full-size loading-memory claim. Plan temporary-plus-final artifact space with metadata/headroom, and retain both preparation and load/solver timing. No k20 autoqueue or production scheduling authorization is supplied by this report.
