# Full40 checkpoint fallback fix — tested and staged, jobs still held

**The wrapper now rejects bad checkpoints and tries older valid ones, instead of stopping at the newest damaged file.** It publishes a complete copied checkpoint directory atomically. If prior attempts exist but none can be validated, it stops explicitly; it never silently starts an empty pool. The CG solver remains pinned to `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`.

Root subsequently staged and hash-verified all five files in the versioned `checkpoint_v2/` directory. No existing wrapper, manifest, checkpoint or job command was changed. CG341405 and MIP341406 remain held; see `deployment_receipt.json`. No replacement was submitted because a whole-job memory request<=120G was not established. Tests ran in isolated temporary directories.

## Changes

- `campaign.py`: versioned worker wrapper based on the audited production source; scientific CG/graph arguments are unchanged. It accepts a separate versioned manifest and invokes checkpoint recovery. Its `submit` entry point is disabled.
- `checkpoint_recovery.py`: considers earlier original and resumed attempts, newest first. It copies files to a hidden staging directory, verifies source/copy hashes and detects concurrent source changes. It validates the private copy, writes a ready record, fsyncs it and atomically renames the whole directory to `attempt/resume/`. It records every rejection in `resume_selection.json`.
- `checkpoint_validator.py`: loads the pinned solver's CLI defaults and calls its own input/model identity checks, safe journal-tail repair, pool checks and iteration-log checks. It does **not** run CG, expand the event graph, perform pricing or solve an LP. Only the private staged files can be repaired.
- `manifest.json`, `worker.sub`: deployment package for `review_full40_20260916/checkpoint_v2/`. The original manifest and original graph worker remain untouched. New file hashes are added to the versioned manifest.

A process killed during copying leaves only a hidden incomplete staging directory. A later attempt ignores that directory and can select the original good source. A kill after directory publication leaves a complete candidate available for subsequent recovery. Atomicity is for the checkpoint set's initial publication; the unchanged native solver subsequently updates its status/journal/iteration files using their existing durability protocol. `ready.json` hashes describe that initial copied state, not the files after resumed CG has progressed.

## Tests

All **11 tests passed** using the cluster's pinned native code and isolated four-trip data:

| Test | Result |
|---|---|
| Latest status is malformed JSON | Older valid checkpoint selected |
| Latest journal is incomplete relative to saved pool | Older valid checkpoint selected |
| Latest model identity differs | Older valid checkpoint selected |
| Journal contains more progress than status | Newest checkpoint accepted |
| Journal ends with a torn JSON record | Private copy repaired; source unchanged |
| Copy raises an injected I/O failure | Older valid checkpoint selected |
| Every candidate is invalid | Explicit failure; no empty restart |
| Prior attempt has no published status | Explicit failure; no empty restart |
| Genuine first attempt | Fresh start allowed |
| Actual child-process hard exit during copying | No partial checkpoint published; next attempt recovers older source |
| Native CG resumes the published nested checkpoint | 13 columns recovered; tiny case certified |

Source files stayed unchanged during every validation/fallback test. `tests.json` records source and published hashes, rejected-candidate reasons and accepted validation results. `native_test_artifacts.tar.gz` retains fixtures/logs/results. `static_validation.json` additionally verifies compilation, unchanged CG and graph command arguments, unchanged solver commit and unchanged resource requests. `campaign.diff` isolates the wrapper edits.

## Deployment recipe — versioned files staged; no replacement scheduled

1. Keep original `/home/nc437/ladder-lite/review_full40_20260916/manifest.json`, its wrapper and existing graph work unchanged.
2. Copy **only** `campaign.py`, `checkpoint_recovery.py`, `checkpoint_validator.py`, `worker.sub`, and `manifest.json` into a **new** `checkpoint_v2/` directory under that campaign root. Compare the copied hashes to this package's `provenance.json`.
3. The versioned worker sets `EVSP_CAMPAIGN_MANIFEST` to `checkpoint_v2/manifest.json`, checks original and new tooling hashes, and continues using the existing `code/`, input hashes, case directories and graph cache. Any separately approved resource adjustment belongs in this new manifest and its own execution receipt, not in the original frozen manifest.
4. If/when scheduling is separately authorized, invoke this versioned `worker.sub` with `cg <existing-case-id>` and preserve the actual graph dependency. Scheduler actions are deliberately not provided or executed by this code task. The existing held job still references the old wrapper; copying this package alone does not change that job's command.
5. Resumed outputs live at `cases/<case>/cg/<job>_r<restart>/resume/cg.json`; the normal case-level `cg.json` symlink is still published on successful completion. Collectors that search attempt directories directly should recognize both the old direct path and the new `resume/` path. Per-attempt `execution.json` stores the exact output argument and selection report.

All-invalid recovery intentionally needs investigation. Repair the source or explicitly authorize a new experiment; do not bypass the failure by removing history.

## Remaining shutdown cost — not changed

The native solver still performs a final LP resolve after receiving a termination signal. This fix addresses checkpoint selection/copying only.

Full telemetry from the six existing k=32 runs gives the following final-resolve costs. Every one of these runs stopped at its ordinary application wall limit; **none received a recorded termination signal**.

| Chain | Final incidence construction (s) | Final LP attempt (s) | Combined (s) | LP outcome |
|---|---:|---:|---:|---|
| 1 | 4.142 | 20.108 | 24.249 | Time limit |
| 2 | 4.774 | 20.122 | 24.896 | Time limit |
| 3 | 3.002 | 8.748 | 11.750 | Solved |
| 4 | 4.072 | 6.678 | 10.751 | Solved |
| 5 | 3.339 | 7.358 | 10.697 | Solved |
| 6 | 3.261 | 3.368 | 6.628 | Solved |

These costs exclude subsequent serialization/publication and do not bound shutdown latency during a large in-flight pricing/LP call. The legacy telemetry field labels final attempts `highs-ds`, but their configured master backend is Gurobi; the label comes from the final-resolve loop, not a backend switch.

Separate instrumented **actual SIGTERM and SIGUSR1** tests on four trips measured 0.00057 and 0.00050 seconds for the post-signal master call, and about 0.208 seconds from signal to process exit. They verify that the extra solve occurs, not that large-run shutdown fits a scheduler grace period. `signal_timing.json` and the corresponding artifact archive retain these measurements. The source and hashes for all six full telemetry files are recorded in `k32_final_resolve.json`.

Other unchanged limits: no LP-basis or in-flight DP persistence; only recorded application time is restored, so lost in-flight time requires external accounting; unpublished graph builds restart; MIP trees restart. Validation has a 600-second per-candidate timeout, logged as rejection on expiration. The copied pool is parsed before CG starts, so it does not coexist with a running solver in this wrapper. This package makes no new memory-sizing claim for full40.
