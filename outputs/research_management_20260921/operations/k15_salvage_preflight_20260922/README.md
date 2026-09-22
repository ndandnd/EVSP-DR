# Authorized supplemental recoveries submitted — 22 September 04:01 UTC

C1 salvage **728184** and C5 salvage **728185** were submitted after coordinator review, real C3 API-field verification, seven successful wrapper smoke/failure-closed tests and a fresh duplicate check under a submission lock. These are the only new allocations. Uploaded wrapper/plan hashes match local bytes. Both effective Slurm configurations are default_partition,8CPU,32G,90-minute allocation, reserved-node exclusion `scaglione-compute-01`, no dependencies and **Requeue=0**. Original scientific solver limits remain1,852/3,266s (fleet stages926/1,633s); no pricing restart, imported incumbent, node fixings or dual reuse.

At **04:02:20 UTC**, both jobs are **PENDING (Priority)**. Neither worker nor native physical replay gate has started; gate success is not claimed. Next monitor: read these two jobs' per-attempt receipts and physical_gate.json, require zero rejected routes, record accepted/repaired counts and ordered pool hash, then separately audit final fleet proof, physical replay and timing. If preempted, preserve the attempt and measure/debit consumed solver time before any new submission; do not reset the original7,200s allowance. Submission commands/configuration are in `submission.json`, initial state in `start_snapshot.json`, and seven test results in `wrapper_tests.txt`. The preparation notes below are historical and remain for provenance.

# k15 salvage preflight — 22 September 2026, 03:55–03:58 UTC

Prepared only; no jobs submitted, no solver run, no code or failed-attempt bytes changed. Reused the03:43 queue collection, then read the remote resource policy before this bounded preflight.

Both failed attempts preserve publishable generated pools. Eight source/augmented result+journal hashes match their recorded manifest hashes. Each augmented journal has a byte-identical complete original prefix, followed by exactly17,859(C1) /19,943(C5) complete JSONL records; all appended records have diving origin and finite generation reduced cost. Neither attempt exported an integer incumbent. Apart from its journal pointer and augmentation block, each CG status remains identical to its original source. The pinned remote checkout is clean at `c50e5f207869bac25507adfae90bb830a44039b7`.

At observation, the two case directories contain only original failed attempts704511_r0 and704515_r0; no salvage directory or live case/salvage-named allocation was found. A submission wrapper must recheck immediately before submission. A name scan and directory inventory are evidence against duplicates at this observation, not a permanent lock.

| Case | Failed charged dive wall | Remaining solver limit | Fleet-stage limit | Imported start |
|---|---:|---:|---:|---|
| C1 |5347.861750387121s|1852s|926s|None|
| C5 |3933.699269213015s|3266s|1633s|None|

The limits are floor(7200−charged dive subprocess wall), with no minimum extension. MIP setup/replay remains external measured overhead, as in the registered protocol. Any additional salvage attempt must debit earlier salvage solver runtime; a fresh requeue output path alone does not preserve cumulative timing fairness. Keep original failures and label this as supplemental salvage.

Exact pinned-worker argument vectors, environment hashes and resources are in `salvage_plan.json`; readable command templates are in `native_commands.md`. Preserve cover, two-stage fleet/charging objective, seed20260921,8threads, gap10⁻⁴ and source-derived240kWh/240kW, zero-reserve, flat-fee5 historical physics. Do not pass the dive fleet target as a hard final-MIP cap: the original final MIP searches fleet size freely. Use default_partition, exclude scaglione-compute-01 and retain32G (no failure suggests increasing memory). The two cases are independent; no dependency on afterok of the failed dive jobs should be introduced. Output paths must be new job/restart directories, with explicit failed-attempt lineage and a duplicate-safe submission lock.

## Native replay gate remains required

The diver validates every admitted record physically before publication. This preflight checks bytes/provenance, not a fresh full replay of the two large pools on the login node. `run_exact_pool_mip.py` natively executes `load_pool(...,deduplicate=True)` and `prepare_strict_partition_pool(...)` with source-input hash checks before optimization. A compute-worker preflight can call those same functions without creating a Gurobi model, persist its detailed audit and ordered-pool hash, and stop on rejected routes or unexplained repairs. Keep original costs and physical settings. The native runner can deterministically repair/reject columns, so do not equate a successful load alone with zero pool changes; inspect and record its audit. There is no justification to relax the C5 LP feasibility tolerance in this salvage step: no failed-node LP duals or LP solution are used.

On success, continue only with immutable source hashes and the original pinned runner. No external witness or own-dive start exists to import. Final selected-route replay, duplicate-service status, shared-capacity validation, finite-pool fleet proof and target attainment must remain separate. No full-model pricing or infeasibility claim is authorized by the failed dive manifests.

## Baseline and strict accounting clarification

Compact allocation accounting confirms all44 graph tasks:41RUNNING and3COMPLETED/0:0. Completed661616_38/_39/_40 are C6 k34/k35/k36 caches. Their CG jobs661727/661729/661731 correctly retain only unfinished predecessor dependencies661724/661727/661729; completed graph dependencies have cleared. No broken predecessor or manual graph recovery is indicated.

Strict668430(k17CG) and668432(k17MIP) are COMPLETED/0:0.668433(k19CG) is RUNNING with no unresolved dependency;668434(k19MIP) waits afterok:668433. The finished k17 MIP is a newly available endpoint for the next scoped scientific collection, not yet audited here. No broad historical collector was run.

Sources: `remote_audit.json`, `graph_strict_accounting.txt`, `current_dependencies.json`, and `salvage_plan.json`. The initial multi-ID scontrol syntax was rejected; the retained individual-ID queries succeeded for active jobs, while already-purged668432 is covered by sacct. This query correction had no cluster side effect.

## Prepared bounded worker

`run_salvage.py` implements the native zero-rejection gate and final pinned MIP invocation; `worker_launch.md` gives exact allocation arguments. It records accepted/repaired/rejected counts, clears the in-memory gate pool before launching the solver, verifies the final ordered pool hash and original immutable hashes, and preserves failures. It transfers no LP duals, artificial values, node fixings or incumbent. A per-case execution lock and refusal of any prior salvage attempt prevent concurrent or unaccounted repeat work. Use --no-requeue until remaining budget is explicitly recomputed after interruption. Root review and fresh submission-time duplicate checks remain required. Syntax and argument/budget checks pass; no native full replay or cluster execution has been claimed.
