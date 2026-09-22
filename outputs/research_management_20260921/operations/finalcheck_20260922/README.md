# Scoped final operations check — 22 September 2026, 03:43 UTC

SSH is healthy. At03:43:11UTC,42 jobs run:41 baseline graph tasks in661616 and strict k19 CG668433. The queue shows119 pending display rows, representing151 tasks after expanding the33-task historical held array537227. No utilization recovery is needed. Historical holds, running pins and all dependencies remain unchanged. This is a scoped queue/k15 check, not a new graph progress audit.

All six approved k15 continuation allocations are terminal. The existing read-only collector was run against `/home/nc437/ladder-lite/integer_columns_k15_20260921`; outputs, full final MIP logs, final results and small failure receipts are saved here.

| Case | Control | Treatment | Interpretation |
|---|---|---|---|
| C1 |704510 COMPLETED;19/bound15|704511 FAILED1:0; no final MIP|RMP solver timeout treated as fatal|
| C3 |704512 COMPLETED;17/bound15|704513 COMPLETED;**15/bound15 proved**|Own-dive15-route start accepted;22,693new columns|
| C5 |704514 COMPLETED;16/bound15|704515 FAILED1:0; no final MIP|RMP numerical row-residual check treated as fatal|

The C3 treatment's [full log](k15_artifacts/results/c3_k15/treatment_s20260921/704513_r0/mip_gurobi.log) accepts the15-route start at line30 and proves fleet15 at line63. Charging subsequently times out:569.992 incumbent,507.6694256 bound,10.9339%gap at line337. Route replay passes; the result reports12overcovered trips, with duplicate removal and shared charger capacity both unvalidated. This is a finite-pool fleet proof under the historical model. No global pricing or dispatch certificate follows. All control fleet gaps remain open. The treatment failures are not completed scientific misses; do not calculate a completed3-pair hit rate.

## Exact failures and next action

- **C1:** final RMP was given1.83155465s remaining. Gurobi returned TIME_LIMIT(status9); `DiveMaster.solve` raised `DivePilotError('dive restricted master did not reach OPTIMAL: status=9')`. Dive ended after30nodes,17,859new records and5,346.53799s internal wall; charged subprocess wall5,347.86175s. The wrapper then refused to launch the final MIP. This is an expected deadline handled as an exception, not OOM or infeasibility.
- **C5:** RMP returned OPTIMAL, but the independent row check found violation5.140335931130835e−6 and raised `DivePilotError('dive master returned a row-infeasible LP: violation=5.140335931130835e-06')`. Dive ended after23nodes,19,943new records (19,937 distinct incidences),3,932.51841s internal wall; charged subprocess wall3,933.69927s. No final MIP ran. Do not silently relax feasibility acceptance.

Both failed attempts have published `dive/cg.json`, its appended journal and manifest. **Next authorized recovery preparation:** read and hash those published artifacts, verify original-prefix preservation and physical replay, then prepare separate final-pool salvage receipts using only the residual7,200s allowance (C1 floor1,852s; C5 floor3,266s), original seed/settings and explicit failed-attempt lineage. Preserve original failures; salvage is supplemental work, not an unchanged completed paired trial. Check current queue and the remote resource policy before any recovery submission; do not duplicate a later attempt.

For future runs, implement isolated, tested handling of an RMP time limit as an uncertified dive stop that safely publishes valid generated columns and reaches the final MIP. Investigate C5 numerical residual with a bounded re-solve/stronger numerical settings and the existing acceptance tolerance; never price from invalid duals or enlarge tolerance merely to pass. No code changes or resubmissions occurred in this check; the coordinating task retains the recovery decision.

MaxRSS for C1/C5 failed batches:6,505,748KiB /5,228,132KiB, against32G requests. C3 treatment:6,872,772KiB. These failures provide no memory-increase rationale. Completed charged times slightly exceed nominal7,200s through solver termination (C3 treatment7,201.17871s).

## Evidence

`queue_and_k15.txt` contains timestamped squeue/sacct; `evidence_latest.json/csv` and `scheduler_latest.txt` are the existing collector outputs. `failure_details.json` preserves exact error manifests and worker tails. `proof_lines.json` indexes the full copied Gurobi logs. `source_hashes.json` records every local source artifact hash. No Doc, Slides or register edit was performed by this check; continuation was updated only to make recovery ownership and next steps explicit.
