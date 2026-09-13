# Controlled comparisons of the recent baseline changes

Prepared 13 September 2026 UTC (12 September evening in New York). The user requested new experiments to measure the changes separately. Submission status and exact IDs are recorded in `jobs.json`; a prepared manifest alone does not mean a job was launched.

**Launched 12 September 2026, 22:11 EDT: all 24 pairs running on default.** At the first verification, all 24 licenses passed, all arms had entered their CG process, three had completed positive CG iterations, and no execution error was recorded. The total EVSP–DR queue had 29 running jobs. These are startup observations, not completed comparison results.

[Exact job IDs and verified resource requests](jobs.json), [startup checks](startup_verification.json), [queue snapshot](queue_after_launch.json).

## Questions and comparisons

| Comparison | First setting | Second setting | What it measures |
|---|---|---|---|
| Faster inherited-route replay | Original arc scanning; 512 sequences | Indexed lookup; the same 512 sequences | Import time and complete CG runtime with the same inherited routes |
| Faster replay on the full pool | Original arc scanning; all sequences | Indexed lookup; the same full pool | Whether full inheritance finishes within the budget, and its speed when both finish |
| More inherited routes | Indexed replay; 512 sequences | Indexed replay; all available sequences | Integer fleet, charging quality, pool size and extra computation from richer initialization |
| Less LP setup work | Indexed full inheritance; construct incidence matrix | Same, but omit the matrix Gurobi does not use | LP-setup savings and their contribution to total runtime |

We selected **chain 1 k=8, chain 4 k=10 and chain 3 k=15** to cover different sizes and include the earlier pool-limitation example. Each comparison runs twice, reversing order. This is **24 independent cluster allocations, 48 CG runs and up to 48 dependent MIP solves**. These are three deliberately selected input datasets, not 24 independent random samples. The two repetitions reveal some hardware/order variability; they do not establish statistical significance across all GIRO subsets.

Each pair runs sequentially on one node, with fresh Python/solver processes. All 24 pairs are eligible together; there is no artificial throttle below the number of independent comparisons. Previous-k work does not need to run again: each case reads a frozen, completed parent pool. No arm inherits another comparison arm's outputs.

## Settings held fixed

- CG source **e091a4dba549510238507ef5e5367abea958bd30**; MIP source **871d057e1067411f09581e37d78f7c1ca43f68bb**. The shutdown repair is common to all arms; we do not deliberately restore a hanging worker.
- Set covering; 240-kWh batteries; 240-kW charging; event graph with 2.5-kWh SOC and 5-minute blocks; flat prices; zero reserve; no return-SOC floor or shared charger-capacity constraints.
- CG objective is 100,000 per route plus electricity plus 5 per charge start. Thirty columns per iteration, reduced-cost selection, tolerance 1e-4 and at most 50,000 iterations.
- Each arm receives **two hours for CG** and **one hour for the final MIP**. MIP stage 1 minimizes buses for at most 30 minutes. Stage 2 minimizes charging with fleet **at most** the validated first-stage incumbent, using the remaining hour budget.
- Eight CPUs and 96 GiB per allocation; Gurobi MIP uses eight threads. Python hash seed is zero. Both arms retain the same solver defaults. Shared-machine interference and parallel MIP timing remain possible sources of variability.
- Default partition, seven-hour allocation, reserved GPU node `scaglione-compute-01` excluded. Held historical work and V2G experiments are untouched. Automatic requeue is disabled because a partially completed pair must be preserved and any replacement assigned a fresh attempt path.

The import-specific time limit is **disabled in every arm**. Both 512-route arms therefore attempt all selected sequences. In this source version, disabling that limit also uses ordered worker results, preventing worker completion order from rearranging imported columns. This is a deliberate difference from the older 512-route/900-second campaign. The overall CG time limit still applies; an interrupted import is reported as incomplete.

## What we will report

1. Imported sequence counts, order/content hashes, rejected routes, import time and index preparation time where retained in source telemetry.
2. Time spent loading the existing graph, preparing the LP, solving the LP and pricing; CG iterations and time to the pricing certificate.
3. Weighted LP objective, fractional route weight, minimum reduced cost and exact reason for stopping. An uncertified RMP objective is not labelled a full-model lower bound.
4. MIP fleet, finite-pool fleet bound/proof, charging objective/gap, both stage times, and individual-route physical validation. Duplicate removal and shared-capacity validation are separate fields.
5. For implementation-only pairs, whether the inherited pool, CG trajectory and certified endpoints agree. If they differ, investigate before reporting a pure speedup.
6. For 512-versus-full pairs, whether extra columns improve integer quality even when certified LP objectives agree. Runtime and solution quality are separate outcomes.

Existing graph construction is excluded from the paired timings, while loading the graph and enabling the replay index remain inside each CG run. Input authentication occurs before the arms; its duration is retained separately. A capped run is censored and cannot be treated as a measured time to convergence. No graph-builder or capacity-pricing acceleration is being tested by this baseline campaign.

## Earlier comparisons already available

The [earlier paired measurements](existing_pair_summary.json) are retained rather than rerun as identical jobs. Two fresh cases isolating the omitted matrix showed 5.8% slower and essentially unchanged complete CG runtime. Four warm pairs combining replay and matrix changes showed 16.2–22.4% less CG runtime, with all certified endpoints agreeing within 1e-4; imported-route checking fell from roughly 366–447 seconds to 5.4–6.8 seconds. Different CG iteration counts and the combined treatment prevent attributing all gains to one change. Three capacity comparisons exhausted their three-hour limits in both arms without certification, so they do not establish a convergence speedup.

The direct full-pool index comparison may exhaust the old scanning method’s CG budget during initialization. Such runs remain censored, and their missing MIP results are labelled as skipped rather than infeasible.

These observations motivate the new separated comparisons; they are not universal speedup claims.

## Provenance and recovery

`manifest.json` binds inputs, parent status/journal, graph cache, reference data, tariff, model settings, execution commits, resources and arm order. Completed parent journals are hard-linked when possible to avoid unnecessary storage duplication. Their original and frozen paths must remain immutable. Protect the existing full-pool source checkout and borrowed Git objects, graph caches and parent CSVs while this campaign exists.

`deployment.json` binds the comparison worker/collector to its own Git commit and file hashes. It does not replace the CG/MIP execution commits. Workers use exclusive job/restart output directories and a Gurobi preflight with 3,001 variables before loading large inputs. Every arm records commands, process status, watchdog decisions and output hashes.

The general collector includes this campaign and the readable `~/ladder-lite/drq` queue groups it separately. Scheduler reliability is recorded as a **paired seven-hour allocation cohort**, separate from standalone one-hour MIPs; timestamps inside each arm identify whether any interruption occurred during CG, MIP or preparation. A completed allocation is not automatically a completed pair or a validated optimization result. Do not duplicate a live pair or overwrite a partial attempt.
