# Execution issues — 10 September 2026

This page distinguishes execution errors from optimization results. Read the register README for its current snapshot timestamp; original failures and later recoveries are retained separately.

| Campaign / jobs | Verified issue | What the evidence does and does not say |
|---|---|---|
| Capacity/speed CG array 772080 | Some shared-capacity cells reached the 90-minute Slurm wall limit. Earlier completed cells used about 0.43–3.55 GiB MaxRSS against a 24 GiB request. | The observed termination is TIMEOUT, not out of memory or mathematical infeasibility. Ten cells have completed CG and two-stage MIP artifacts in the current snapshot. Missing terminal CG output is not a pricing certificate. |
| Terminal-energy joint MIPs 778802, all three tasks | `KeyError: 'expanded_grid_terminal_soc_kwh'` during replay of saved columns, before optimization. | Older singleton records lack the newer terminal-energy metadata. Production Gurobi license preflight passed. This failure is unrelated to a solver fleet bound or charging optimum. |
| Terminal-energy fixed-duty frontiers 778801 | All three completed. | Their completion markers match the retained output hashes. The fixed-duty results remain valid; no frontier rerun is required for the metadata repair. |
| Warm-chain-3 k=10 job 772009 | Final publication collided with canceled duplicate 772031 after optimization. | The validated result was recovered without rerunning: fleet 10, pool fleet bound 10; charging objective 393.664, bound 337.0023334, gap 14.3934%. Scheduler FAILED does not invalidate these separately recovered and checked solver results. |

## Terminal-energy repair

**Recovery completed:** retry array **810459** completed all three tariff MIPs using immutable commit `5cdb8138c29faef9d5bf949175cb1e815a0b4220`, based on the original `2424369` execution. Original fixed-duty frontiers were reused with identical hashes. All three runs prove fleet five and charging-stage optimality within their replay-validated saved pools. Model charging objectives: 08:00 **261.5697635682**; 12:00 **332.0255245396**; 18:00 **217.5050887809**. The shared aggregate terminal-energy floor remains 280.7833253 kWh. These are finite-pool results, not a new full-model pricing certificate. See [recovery provenance](../parallel_research_20260911/README.md). The earlier local draft is historical, not the production retry source.

The bounded fix deterministically replays every saved route and carries the recomputed terminal energy and cost into the in-memory record. Where old metadata exists, it is checked against the replay. Missing energy is never replaced by a guessed SOC or zero.

The compatibility regression and real-pool smoke checks cover all three saved tariff pools: 3,464 records at peak08, 3,739 at peak12, and 3,328 at peak18, including 62 older singleton records per pool. The retry must preserve the original fixed-duty frontiers, saved CG pools, input hashes, physics and objective. Its immutable execution code must retain the original terminal-threshold dependencies. A successful smoke check is not a completed optimization result.

The original failed attempts remain in the register. Any repair submission and eventual solution require separate job IDs, code provenance and output validation. See [terminal-energy campaign](../post_meeting_20260910/terminal_energy/README.md) for the experimental model and original launch evidence.

## Evidence

- [Current collector snapshot](../post_meeting_20260910/monitor/20260910T225621Z.json)
- [Queue at register delivery](queue_at_register_delivery.json)
- [Cluster limits and memory observations](../post_meeting_20260910/capacity_speed/concurrency50_cluster_limits.json)
- [Verified warm-k10 recovery](../post_meeting_20260910/license_and_mip_recovery/STATUS.md)

Independent default-partition arrays now request concurrency 50. Increasing an array throttle does not increase an individual job's wall-time or memory allowance.

## Capacity timeout reruns

Six original cells saved no resumable pools: the driver kept columns in memory and published only at termination. Fresh matched reruns are now CG **811181**, dependent MIP **811182**, at `/home/nc437/ladder-lite/capacity_speed_pilot_20260910_timeout6_rerun_7d38ef`. The CG budget is extended to eight hours with nine-hour scheduler allocation. Code, inputs and arm physics remain fixed. A longer budget is an explicit experimental change; these are not resumed checkpoints. Original timeouts remain recorded.

## Fresh-covering downstream dependencies — 11 September, 04:31 EDT audit

Fresh CG array 810454 had 72 completed tasks and three running tasks, while 49 freeze tasks with completed predecessors still reported unfulfilled `aftercorr` dependencies. This was a scheduler gating issue, not a CG or MIP failure. The existing pending freeze and MIP jobs were repaired in place after checking per-case prerequisites; no scientific setting or solver budget changed. The repair audit preserves all passes, including an initial source-path validation error before freeze release and an intermediate array-wide update that was corrected with explicit `array_task` identifiers. Slurm can interpret the numeric array parent ID as the entire array: use the composite task identifier for every task-specific update and verify the resulting prerequisite.

Completed CG prerequisites were discharged after successful accounting exit and final-status/journal checks. Unfinished CG tasks retain individual `afterok` dependencies; every MIP retains its own freeze prerequisite. See [repair and verification](../parallel_research_20260911/results_20260911T0831Z/README.md). The hourly collector retains this audit under campaign workflow evidence.

## Nine-hour capacity rerun outcome — 11 September 08:36 EDT

CG811181 tasks5/9/11/13/15 reached TIMEOUT after9h02m, without final pool/status. Task7 completed after8h34m and its MIP completed: duty13406 with documented opportunity capacity and60kW PARX yields one bus, cost56.952, both stages optimal within a35-column pool; all14 trips covered once and station-capacity audit passes. CG itself is incomplete (`cg_wall_limit`), with no pricing certificate. This is not a preemption or a proof of model infeasibility.

The nested retry result schema was added to the collector/register; the successful retry is now visible. A Sol high subagent is implementing and testing bounded pricing deadlines and safe checkpointing; do not repeat the five timed-out jobs unchanged. See [evidence](../parallel_research_20260911/results_20260911T1236Z/README.md).

Recovery implementation passed23 tests locally and on Unicorn. Fixed CG commit `253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6` adds cooperative pricing deadlines and atomic, identity-bound pool checkpoints. Five fresh retry tasks are array872397; individually dependent MIPs872398–872402 retain9bf3f75. Same8hCG/9hallocation and25minMIP/30minallocation. Successful old task7 is not repeated. These are launches, not recovered results. The collector includes `capacity_deadline5_retry`; exact job/resource/provenance records are in the linked retry evidence.

## Warm chain 1, k=7 — initialization timeout, job 810293

Verified 11 September at 20:14 UTC: Slurm TIMEOUT after 08:17:13 against 08:15:00 allocation. The event network loaded in 7.03 seconds (15,642 nodes, 74,597,352 arcs). The persisted status remains initializing, with zero iterations, no final LP, no pricing certificate and an empty column journal. The 7.66-second status wall time is the initial publication timestamp, not the completed runtime. The inherited-event-pool audit is null. Evidence places the timeout during initialization before the first recorded CG iteration; it does not identify a particular inherited route or establish infeasibility.

No usable child pool was saved, so its MIP cannot run and chain-1 k8–10 remain blocked by their true dependencies. The predecessor k6 pool and cached network remain available; this does not resume the lost child initialization work. No blind rerun was submitted. Recovery needs bounded, checkpointed initialization or a measured justification for a larger budget, tested against the same inputs and physics. Other chains and capacity retries remain active.

Evidence: outputs/parallel_research_20260911/warm_p1_k7_timeout_810293/evidence.json; collection monitor/20260911T201444Z.json.
