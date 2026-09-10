# Execution issues — 10 September 2026

This page distinguishes execution errors from optimization results. The normalized result register uses the 18:56 EDT snapshot; detailed recovery evidence may have a later timestamp.

| Campaign / jobs | Verified issue | What the evidence does and does not say |
|---|---|---|
| Capacity/speed CG array 772080 | Some shared-capacity cells reached the 90-minute Slurm wall limit. Earlier completed cells used about 0.43–3.55 GiB MaxRSS against a 24 GiB request. | The observed termination is TIMEOUT, not out of memory or mathematical infeasibility. Ten cells have completed CG and two-stage MIP artifacts in the current snapshot. Missing terminal CG output is not a pricing certificate. |
| Terminal-energy joint MIPs 778802, all three tasks | `KeyError: 'expanded_grid_terminal_soc_kwh'` during replay of saved columns, before optimization. | Older singleton records lack the newer terminal-energy metadata. Production Gurobi license preflight passed. This failure is unrelated to a solver fleet bound or charging optimum. |
| Terminal-energy fixed-duty frontiers 778801 | All three completed. | Their completion markers match the retained output hashes. The fixed-duty results remain valid; no frontier rerun is required for the metadata repair. |
| Warm-chain-3 k=10 job 772009 | Final publication collided with canceled duplicate 772031 after optimization. | The validated result was recovered without rerunning: fleet 10, pool fleet bound 10; charging objective 393.664, bound 337.0023334, gap 14.3934%. Scheduler FAILED does not invalidate these separately recovered and checked solver results. |

## Terminal-energy repair

The bounded fix deterministically replays every saved route and carries the recomputed terminal energy and cost into the in-memory record. Where old metadata exists, it is checked against the replay. Missing energy is never replaced by a guessed SOC or zero.

The compatibility regression and real-pool smoke checks cover all three saved tariff pools: 3,464 records at peak08, 3,739 at peak12, and 3,328 at peak18, including 62 older singleton records per pool. The retry must preserve the original fixed-duty frontiers, saved CG pools, input hashes, physics and objective. Its immutable execution code must retain the original terminal-threshold dependencies. A successful smoke check is not a completed optimization result.

The original failed attempts remain in the register. Any repair submission and eventual solution require separate job IDs, code provenance and output validation. See [terminal-energy campaign](../post_meeting_20260910/terminal_energy/README.md) for the experimental model and original launch evidence.

## Evidence

- [Current collector snapshot](../post_meeting_20260910/monitor/20260910T225621Z.json)
- [Queue at register delivery](queue_at_register_delivery.json)
- [Cluster limits and memory observations](../post_meeting_20260910/capacity_speed/concurrency50_cluster_limits.json)
- [Verified warm-k10 recovery](../post_meeting_20260910/license_and_mip_recovery/STATUS.md)

Independent default-partition arrays now request concurrency 50. Increasing an array throttle does not increase an individual job's wall-time or memory allowance.
