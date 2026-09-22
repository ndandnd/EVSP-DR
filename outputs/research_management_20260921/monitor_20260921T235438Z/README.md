# Research check — 21 September, 19:54–20:00 EDT

## Integer-directed pricing: complete paired k8 replication

All sixteen allocations completed. On four cases with two solver seeds each, the treatment recovers eight buses in **7/8 runs**, versus **0/8 controls**. Controls all return nine; five independently prove nine in their finite pools, three retain bound eight. Every successful treatment proves eight in its augmented finite pool. This is repeated evidence on four selected cases, not eight independent instances or a population success estimate.

| Case | Control fleets, seeds 21 / 22 | Treatment fleets, seeds 21 / 22 |
|---|---:|---:|
| C1 k8 | 9 / 9 | 8 / 8 |
| C3 k8 | 9 / 9 | 8 / 8 |
| C4 k8 | 9 / 9 | 9 / 8 |
| C5 k8 | 9 / 9 | 8 / 8 |

The seven successful treatments use only their own generated dive incumbents, accepted as MIP starts without added/replaced handoff columns. Their actual end-to-end times are 14.1–35.4 minutes. C4 seed 20260921 stopped its dive at the wall limit without an incumbent, then returned 9 with bound 8; it is an open gap, not an impossibility proof. All selected routes pass individual physical replay. One successful treatment validates exactly-once coverage. The other six contain 1–7 extra trip assignments; shared station capacity remains unvalidated for all arms.

The registered allowance is 3600 seconds of measured dive wall plus MIP solver time, with measured setup/replay overhead outside it. Nine time-limited arms overshoot the nominal shared allowance by 0.68–4.70 seconds; no hard end-to-end one-hour claim is made. No external GIRO or sequential witness columns were used. Full source, handoff, budget and log audits are in [integer_audit](integer_audit/README.md).

## Recovered approved continuation

The automatic k15 gate failed at its Slurm accounting query before scientific validation. The unchanged idempotent command succeeds from the login node, with all 16 predecessors complete and seven validated native handoffs. Six already-approved C1/C3/C5 control/treatment jobs **704510–704515** are now verified running. They retain the registered 7200 s shared allowance, max 5400 s dive, 8 CPU / 32G / 3 h / default / requeue and reserved-node exclusion. [Submission and failure records](k15_recovery/README.md).

## Strict implementation benchmarks and baseline scaling

The new 53-trip and 90-trip same-node comparisons preserve all five tested reduced costs and pass replay for all 15 generated routes per case. Packed graph construction is 2.84×/2.58× faster and per-process peak memory 29.35×/40.69× lower than original explicit storage. These are fixed-dual implementation benchmarks, not end-to-end CG or charging-capacity results. [Exact table and source checks](../operations/heartbeat_20260921T235509Z/README.md).

At 23:55 UTC, all 44 baseline k33–40 graph builds and strict k17 CG were running, with 99 true successor dependencies. No new baseline preemption or unhandled failure occurred. Source-row progress is not a reliable wall-time ETA. Historical held jobs and the stochastic project were untouched.

A bounded endpoint check at00:01:48UTC then verified strict k17 CG completion:277 trips from10 reference duties,5,236iterations,8,343columns and fractional route weight10. The four-hour deadline stopped CG without a pricing certificate; weighted RMP1,000,391.890434 is not a certified full-model lower bound. Graph building73.62minutes is included. Its MIP has no verified endpoint yet. [Source result and qualified completion](../operations/heartbeat_20260921T235509Z/k17_endpoint_summary.json).

## Next triggers

- Collect the six k15 endpoints and retain control/treatment pair and seed labels; do not duplicate the jobs or relabel a miss as a pool impossibility proof.
- Collect strict k17/k19 CG and MIP separately; retain subgroup duty count versus global prefix labels, capacity omission and duplicate checks.
- Let all 44 large graphs continue; recover actual failed predecessors and preserve genuine chain dependencies. Graph checkpoint deployment still requires its bounded native storage/durability pilot; current pins remain unchanged.
- No additional campaign is justified merely to inflate job count. Continue four-hour quiet monitoring, with immediate access-loss and actionable-failure reporting.
