# MIP repeatability and three remaining chain gaps

**Launched 14 September, 04:42 EDT: all 30 jobs running, 48 EVSP–DR jobs running overall.** Jobs **186672–186701** have no dependencies. All native unrestricted-license checks passed. [Scheduler evidence](scheduler_verification.json), [startup checks](launch_status.json), [case/job map](case_jobs.json).

| Experiment | Jobs | Solver allowance | Purpose |
|---|---:|---|---|
| Repeat nine inherited-pool MIPs three times each | 27 | One hour total; up to 30 minutes for fleet search | Measure how reliably the original settings recover a target already known to exist in the saved pool |
| Longer MIPs on C1 k19, C3 k22 and C3 k23 | 3 | 3½ hours total; up to three hours for fleet search | Close the three remaining extension gaps selected from the 04:28 result snapshot |

Why repeat the shorter search? [All nine earlier longer reruns recovered their targets](../overnight_diagnostics_20260914/status_20260914T082509Z/LONGER_MIP_RESULTS.md), and seven proved the fleet within 30 minutes despite the original 30-minute searches missing. This demonstrates that those solutions were in the original pools. It does not establish that additional elapsed time alone caused recovery.

The repeats keep the original ordered columns, input hashes, solver commit, default seed, eight threads, greedy initializer and two-stage objective. They do not import the newly recovered incumbent. Report fleet found/proved by the first-stage deadline, time, solver work where available, host, CPU use and preemption separately. Three repetitions are a small operational pilot on nine deliberately selected pools, not new random datasets or a population success-rate estimate. The manifest's `independent_input_instances=12` means twelve distinct input sets; nested sets are not statistically independent.

Stage 1 minimizes buses. Stage 2 minimizes electricity plus charging-start cost with fleet no greater than the first-stage incumbent. These remain covering models with 240 kWh batteries, 240 kW charging, fee 5 and flat prices, without shared charger capacity or a terminal-SOC floor. Fleet proofs concern saved columns; individual-route replay and target attainment are separate claims.

All jobs request 8 CPUs and 24 GiB on `default_partition`, exclude `scaglione-compute-01`, and use private job/restart output directories. Existing previous-k dependencies, held historical jobs and V2G work are untouched. Frozen worker and solver code were reused unchanged. Gurobi starts a new tree after requeue.

[Manifest](manifest.json) records every input, source pool and execution hash, arguments and resource request. [Preflight](validation.json) checked all 30 cases. The collector now recognizes this campaign separately. The unchanged worker uses the existing preemption-study cohort label; distinguish this campaign by its recorded root and unique case IDs. No repeatability results are claimed at launch.

## Next useful comparisons

1. Combine original, 200-column and complementary pools for the same input, then solve the union. Thirteen inputs had both treatment pools complete at 04:28. This could test whether the independently generated routes complement one another. It needs a validated union/physical-replay adapter; it has not been launched.
2. Retain distinct station-time charging schedules in capacity-constrained pools before expanding stricter physics. The baseline trip-set deduplication is not sufficient evidence for shared-capacity inheritance.
3. Profile and checkpoint the large graph builder before restarting decomposition. The prior 750-trip build timed out before producing the graph; more downstream dependencies cannot remedy that.

Remote root: `/home/nc437/ladder-lite/mip_repeatability_20260914`; large outputs: `/share/scaglione/nc437/evsp-dr/mip_repeatability_20260914`.
