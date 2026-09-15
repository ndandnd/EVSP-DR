# Research results — 15 September, 01:16 EDT

**Chain 6 now matches 27 buses.** The one-hour MIP proves that fleet within its 210,085-column pool in 9.1 minutes; total MIP time is60.2 minutes. Individual-route replay passes. CG stopped after239.9 minutes without convergence; charging optimality remains open. There are53 duplicate trip assignments without separate removal validation. Shared charger capacity and a minimum ending SOC are absent from this baseline.

Largest individual one-hour target matches across chains1–6 are now **25,26,27,23,24,27**. Separate longer MIPs raise chain4 to25. These maxima do not imply success at every smaller k in the original one-hour controls.

**The smaller core start recovers chain4 at k15.** Its CG converged in166.5 minutes. The final pool has64,741 columns; the MIP proves15 buses in3.5 minutes and proves its charging-related objective849.736 within that pool, finishing after133.8 total minutes. This is an objective in the discretized model including charging-start fees, not a pure electricity bill. Individual-route replay passes;24 duplicate assignments and shared capacity remain separate validation questions.

Compact starts now have36 certified CGs and31 published MIPs:30 target matches and one open gap. At k15, six of seven published MIPs match target; five results remain unpublished in this campaign collection. C3's core still has17 buses with a bound of15, so15 has not been ruled out. [Every paired result](../../overnight_evening_20260914/status_20260915T050815Z/README.md).

**One added-column pool provably still misses target.** C3 k8, starting from the earlier integer-route-seeded pool and adding donor routes with positive LP weight, finishes at9 buses with bound9. Its18,289-column pool cannot supply8 buses. Fleet proof took65.0 minutes; total solve139.4 minutes also proved its charging-related objective216.488. Individual-route replay passes. The matched zero-LP-weight addition run is unfinished; only the previously reported C5k8 pair is complete, so no general selector winner is established. Zero LP weight is not zero reduced cost. These constructed pools have no new CG certificate. [All35 planned outcomes, including pending cells](lp_addition_results.csv).

**Larger compact starts:** C2k20 with512 initial sequences newly certifies CG in171.4 minutes, at weighted LP objective2,000,806.3818165 and fractional route weight20. Its54,090 columns feed the already-running MIP. Three large-cohort CGs have certificates; no large-cohort MIP is published in this collection. [Current CG/MIP table](compact_large_results.csv).

New continuation CG endpoints C3k28 and C4k26 reach their four-hour limits, after239.9 and239.7 minutes. Their last minimum reduced costs are−0.0001716 and−0.0636014. Neither has a pricing certificate; their RMP objectives are not certified full-model lower bounds. Their existing dependent MIPs are running.

**Queue in this collection:**59 running /36 genuine dependency waits;33 held historical tasks excluded. No new execution failure, confirmed preemption or invalid dependency. The preemption study contains906 attempt records. Two results arrived after their campaign was read: C5k26 MIP and C4k15 expanded-start MIP. Their scheduler completions and hashes are retained separately for the next full collection, and are not counted as new verified targets here. No jobs were submitted, cancelled or requeued.

The reserve screen, completed unchanged-pool reruns and fixed-state pricing comparison are unchanged. [Previous findings](../status_20260915T040735Z/README.md).

Snapshot20260915T050815Z completed05:16:48UTC, SHA256 `5c34246ee11c96d3697aad6dfb11ad313a3fb0162d9b85b7947f352e7b5c020f`. Register/workbook:3,203 records across70 source groups; all six supplements retained. [Changed endpoints and source hashes](new_endpoints.csv) · [Pool-experiment source validation](pool_experiments_validation.json).
