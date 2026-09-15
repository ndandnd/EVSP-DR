**Latest verified results, 20:12 EDT:** [26-bus match and 14 completed new compact-seed MIPs](status_20260915T000359Z/CURRENT_STATUS.md). The earlier five-result preview below is retained with its timestamp.

**First results, 19:35 EDT:** [Five completed new MIPs all match their targets](FIRST_RESULTS.md). This is a completion-selected subset, not the full experiment.

# Overnight work — evening of 14 September

The queue thinned because independent experiments finished while the larger chains still require their preceding k and prepared graph. Those data dependencies cannot be removed without changing the experiment.

| New experiment | Independent jobs | Following jobs | Budget | Decision it supports |
|---|---:|---:|---|---|
| Compact inherited pools, six chains at k=8, 10, 15 | 36 CGs | 36 MIPs, each after its own CG | 4 h CG; 3.5 h MIP including up to 3 h fleet search | Can a compact selection replace full-pool inheritance? |
| Nine remaining original chain gaps | 9 MIPs | 0 | 3.5 h total, up to 3 h fleet search | Can the unchanged saved pool reach its target with more integer search? |
| Fixed-dual capacity pricing | 4 pricing calls | 0 | Up to 4 h per call | At the same starting state, is slow pricing specific to the cached selector? |

All three experiments are submitted: 49 independent research jobs plus 36 MIPs that wait only for their own CG. All 36 CGs and all nine longer MIPs were observed running together. Two capacity-prefix attempts failed at a wrapper import before pricing could proceed; the two reference jobs continued. A native check exercised the corrected wrapper with a nonzero capacity dual, and two replacement attempts were submitted. The failed attempts are preserved; the pricing algorithm is unchanged.

The compact seed keeps the previous integer solution and every route with positive weight in the final LP solution, representing 39–234 distinct trip sets across the 18 inputs. The other arm retains exactly that core and fills to 512 distinct trip sets from the same saved pool. Both use singleton routes for the newly added trips. Here, 512 is the total number of inherited trip sets, not a time limit or an iteration count. Both arms use the same model, solver revision, input, cache and budgets: covering, 240 kWh / 240 kW, start fee 5, without shared charging capacity or a terminal-SOC floor. Original integer-only runs are useful context; the new pair is the direct comparison. Prior CG/MIP computation remains recorded and is not free.

The nine longer searches are C1 k20/k22, C2 k23/k24, C3 k25, C4 k21/k24/k25 and C6 k25. They all have open original fleet gaps. Preparation checked that no identical longer treatment was already running/completed and no target had already been recovered. These MIPs generate no new columns. An improved result establishes that the saved pool contains a better solution; a failed search does not prove that it does not.

Both MIP campaigns minimize fleet first, then constrain fleet to be no greater than the incumbent and optimize charging-related cost with the remaining time. These are deliberately longer diagnostic MIPs, distinct from the original one-hour controls.

The pricing-call comparisons use k1 duty 13407 and the k2 capacity case. Each selector starts from the same verified saved pool and the same LP dual vector. They test one call, not a complete CG run. A time limit is censored timing evidence, not a convergence certificate. No MIP is needed for this question.

All new work uses the default partition and excludes scaglione-compute-01. There is no additional concurrency throttle; Slurm controls admission. Existing previous-k and own-CG dependencies remain. Held historical jobs and V2G are untouched. Unique attempts preserve preempted work and restart accounting.

**Queue check at 2026-09-14T23:52:32 UTC:** 55 EVSP–DR jobs running and 55 pending. No unsatisfiable dependency or use of the reserved node was observed. [Exact queue](queue_final.json). Pending jobs mostly need their own CG, the preceding chain step, or a completed graph; those are required inputs.

## Source records

- [Compact-pool campaign and exact core sizes](../compact_seed_support_20260914/README.md): CGs 201438–201473; dependent MIPs 201474–201509.
- [Nine remaining gap searches](../remaining_chain_gaps_20260914/README.md): MIPs 201512–201520.
- [Fixed-state capacity pricing](../capacity_fixed_dual_20260914/README.md): reference jobs 202362/202364; failed prefix attempts 202361/202363. [Corrected prefix attempts](../capacity_fixed_dual_retry_20260914/README.md): 202450/202451; native gate 202418.
- [Latest scientific results collected at 19:04 EDT](../overnight_parallel_20260914/status_20260914T225805Z/CURRENT_STATUS.md).
- [Completed equal-time fresh versus warm comparison](../cumulative_budget_20260913/status_20260914T225805Z/README.md).

The result workbook remains explicitly dated to the 18:58–19:04 collection. Later launch records are not new scientific endpoints. The hourly collector and normalizer now recognize the new campaigns; the single-pricing-call adapter records its own proof scope. Figures and Slides are unchanged.
