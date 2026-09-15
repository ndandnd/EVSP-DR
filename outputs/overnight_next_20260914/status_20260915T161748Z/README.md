# Research update — 15 September, 12:27 EDT

**Chain 1 now matches target 27.** Its original one-hour MIP found 28 buses. A new, longer search on exactly the same 252,540-column pool found 27 and proved that this pool cannot use fewer. The fleet stage ended after 83.97 minutes; both stages together took 210.19 minutes. Individual routes pass replay. Charging optimality, duplicate removal and shared station capacity are separate and are not proved here.

| Longer search | Target | Buses found | Saved-pool fleet bound | Conclusion |
|---|---:|---:|---:|---|
| Chain 1 | 27 | 27 | 27 | Target matched; minimum fleet proved in this pool |
| Chain 5 | 27 | 29 | 26 | Target unresolved after 180.30 minutes of fleet search |

Including separately labelled longer searches, largest observed target matches by chain are **27, 28, 28, 27, 26, 28**. Original one-hour results remain **26, 28, 27, 27, 26, 28**. These are observed matches, not maximum solvable sizes. [Exact source and matched-pool checks](longer_gap_results.csv).

## Seven direct target tests remain inconclusive

All seven tests asked whether their combined saved-column pool could cover the trips with at most the target number of buses. Every run reached its one-hour solver limit without a feasible solution or an infeasibility proof. Thus none establishes that its pool lacks a target solution, and none supplies a new minimum-fleet or full-model bound. These tests were C1k15, C2k20/k25, C3k20, C4k25 and C5k20/k25. They are different pools from the full-inheritance chain runs. Do not repeat identical tests simply to fill the queue.

## Work continuing

27 allocations are running; 49 wait on genuine graph, previous-k or own-CG dependencies. No new failure, confirmed preemption or unsatisfiable dependency appeared. All 24 graph tasks for targets29–32 remain active; C1k28CG and C4/C5k28 longer MIPs continue. No jobs were submitted, cancelled or requeued during this check. Held historical and EVSPV2G jobs remain untouched. SSH works.

The registered ready backlog was reviewed: completed compact-pool, repair and target tests should not be duplicated. The fair zero-start-fee full-CG comparison still needs the validated terminal-energy pricing and column-retention changes described in ../../zero_fee_full_cg_design_20260915/README.md before production submission.

The current Google Doc received only three changes: evidence date, Chain 1's table value, and a short diagnosis explaining longer integer search versus unresolved tests. Figures, historical tabs and Slides were preserved. The experiment register now contains3348records in78groups; records are not independent experiments. Detailed scheduler/preemption records remain outside the short Doc.
