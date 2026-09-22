# Reading the September 21 figures

These notes explain the five-figure [results preview](../../research_management_20260921/paper_results/RESULTS_PREVIEW.md) at Git commit `e07c877f0adc23bb8d64b1db1c3d104b0ceefc19`. They do not add new solver results. Figure 1 needs no correction.

## Figure 2 Finding routes that work together as whole buses

The right panel compares the final number of buses for four selected k=8 cases. Lower is better; eight meets the target.

| Existing label | Meaning |
|---|---|
| Pool MIP | Gurobi selects whole bus routes from the saved list of CG columns. It cannot create a missing route. |
| Pool MIP, 1 hour | Search the unchanged saved list, using the usual two-stage MIP with a total one-hour allowance. |
| Setup allowance | Give the unchanged-pool MIP extra time equal to the historical graph-build time. The total allowances are 181.3, 114.9, 129.5 and 102.0 minutes for C1, C3, C4 and C5. The extra time is real optimizer time, not a pause for setup. |
| Original dive + final MIP | Tentatively commit to a promising route, re-solve the LP and price new routes that complement that choice. Repeat or backtrack; finally run MIP on the enlarged route list. “Original” identifies the first pilot, before the C1 follow-up. |
| Purple C1 later incumbent transfer | The C1 dive found eight buses, but its first final MIP was not given that complete solution and ended with nine. A later run explicitly supplied the dive's eight-bus solution as a MIP start and proved eight optimal in the resulting pool. |

The purple point is additional work: about 29.4 minutes before publication, beyond the original C1 pilot's 61.5 minutes. It is not a fourth success within the original one-hour treatment. The follow-up used the dive's own routes, not GIRO or sequential witness routes. The transfer helper introduced duplicate column variables, so “same distinct routes” is more accurate than “identical MIP matrix.”

The extra-budget comparison asks whether ordinary MIP search can compensate for route generation if it receives the time credited to building the pricing graph. There is one treatment run, compared with two control budgets; the treatment reused the graph rather than rebuilding it.

Sources: [pilot design and budgets](../../independent_review_20260916/advisor_diving_pilot_20260920/README.md), [final results and native log paths](../../research_management_20260921/paper_results/figure2_k8_pilot.csv), [C1 follow-up](../../week_20260921/evidence/c1_followup/628441_r0/result.json).

## Figure 3 Better feasible solutions are upper bounds

For minimization, finding a schedule with 17 buses proves that at most 17 are needed: an upper bound. A lower bound of 15 says fewer than 15 are impossible in that model. The graph's segments show the remaining gap between these quantities.

| Twelve-hour fleet search | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---:|---:|---:|---:|---:|---:|
| Plain Gurobi settings | 18 | 17 | 17 | 19 | 16 | 18 |
| More effort on finding feasible solutions | 18 | 17 | 16 | 17 | 16 | 17 |
| Saved-pool lower bound in both searches | 15 | 15 | 15 | 15 | 15 | 15 |

“Heuristic-focused” is still Gurobi MIP, with `MIPFocus=1`, `Heuristics=0.5` and `NoRelHeurTime=1800`. It spends more effort finding good integer solutions. Both configurations still run branch-and-bound and report lower bounds.

The setting improved three cases and tied three at seed zero. That makes it a useful candidate for finding feasible fleets, but does not establish a universal default, a benefit at a one-hour allowance, or better proof times. None reached 15. More heuristic effort can trade slower bound progress for better feasible solutions, as described in the [Gurobi parameter reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#parameterMIPFocus).

Source: [all twelve results and full logs](../../research_management_20260921/paper_results/figure3_k15_12h.csv).

## Figure 4 Importing previous routes more efficiently

| Existing label | What changes | Measured target-step CG time reduction |
|---|---|---:|
| Indexed route replay | When rechecking saved sequence A→B→C, jump directly to A→B transitions instead of scanning A→every possible next trip. Keep all relevant battery states. | 12.4–18.1% |
| Omit unused LP setup | Stop rebuilding a duplicate route-to-trip matrix that the incremental Gurobi master already maintains. The LP is still solved. | 9.3–14.6% |
| Full vs 512 | Import all eligible saved previous-k route sequences instead of selecting at most 512. Here the full imports contain 74,019, 40,954 and 61,584 routes. | 40.6–64.5% |

512 is an earlier limit on the number of inherited routes, not a bus count, number of iterations, or mathematical requirement. The selection favored long routes, then low cost per trip. Retaining the whole pool preserves useful alternatives that truncation can discard.

The first two changes remove redundant work while preserving the compared algorithmic answers. Full inheritance changes the initial column set and subsequent search: final fleets changed 9→8, 11→10 and 17→15. The first two truncated pools proved their extra bus necessary; 17 in the third was only an incumbent with bound 15. All three full pools proved their attained fleets optimal within the pool.

These are three cases, each measured in two execution orders. The percentages are separate contrasts and must not be added. Times exclude earlier chain work, prior graph construction and the final MIP.

Caption correction: six full-scan/full-pool runs exhausted the **overall two-hour CG budget while still importing routes**. “Two-hour import limit” was imprecise: the separate import-specific timer was disabled. No plotted result changes.

Sources: [comparison definitions](../../controlled_comparison_20260913/README.md), [measured results](../../controlled_comparison_20260913/status_20260913T063519Z/README.md), [all allocations](../../research_management_20260921/paper_results/figure4_controlled_algorithms.csv).

## Figure 5 Representing and searching the pricing graph

The graph describes feasible transitions between trips and battery/time states. A shortest-path calculation searches it for a useful new route.

| Label | What it means | Build time | Peak RAM | Mean pricing call |
|---|---|---:|---:|---:|
| Original explicit | Store each transition as Python objects and action dictionaries. Compute textual tie-breaking keys eagerly. | 121.1 s | 2.062 GiB | 33.015 s |
| Deferred explicit | Keep those objects, but construct expensive tie-breaking strings only when costs tie. | 52.3 s | 2.063 GiB | 32.358 s |
| Deferred packed | Also use compact numeric arrays, discard dominated transitions where valid, and search using vectorized array operations. Reconstruct full actions only for selected paths. | 41.8 s | 0.147 GiB | 0.0622 s |

“Deferred” means postponing unnecessary tie-breaking calculations. “Packed” means compact graph storage, combined here with fewer retained edges and a different implementation of the shortest-path scan. It is not packing more buses onto a route.

Yes, deferred packed wins these measurements. It returned the same minimum reduced costs on all five test queries, with all 15 route replays passing. This was one 26-trip case, not a complete CG speed comparison. The 530.8× pricing gain belongs to the combined changes; it cannot be attributed to compact storage alone. Shared charger-capacity duals were absent, and the current packed pruning is not valid as a drop-in implementation for that capacity-enabled model.

Sources: [benchmark CSV](../../research_management_20260921/paper_results/figure5_packed_benchmark.csv), [implementation and capacity scope](../../week_20260921/capacity_strict/README.md).
