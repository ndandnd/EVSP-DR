# 11:10 EDT briefing evidence

P4 k9 CG completed in310.53 minutes with conservative expanded-grid pricing certificate; weighted objective900360.376544336 and fractional route weight approximately9. MIP810993 is running; k10 CG810344 started. No additional completed integer result since the10:37 snapshot.

Default MIPs:75 completed,0 confirmed preemptions, all75 with selected-route replay flag. Eligible-to-start wait median44s/max120s; these exclude time waiting for source dependencies. Allocation exposure171259s (47.57 job-hours);44 solves near the full hour. One night of correlated cases cannot establish a general preemption probability. Six Scaglione nodes:compute-01 reserved for GPU users, five56-CPU nodes usable. At live inspection four CPU nodes idle and one mixed; pending MIPs were dependency-gated.

Register and workbook refreshed to1513 records/31source groups. Google Doc current status and register timestamps updated in place to11:10; exact old-text matches disappeared. Slides unchanged.

## Proposed decomposition experiment, not launched

Start with matched k10 cases, dividing duties into two k5 groups under unchanged physics/objectives. Generate columns independently and recombine them with original input IDs in a global covering MIP including shared station-time capacities. Preserve a feasible combined incumbent only after global validation. Add overlapping boundary subproblems and cross-group pricing to recover opportunities omitted by a hard partition. Retain old feasible incumbents in every iteration.

Compare fresh global CG, inherited columns, disjoint-group pool, and disjoint-plus-overlapping-group pool. Report fleet, charging cost, capacity/coverage replay, queue-inclusive wall time, CPU-hours, and certified lower-bound scope. Equalize both per-solver settings and total compute when measuring algorithmic advantage; also report parallel elapsed time as a separate operational advantage. Start from small cases with known global evidence, then scale to20 and40 only after measuring partition loss.

Existing geography pilot:14duties/203trips inRoute21 and26duties/745trips inPartille; both sharePARX. Optimistic30-minute-wait graph has7561cross-group edges/23959total andoneconnected component. These are not exact energy-feasible transitions. No exact geographic decomposition is established. Pricing only within clusters cannot certify the unrestricted model.

Related primary research: ten Bosch et al., Scheduling electric vehicles by simulated annealing with recombination through ILP (2026), https://link.springer.com/article/10.1007/s12469-026-00424-2 . Route-pool recombination is established; our potential contribution must concern tested EVSP–DR charging/capacity behavior or a demonstrable algorithmic improvement.
