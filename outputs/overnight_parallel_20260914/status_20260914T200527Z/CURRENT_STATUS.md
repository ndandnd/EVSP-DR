# Overnight priorities and current evidence — 14 September, 16:13 EDT

**99 jobs are already running.** The pending jobs need real inputs; no broken dependency was found. The queue is now substantially parallel. [Current queue table and every dependency](../queue_20260914T201351Z/FINDINGS.md).

| Priority | Already submitted | What tomorrow's results should tell us |
|---|---|---|
| Explain the benefit of inherited columns | 36 small-seed CGs plus their MIPs; 28 CG certificates and 7 completed MIPs in the 16:09 collection | Whether a small selected subset can preserve good integer solutions without replaying the whole pool. |
| Compare fair computation budgets | Completed 24-case cumulative-budget comparison; 3 additional near-four-hour fresh-pool MIPs running | Whether warm-start improvements persist after counting earlier computation. |
| Test decomposition | Two route-selection treatments, 46 MIPs each; 40 still running at 16:13 | Whether routes from different 4×8 decompositions can improve the 32-duty solution. |
| Extend the scaling boundary | Six chains through k=28; all 18 new graphs running | How far the current baseline can go. |
| Resolve realistic charging bottlenecks | Five one-bus capacity diagnostics still running | Which duties make exact pricing difficult and whether cached pricing helps. |

The completed accumulated-time comparison has 24/24 fresh CG pricing certificates, but only 6/24 fresh integer target matches versus 24/24 warm references. Four fresh pools are proved to require extra buses; fourteen searches have open fleet gaps. Thus extra CG time alone does not guarantee a useful integer pool. Historical code revisions and hardware varied, so this retrospective comparison does not isolate one algorithmic change. [Full budgets, outcomes and caveats](../../cumulative_budget_20260913/status_20260914T200527Z/README.md).

The small-seed comparison currently has 28 certificates and seven completed MIPs, six matching target. New C2 k8 integer seeds recover eight buses, proved within the pool. C5 k8 remains the instructive contrast: eight with integer-selected seeds versus nine with LP-weight seeds, both pool-proved at the same certified weighted LP objective. Their selected sequences cover different numbers of trips; this is not an isolated test of integrality. [Exact seed table](README.md).

New original C1 k23 uses 23 buses, with a fleet proof in its saved pool and individual-route replay. Its CG stopped at the time limit without a pricing certificate. Across original extension runs: 56 CG endpoints (45 certificates, 11 caps), 55 one-hour MIPs (32 targets, 23 misses). Separate longer searches recover 15 misses, leaving eight unmatched targets. Largest original matches by chain are 23, 22, 24, 23, 23, 24; the separate longer C2 k25 result remains a 25-bus match.

The first decomposition treatment has 45 verified results, fleets 34–37, all pool-proved and replayed, with no pair improving its best contributing partition. The second has seven published individual controls: p01/p02/p04/p05/p09=35, p03=36, p07=34, all pool-proved and replayed. Pair results are pending in this scientific collection. Target 32 is not a global lower bound; service overlap gives an independent lower bound of 29. These treatments share one 750-trip input and are not independent datasets. Both omit shared charging capacity and terminal SOC floors.

One default-partition MIP, C3 k8 integer seeds, was preempted after 2h4m11s without a final result and automatically restarted after three minutes. Reporting now binds the two attempts to their separate result paths; the interrupted attempt is not overwritten. [Correction and verification](../../research_register/preemption_study/attempt_binding_fix_20260914/README.md).

No new campaign was submitted in this check. All listed overnight work was already authorized and submitted; no duplicate jobs, held historical jobs, V2G experiments or Slides were changed. The Google Doc's current status and editable queue table were updated; reference material and both figure tabs were preserved.
