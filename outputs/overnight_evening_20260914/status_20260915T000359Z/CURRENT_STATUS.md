# Current research — 14 September, 20:12 EDT

**Chain 3 now matches k=26 with 26 buses.** Its CG certified after 113.2 minutes. The first MIP stage proved 26 optimal within the 148,580-column pool after 28.1 minutes. Total MIP time was 60.1 minutes; the charging-cost stage hit its limit, so charging optimality remains open. The canonical publication and immutable attempt contain identical JSON data, despite different whitespace and file hashes. [Verified publication binding](k26_publication_binding.json).

The selected covering routes pass individual-route replay, but include 50 duplicated trip assignments. Duplicate removal was not separately replayed. Shared charger capacity and a terminal-SOC floor are absent from this baseline. The fleet proof is a saved-pool proof, not a branch-and-price claim.

| Chain | Largest target matched by a one-hour MIP | Buses | CG minutes at that target |
|---|---:|---:|---:|
| 1 | 24 | 24 | 239.6 |
| 2 | 22 | 22 | 86.3 |
| 3 | 26 | 26 | 113.2 |
| 4 | 23 | 23 | 219.6 |
| 5 | 24 | 24 | 239.6 |
| 6 | 24 | 24 | 171.6 |

This combines the original k=16–25 campaign and its separate k=26–28 continuation. It does not imply every smaller target is recovered. A separate longer C2 k25 MIP matches 25. Original controls: 58 MIPs, 34 target matches and 24 misses; earlier longer searches recover 15 misses, leaving nine unresolved. [Every original case and stopping reason](../../cumulative_budget_20260913/status_20260915T000359Z/README.md).

C1 k24 matches target even though its CG stopped without a certificate. The original campaign now has 59 CG endpoints: 45 certified, 14 capped. C5 k25 newly hits the 239.4-minute limit, with minimum reduced cost −0.146677. In the separate continuation, C6 k26 now certifies after 119.9 minutes; its MIP is not published in this snapshot.

| Experiment | Verified endpoints in this collection | Interpretation |
|---|---|---|
| Core versus 512 inherited routes | 20 CG certificates; 14 MIPs, all target matches and pool fleet proofs | Promising early k8/k10 results; all k15 results still pending |
| Earlier small integer/LP seed sets | 36 CG endpoints; 29 MIPs | 8 target matches, 13 pool-proved misses, 8 open gaps |
| Nine remaining large-chain gaps | No MIP endpoint yet | All nine searches running |
| Fixed-state capacity pricing | No diagnostic endpoint yet | Four calls running; no convergence claim |

[Editable compact-seed result table, CG minutes and hashes](README.md). Fourteen earlier small-seed pairs are complete: integer-route seeds produce fewer buses in seven and tie in seven. New paired outcomes are C1 k10: 11 versus 12, and C5 k15: 15 versus 16. Their LP-seeded gaps remain open. C6 k15 LP seeds give 18 with bound 15. Its integer counterpart completed during collection; it is retained as a later scheduler observation and awaits the next detailed publication collection. [Earlier seed results](../../overnight_parallel_20260914/status_20260915T000359Z/README.md).

**Queue: 49 running, 44 solver dependencies and two conditional graph checks.** The 33 held historical tasks are excluded. No unsatisfiable dependency, new confirmed preemption or MIP execution failure appeared. The previously reported capacity-wrapper failures are preserved; both corrected attempts and both references are actively pricing. No new submissions, retries, cancellations or solver changes were made by this check.

The collection ran from 20:03:59 to 20:12:22 EDT (502.8 seconds). Snapshot SHA-256: `8050a0e803145c7cb184bd2fa1b25694721e112179ab0fc5d65f878a3d59ec3f`. Register/workbook: 3,051 artifact-stage records across 67 source groups, with all six supplements retained. Source checks cover 318 core endpoints plus 102 evening/seed/continuation endpoints. All 18 checked workbook views render; the formula scan found no errors. These record counts are not independent experiment counts.
