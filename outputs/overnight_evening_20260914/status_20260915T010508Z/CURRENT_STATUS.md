# Current research — 14 September, 21:13 EDT

**The richer starts now match all k=8/k=10 targets: 24 of 24 MIPs across the two methods and six chains.** All have fleet proofs within their saved pools and pass individual-route replay. The first completed k=15 result, C6 with 512 inherited routes, also matches 15. Total: 30 certified CGs and 25 target-matching MIPs; eleven k=15 MIPs remain unpublished. [Full table, CG minutes and source hashes](README.md).

The core keeps the previous integer solution and every positive-weight LP route; the other method fills that core to 512 distinct trip sets. The experiment changes the inherited routes. These are the same twelve k=8/k=10 inputs under checked matching model fields, data hashes and CG revision.

**A precise diagnosis for thirteen earlier misses:** those older pools proved they could not support the target fleet. Both richer starts now reach the target on the same inputs, while the certified weighted LP objectives agree within 0.000002 (maximum observed difference: 0.000001939945). More MIP time on those old pools could not have fixed their missing integer combinations. CG convergence alone did not generate an adequate integer pool in those cases. Other old open MIP gaps remain unresolved; this does not attribute every miss to missing columns. [Matched-input, objective and pool-proof audit](pool_limitation_evidence.json).

The earlier small-seed comparison is complete: 36 MIPs give nine target matches, thirteen proved pool limits above target and fourteen open gaps. Across all eighteen pairs, integer-route seeds use fewer buses in eleven and tie in seven; they match eight targets versus one for LP-selected seeds.

| k=15 chain | Buses after integer-route seeds | Buses after LP-selected seeds | Pool fleet bounds |
|---|---:|---:|---|
| 1 | 16 | 19 | 15 / 15 |
| 2 | 15 | 18 | 15 / 15 |
| 3 | 15 | 17 | 15 / 15 |
| 4 | 16 | 18 | 15 / 15 |
| 5 | 15 | 16 | 15 / 15 |
| 6 | 16 | 18 | 15 / 15 |

These are final integer results, not LP route weights. Fleet is proved in a pool when its incumbent equals the pool bound. Thirty-five of these CGs certified; C1 k15 with LP-selected seeds reached its time limit. [Completed small-seed results](../../overnight_parallel_20260914/status_20260915T010508Z/README.md).

**Chains 3 and 6 both now match 26 buses in one-hour MIPs.** C6's new fleet proof took 28.1 minutes in a 201,073-column pool, following 119.9 minutes of CG. Total MIP time was 60.1 minutes; charging remains unproved. Its selected covering routes pass individual replay but contain 57 duplicated trip assignments; their removal was not separately validated. C3 k27 has now certified CG after 118.3 minutes; its integer result is still pending.

Largest individual one-hour matches across chains 1–6 are 24, 22, 26, 23, 24, 26. This combines the original and k26–28 continuation campaigns and does not mean every smaller target is matched. The separate longer C2 k25 match remains 25. Original campaign totals remain 59 CG endpoints and 58 collected MIPs; C5 k25 completed later during collection and awaits the next detailed result sample. [Original results](../../cumulative_budget_20260913/status_20260915T010508Z/README.md).

Baseline physics remain covering, 240 kWh / 240 kW and start fee 5, without shared charging capacity or a terminal-SOC floor. Pool fleet proof, pricing certificate and physical validation remain separate.

**Cluster: 32 running and 32 pending jobs.** The nine longer saved-pool MIPs and four fixed-state capacity calls remain active with no published endpoints. One original graph build, C1 k28, hit its 12-hour limit; its previously queued 24-hour recovery is running. Seventeen other original graphs are validated. [Timeout and recovery evidence](graph_timeout_recovery.json). No new confirmed preemption or MIP execution failure appeared; no job was submitted, canceled or requeued by this check. The 33 held historical tasks remain untouched.

Snapshot `20260915T010508Z` ran 21:05–21:13 EDT (478.5s), SHA `9fe87b1841e1fb66d121315b5c2ba766677ed378532a402cc5c57f46479a4ee5`. Register/workbook: 3,053 artifact-stage records across 67 source groups, all six supplements retained. Source checks cover 318 core endpoints plus 132 evening/seed/continuation endpoints. Thirteen workbook views were checked; the formula scan found no errors. The unchanged prior workbook supplied the before-view baseline. Figures and Slides are preserved.
