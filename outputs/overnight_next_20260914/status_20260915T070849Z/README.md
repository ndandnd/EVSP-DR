# Research results — 15 September, 03:17 EDT

**The larger compact-start comparison now has seven verified MIPs: six target matches and one unresolved miss.** All 24 CG runs have ended; three have pricing certificates and 21 reached four-hour limits. The remaining 17 MIPs are not yet verified. Early completion still selects this subset, so six of seven is not the overall experiment success rate.

| Chain and target | Starting method | Integer buses | Saved-pool fleet bound | CG minutes and stop | Fleet-search minutes | Total MIP minutes |
|---|---|---:|---:|---|---:|---:|
| C1 k25 | Core | 25 | 25, proved | 239.1, time limit | 0.5 | 5.7 |
| C1 k25 | Expanded to 512 sequences | 25 | 25, proved | 239.2, time limit | 0.5 | 6.2 |
| C3 k20 | Expanded to 512 sequences | 23 | 20, gap open | 102.7, converged | 180.0 | 210.1 |
| C4 k20 | Core | 20 | 20, proved | 239.1, time limit | 0.4 | 11.9 |
| C4 k20 | Expanded to 512 sequences | 20 | 20, proved | 239.1, time limit | 0.4 | 47.4 |
| C6 k20 | Core | 20 | 20, proved | 239.2, time limit | 0.1 | 11.2 |
| C6 k20 | Expanded to 512 sequences | 20 | 20, proved | 239.2, time limit | 0.1 | 5.4 |

All seven pass recorded individual-route replay. The six target matches also prove their charging-related objective within their saved pools. C3's charging search remains open. Shared charger capacity and a minimum ending SOC are absent; duplicate trip removal has not been separately validated. These are baseline covering tests with 240 kWh/240 kW and a charging-start fee of 5. “Core” retains earlier integer-solution routes plus positive-weight LP routes. The expanded start fills that core to 512 distinct trip sequences; those sequences are replayed on the new graph. [All 24 cases, CG values, MIP bounds and source hashes](compact_large_results.csv).

**The bottleneck depends on the case.** C6 k20 takes approximately four hours of CG, then proves 20 buses in 8–9 seconds of fleet search. C3 k20 expanded converges in 102.7 CG minutes but still has 23 buses and bound20 after three hours of fleet search. Its weighted LP objective is 2,000,780.100492 and fractional route weight is20. This is an open integer gap, not proof that its pool cannot support20. Do not conflate it with the [proved C1 k15 compact-pool limitation](../status_20260915T060910Z/c1_k15_pool_limitation_audit.json).

**Chain continuation is still running.** Newly collected C1 k26 and C2 k27 CGs stop at their four-hour limits, with final minimum reduced costs −0.002510846 and −0.009457100. Their weighted RMP objectives are 2,601,093.913449 and 2,701,137.303396; these are not certified full-model lower bounds. Their final MIP results were not in this scientific collection. Largest verified one-hour target matches remain25/26/27/26/26/27 across chains1–6.

**A useful additional search is running.** Job222757 searches the unchanged C3 k28 pool (166,052 columns), following its original29/bound28 result. All non-time solver settings, the input, original CG/journal and recorded ordered-pool hash are frozen. Limits are10,800s fleet and12,600s total; this starts a new tree with the same greedy initialization policy. Default partition,8CPUs,24GB,4.5-hour allocation, private requeue attempts, scaglione-compute-01 excluded. Native large-license check passed; observed running03:14EDT. [Launch and matched-design audit](../../continuation_gap_20260915/README.md). Job220545 continues the longer C5 k25 search. Neither has a verified final result yet. [Registered longer-search status](longer_gap_results.csv).

Queue at the end of collection:24 running,10 genuine input dependencies and one newly ready job awaiting scheduling;33 held historical tasks excluded. No new execution failure, confirmed preemption or invalid dependency. The new C3 campaign was registered after this full collection began and will enter the next full scientific collection; its launch has separate timestamps. No real dependency was removed, and held/V2G jobs were untouched.

Two MIPs completed after their campaign's endpoint scan:207680 (C3 k20 core) and207694 (C6 k25 expanded). Their scheduler paths and hashes are [retained for the next verification](late_scheduler_results.json); they are not silently added to the seven-result table. The completed smaller-start, LP-addition, reserve and fixed-state pricing comparisons are unchanged.

Snapshot20260915T070849Z began07:08:49UTC and finished07:17:45UTC after535.7seconds. SHA256 `69fb81730437e6acac30011013aaa8531d5ccbf3398cd44bc00dcfbce2b32c4c`. Preemption study929 attempts. Register/workbook3,212 records across71 source groups, exact six supplements retained. Checks cover321 core endpoints,171 evening endpoints and66 large/pool-diagnostic endpoints. The new longer-search verifier passed nine completed control pools and rejected an altered ordered-pool hash. [Ten changed endpoints](new_endpoints.csv) · [Pool validation](pool_experiments_validation.json) · [Document verification](doc_verification.json).
