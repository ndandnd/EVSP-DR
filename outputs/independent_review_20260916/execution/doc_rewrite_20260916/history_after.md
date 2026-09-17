# **EVSP DR Current Research**

Week of 14 September. Latest full results collection: 15 September, 00:07–00:16 EDT. Queue observations and launch records have their own timestamps. The cumulative-budget comparison completed 13 September; the model/source audit is dated 13 September. Figures remain in the two figure tabs.

**Reading order:** current chain results and definitions → measured algorithm improvements → charging comparisons → GIRO assumptions, stricter results and next tests. Older tables are under Historical appendix. Figures remain in the two figure tabs.

## **Overnight additions — 14 September, 21:54 EDT**

The 00:16 EDT collection on 15 September shows 63 running jobs and 38 waiting for required inputs. All new work is on the default partition. No job is blocked by an array limit or an unsatisfiable dependency; the GPU node remains excluded. The 33 held historical tasks are separate.

| New experiment | Submitted work | Question |
| ----- | ----- | ----- |
| **Compact inherited pools at k=20 and k=25; all six chains** | **24 CGs (2 certified) \+ 24 own-CG MIPs** | **Can a few hundred retained routes preserve the larger chains’ integer performance?** |
| **Repair 13 pools that proved unable to match target** | **26 independent MIPs** | **Do LP-used routes help more than the same number of other routes?** |
| **Use only each donor LP’s positive-weight routes** | **9 independent MIPs; all finished** | **Does the fractional solution’s route set contain a target-sized integer solution?** |
| **Smaller battery and 15% reserve on four duties** | **10 independent allocations; all completed** | **Do stricter energy requirements cause difficulty before shared-charger contention?** |

**Total: 69 independent allocations plus 24 dependent MIPs.** All are submitted and their resources checked. The larger tests reuse completed k19/k24 parents and existing graphs. Each child keeps either the previous integer routes plus all positive-weight LP routes, or that same core expanded to 512\. Both larger arms use the same a0e0 solver revision; the earlier smaller-seed cohort used e091, so comparisons across those cohorts are not a pure size effect.

**First finding:** all nine LP-support-only pools prove that they need 9–13 buses for targets of 8 or 10, although their full donor pools match target. Some routes with zero weight in the final LP therefore matter for the integer solution. Zero LP weight does not mean zero reduced cost. The first addition pair has finished: on chain 5, target 8, both the LP-used-route additions and the same-count zero-LP-weight additions still require 9 buses within their pools. The other twelve pairs are unfinished.

Large CGs allow four hours, followed by MIPs allowing 3½ hours total, including up to three hours for fleet search. Stage two limits fleet to no more than the first-stage incumbent and optimizes charging. The reserve screen uses 236.44 kWh batteries and 35.466 kWh reserve, with 220 minutes CG and a ten-minute MIP. It tests constant-rate charging, not the nonlinear GIRO curve, and assumes no 65% terminal target. Prior seed computation is recorded separately.

[Overnight plan, exact jobs, budgets and source records](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/README.md) · [Nine completed diagnostic results](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/FIRST_DIAGNOSTIC_RESULTS.md). The document and workbook now include the full 00:07–00:16 collection on 15 September. The launch record keeps its original timestamp.

## **Current results — 15 September, 00:16 EDT**

All nine longer searches recover their targets from unchanged pools. Every fleet minimum is proved within its saved columns; individual-route replay passes. Charging cost remains unproved after the full 3½-hour MIP allowance. Shared charger capacity and duplicate-removal validation are separate.

| Case / target | Original buses | Rerun buses | Minutes to fleet proof |
| ----- | ----- | ----- | ----- |
| C1 / 20 | 21 | 20 | 40.8 |
| C1 / 22 | 23 | 22 | 33.1 |
| C2 / 23 | 25 | 23 | 7.5 |
| C2 / 24 | 25 | 24 | 134.0 |
| C3 / 25 | 26 | 25 | 52.6 |
| C4 / 21 | 22 | 21 | 19.1 |
| C4 / 24 | 25 | 24 | 111.0 |
| C4 / 25 | 26 | 25 | 68.1 |
| C6 / 25 | 26 | 25 | 36.8 |

The code, pool, non-time solver settings and initializer summaries match the originals. Fleet search was allowed up to three hours instead of 30 minutes. Each rerun starts a new search tree. Two proofs finished within 30 minutes, so extra time alone does not explain every recovery; execution timing was not controlled. These results establish that the saved columns already support the targets. They do not establish global fleet optimality.  
New one-hour fleet matches: chain 1 reaches 25 buses and chain 2 reaches 26\. Their fleet proofs took 10.9 and 7.2 minutes; total MIP time was about 60 minutes each. Both CG runs stopped near four hours without pricing certificates. Each selected route passes replay; duplicate removal has not been separately validated, and shared charging capacity is absent from this baseline.

Chain 3 still matches 27 buses.

 CG took 118.3 minutes. Gurobi proved a 27-bus minimum within the 156,516 saved columns after 23.8 minutes; the full two-stage MIP took 60.2 minutes. Charging optimality remains open. Each route passes replay, but removal of 57 duplicated trip assignments has not been separately checked. This baseline omits shared charger capacity and a terminal energy floor.

Smaller battery and 15% reserve: all ten tests completed. Eight recover one bus. Duty 13405 still needs two buses within both tested pools, despite CG convergence. Duty 13408 also recovers one bus when capacity limits and PARX at 60 kW are both enforced.

| GIRO duty | Charging and capacity treatment | Integer buses | CG minutes |
| ----- | ----- | ----- | ----- |
| 13405 | PARX 240; no capacity rows | 2 | 6.0 |
| 13405 | PARX 60; no capacity rows | 2 | 8.6 |
| 13406 | PARX 240; no capacity rows | 1 | 8.1 |
| 13406 | PARX 60; no capacity rows | 1 | 11.1 |
| 13407 | PARX 240; no capacity rows | 1 | 15.3 |
| 13407 | PARX 60; no capacity rows | 1 | 20.4 |
| 13408 | PARX 240; no capacity rows | 1 | 2.2 |
| 13408 | PARX 60; no capacity rows | 1 | 3.1 |
| 13408 | PARX 240; capacity enforced | 1 | 66.4 |
| 13408 | PARX 60; capacity enforced | 1 | 77.0 |

All ten CGs certify convergence for their weighted objective, and all ten fleet and charging-cost minima are proved within their saved pools. Batteries are 236.44 kWh, with a 35.466 kWh reserve. Charging power is constant; there is no assumed 65% end-of-day target. Eight tests omit shared-capacity constraints; two enforce them. All ten pass a station-count check afterward. Route feasibility follows from the driver construction; independent continuous replay has not been done. A one-bus capacity test does not establish performance when several buses compete for chargers.

For duty 13405, the certified weighted LP objective is 109177.584 with PARX at 240 kW and 109177.606545 at 60 kW; the fractional route weight is 1.090909 in both. That weight is not a separately optimized fleet lower bound. The two-bus pool result alone does not prove that every possible one-bus route is infeasible.

Fixed-state capacity pricing: the reference finished both tests; the cached version hit its deadline in both.

| Frozen LP state | Reference pricing | Cached pricing |
| ----- | ----- | ----- |
| One duty, 17 trips | Completed in 214.9 min | No result by 238.4 min |
| Two duties, 23 trips | Completed in 191.5 min | No result by 237.1 min |

The source pools, raw dual vectors, LP objectives and matrix sizes match within each pair. No caching speed improvement is demonstrated by these two tests; different machines and execution timing limit a general performance claim. The cached times are deadlines without completed pricing results, not successful solve times or preemptions.  
The two-duty LP has 7,823 rows, 50 columns and 249 nonzeros, and took 0.024 seconds to solve. Its reference pricing call returned reduced cost −799976.896; the one-duty call returned −599985.576. These are single calls at fixed dual values, not converged CG runs.

Larger compact starts: the first two k20 CGs are complete, both on chain 3\. The 306-sequence core takes 110.7 minutes; the 512-sequence start takes 102.7 minutes. Both certify weighted LP objective 2,000,780.100492 and fractional route weight 20\. Their integer solves are pending.

[Current results, proof limits and source records](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T040735Z/README.md)

## **Earlier evening experiments and verified results**

## 

## 

The new batch has 49 independent jobs: 36 CG comparisons, nine longer MIPs and four capacity-pricing calls. All are submitted; the 36 following MIPs wait only for their own CG. Two pricing attempts hit a wrapper import error and were replaced after a native check; the failed attempts remain recorded.

| New work | Independent jobs | Question |
| ----- | ----- | ----- |
| **Useful inherited routes versus the same set expanded to 512** | **36 CGs, then 36 MIPs** | **Can a compact pool reproduce the benefit of full-pool inheritance?** |
| **Nine remaining larger-chain gaps** | **9 MIPs** | **Does more integer search on unchanged columns recover the target?** |
| **Two fixed starting states, two pricing versions** | **4 pricing calls submitted** | **Why does pricing become slow when charger-capacity dual prices are nonzero?** |

Verified at 00:16 EDT, 15 September: 30 of 36 MIPs have completed. Twenty-nine match target and prove fleet within their pools; one retains an open gap. Both treatments still match every k=8 and k=10 case. At k=15, chain 3 uses 17 buses with the core start, versus 15 with the 512-route start; chains 5 and 6 match 15 with both. All selected routes pass replay, and all 36 CG runs have certified convergence.

Six k=15 MIPs remain unpublished. For chain 3, the core starts with 198 trip sequences and ends with 26,239 columns; the 512-route start ends with 24,459 columns. Their certified weighted LP objectives agree to numerical precision at 1,500,507.4203245. The core MIP finds 17 buses after three hours of fleet search, with lower bound 15; the other proves 15 in 4.4 seconds. Thus the core may still contain a 15-bus solution. This comparison does not prove that its columns are insufficient. The completed k=8/k=10 comparison is encouraging; the k=15 comparison is still incomplete. “Core” keeps the previous integer solution plus every positive-weight LP route; “512” keeps that core and adds routes up to 512\. CG minutes include import and CG at this k; previous computation and the MIP are separate.

## 

| Case | Buses: core | Buses: 512 | CG minutes: core | CG minutes: 512 |
| ----- | ----- | ----- | ----- | ----- |
| C1, k=8 | 8 | 8 | 38.5 | 31.4 |
| C2, k=8 | 8 | 8 | 15.5 | 14.2 |
| C3, k=8 | 8 | 8 | 10.2 | 5.5 |
| C4, k=8 | 8 | 8 | 48.0 | 36.4 |
| C5, k=8 | 8 | 8 | 14.0 | 6.5 |
| C6, k=8 | 8 | 8 | 8.7 | 4.5 |
| C1, k=10 | 10 | 10 | 93.8 | 49.0 |
| C2, k=10 | 10 | 10 | 38.9 | 29.6 |
| C3, k=10 | 10 | 10 | 21.8 | 15.7 |
| C4, k=10 | 10 | 10 | 43.8 | 24.1 |
| C5, k=10 | 10 | 10 | 21.6 | 17.9 |
| C6, k=10 | 10 | 10 | 12.5 | 8.1 |
| C1, k=15 | pending | pending | 168.4 | 147.6 |
| C2, k=15 | pending | pending | 161.2 | 159.6 |
| C3, k=15 | 17 | 15 | 50.9 | 46.5 |
| C4, k=15 | pending | pending | 166.5 | 132.3 |
| C5, k=15 | 15 | 15 | 99.4 | 100.2 |
| C6, k=15 | 15 | 15 | 98.5 | 77.1 |

Six of twelve k=15 MIPs are now published; the other six remain pending in this result table. Baseline: covering, 240 kWh / 240 kW, start fee 5, without shared charger capacity or a terminal-SOC floor. [Complete result table, definitions and source hashes.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_evening_20260914/status_20260915T040735Z/README.md)

What this establishes: Across the matched k=8/k=10 tests, the old and richer-start runs have certified weighted LP objectives agreeing within 0.000002. Yet thirteen old pools provably require extra buses, while both richer starts now reach every target. For those thirteen cases, more MIP time on the old pool could not recover the target: its column set was insufficient. CG convergence alone did not produce every combination needed by the integer stage. The effective model, inputs and CG revision were checked; the experiment changes the inherited routes. [Matched-input and LP-objective audit.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_evening_20260914/status_20260915T010508Z/pool_limitation_evidence.json)

## 

## 

The compact core contains the previous integer solution and all positive-weight LP routes: 39–234 distinct trip sets. The other arm keeps that core and fills to 512\. Both use identical inputs, code and budgets. CG allows four hours; MIP allows up to three hours for fleet search within 3½ hours total. These are baseline models without shared capacity or a terminal SOC floor; the separate four pricing calls include capacity.

Queue in the 21:05–21:13 EDT collection: 32 EVSP–DR jobs running and 32 pending; no unsatisfiable dependency or use of the reserved GPU node. The earlier queue thinned because independent batches finished. Previous-k and graph dependencies are real and remain intact. All new work uses the default partition. [Exact overnight plan, budgets and job records.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_evening_20260914/README.md)

## **Overnight diagnostics: 14 September**

**Results collection, 21:05–21:13 EDT: 32 EVSP–DR jobs running and 32 solver jobs waiting for required inputs. No new confirmed preemption or MIP execution failure appeared. The original C1 k28 graph build hit its 12-hour limit; its previously queued 24-hour recovery is now running. The four fixed-state capacity calls continue, with no diagnostic endpoint yet.**

Queue near the end of this collection: 32 running jobs and 32 pending. The 33 held historical tasks are excluded. Scheduler counts and published result counts are sampled at different points in the collection; a job can finish between them.

| Experiment | Running jobs | Verified result status |
| ----- | ----- | ----- |
| Core versus 512 inherited routes | 6 CG \+ 5 MIP | 30 CG certificates; 25 target-matching MIPs |
| Nine unresolved larger-chain pools | 9 MIPs | No published endpoint yet |
| Earlier small integer/LP seed sets | 0 | All 36 MIPs complete; comparison below |
| Current larger-chain steps | 6 CG \+ 1 MIP | C3 and C6 both match 26; C3 k27 CG certified |
| C1 k28 graph recovery | 1 | Started after the original 12-hour limit |
| Fixed-state capacity pricing | 4 calls | No endpoint or pricing-result claim yet |

These campaigns are already submitted. Keep their real input dependencies. The independent work provides parallelism while the six chains advance sequentially.

### **Additional overnight work submitted, 14 September**

| Experiment | CG jobs | Later MIPs | What it tests |
| ----- | ----- | ----- | ----- |
| **Previous-k integer routes versus same-count routes chosen by LP weight** | **All 36 CGs ended: 35 certificates, one time limit** | **36, each waits only on its own CG** | **Which inherited routes help produce a good integer pool? Six chains at k=8, 10 and 15\.** |
| **Capacity pricing on four one-bus duties: reference versus cached version** | **8 launched; 7 completed, 1 preempted** | **Short diagnostic within each allocation** | **Which duty structures make pricing slow?** |

**The 36 warm-start tests use already-built graphs. Both methods start with singleton routes and replay a small subset from the previous k. Each pair has the same inputs, code, model and time limits. Prior CG and MIP costs are retained, so the inherited information is not treated as free. The selected sequence counts match, but their coverage differs: the integer-selected routes cover more of the previous instance’s trips in all 18 pairs. None of the two selections is identical. This compares two practical seeding strategies; it does not isolate integrality from trip coverage.**

**CG allows four hours; its later MIP allows three hours for fleet search within 3½ hours total. The eight pricing diagnostics allow 220 minutes of CG plus a ten-minute MIP. All independent cases are eligible together on the default partition; Slurm determines resource admission.** [Overnight plan, reasons and exact launch records.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/research_register/OVERNIGHT_PLAN_20260914.md)

### **Further work added this afternoon**

| Experiment | Parallel work | Question |
| ----- | ----- | ----- |
| **Fresh columns available just before four hours** | **Three reconstructed pools and all three MIPs complete** | **How do the small warm starts compare with fresh pools from approximately the same four-hour CG allowance?** |
| **Combine solutions from different partitions of the 32-duty instance** |   | **Can routes from different partitions work together to reduce the fleet?** |
| **Protect the larger-chain graph builds** | **18 conditional recovery checks queued** | **Reuse successful builds; retry only an actual timeout with a longer allowance.** |

**The three fresh pools come from logged iteration boundaries at 239.881, 239.917 and 239.990 minutes. They contain 67,294, 64,225 and 71,847 columns. Their MIPs have now finished, with the same three-hour fleet allowance and 3½-hour total allowance as the small-seed tests.**

| Fresh pool | Target | Buses found | Pool fleet bound | Fleet proved? |
| ----- | ----- | ----- | ----- | ----- |
| **Chain 1, approximately four-hour pool** | **15** | **19** | **15** | **No** |
| **Chain 2, approximately four-hour pool** | **15** | **17** | **15** | **No** |
| **Chain 4, approximately four-hour pool** | **15** | **19** | **15** | **No** |

**All selected routes pass individual replay. These are open integer gaps: the unchanged pools may still contain better solutions. They are reconstructed iteration pools, not completed four-hour CG endpoints or new pricing certificates. The corresponding k=15 small-seed MIPs remain pending, so the matched comparison is not yet complete.**

 [Four-hour pool control results and source hashes.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_parallel_20260914/status_20260914T215636Z/CURRENT_STATUS.md)

The decomposition experiment uses one 750-trip, 32-duty input. We already solved nine different ways of dividing it into four groups of eight duties. Now we combine their saved routes: nine individual pools, all 36 pairs, and one pool containing all nine. Individual and pair MIPs have equal two-hour limits; the all-nine case has four hours. These are alternative treatments of one instance.

**First selection, complete at 19:04 EDT:** All 46 searches have finished. The 45 individual/pair searches use 34–37 buses and prove their fleet minimum in their selected pools. The all-nine union finds 34 buses with a pool bound of 33 after four hours; its fleet minimum is not proved. All selected solutions pass individual-route replay. No combination improves its best contributing partition.

**LP-preserving selection, complete at 19:04 EDT:** All 46 searches have finished. No pair improves its best contributing partition. Every pair excludes 32 buses within its pool, although 35 pair searches retain larger fleet gaps. The all-nine union finds 34 buses with a pool bound of 32 after four hours. Therefore, a 32-bus solution in this larger selected pool remains unresolved. Neither treatment found the target.

**How the two selections differ:** the first selector kept 512 routes per group, including its integer solution, but omitted 3,187 of the 3,413 routes with positive weight in the source LP solutions. The second matched batch is now submitted: 46 independent MIPs, jobs 190402–190447. Its nine constructed pools retain all source LP routes as well as the integer routes, then fill to the same 512 per group. Every group fits; native validation accepted all 2,048 columns in the test partition with no rejected or repaired columns. This tests route selection while keeping pool size and MIP allowances fixed.

No full parent graph or new CG certificate is produced here. The independent simultaneous-trip lower bound is 29 buses: a feasible 32-bus result would match GIRO, but would not prove global optimality. Shared charger capacity and a terminal SOC floor remain absent from this baseline.

### **Capacity-aware pricing: seven completed tests and one interruption**

### **Each cell below gives reference / cached pricing. All seven completed MIPs find one bus, prove their fleet and charging-related cost within the saved pool, and pass the shared-charger check with no duplicated trips.**

| Duty | CG minutes | CG certificate | Integer buses | Charging-related cost |
| ----- | ----- | ----- | ----- | ----- |
| **13405** | **220 / 220** | **No / No** | **1 / 1** | **94.720 / 94.720** |
| **13406** | **220 / 220** | **No / No** | **1 / 1** | **56.952 / 56.952** |
| **13407** | **220 / interrupted** | **No / none** | **1 / no result** | **81.584 / no result** |
| **13408** | **2.93 / 1.75** | **Yes / Yes** | **1 / 1** | **36.536 / 36.536** |

### **The five 220-minute endpoints stopped during pricing, without a convergence proof. The cached 13407 job was preempted after 2h20m10s; its old worker file incorrectly still says running. Its partial pool is preserved, but it has no completed CG or MIP result. This interruption is separate from the earlier MIP preemption.**

### **These are one-bus tests with shared-capacity rows, constant-rate 240 kWh / 240 kW, zero reserve, flat prices and no terminal SOC floor. Individual route feasibility is by construction; the shared-capacity sweep is a separate check. One-bus success does not establish performance when several buses compete for chargers.**

###  **[Current capacity results and interruption evidence.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_parallel_20260914/status_20260914T205702Z/CURRENT_STATUS.md)**

### **Earlier parallel work, 14 September**

### 

| Experiment | Independent work | What we learn |
| ----- | ----- | ----- |
| Extend six chains to k=26–28 | 18 graph builds submitted; then 18 CGs and 18 MIPs | How far the unchanged baseline scales. Only each chain’s CG remains sequential. |
| Search unresolved saved pools | All 14 complete: four target recoveries, three proved pool limits, seven open gaps | Whether the existing columns already support better integer solutions. |
| Combine independently generated columns | All six MIPs complete: three target recoveries and three open fleet gaps | Whether different CG runs found routes that work better together. |
| Stricter charging tests | All 12 CG tests now have endpoints. The four capacity-enforced k=2 tests hit their pricing deadline. Separate one-hour MIP follow-ups remain the matched integer-search comparison. | Separate charger capacity, depot power and energy reserve; compare reference pricing with cached pricing. |

At the 13:46 EDT check, all 18 graph builds were running and making progress. Each now has a conditional recovery job: reuse a verified successful build, or retry a confirmed timeout with a 24-hour internal limit. Active builds are preserved. Larger independent CG arrays retain the default limit of 50\. More CPUs cannot remove a genuine previous-k dependency; independent experiments provide the parallel work. The strict tests first save short preliminary MIPs; their separate one-hour MIPs provide equal integer-search budgets. Capacity cases receive more CG time, so this is a feasibility pilot, not a controlled comparison of total running time. The batches below are earlier completed or finishing diagnostics. [Current launch plan and job maps.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/parallel_followup_20260914/README.md)

### **Combining saved columns: three target recoveries**

| Case | Best fleet found in each separate pool | Fleet from combined pool | Best possible fleet in combined pool |
| ----- | ----- | ----- | ----- |
| Chain 1, target 8 | 9 / 9 / 9 | 9 | Not proved; lower bound 8 |
| Chain 2, target 8 | 9 / 9 / 9 | 8 | 8, proved |
| Chain 4, target 8 | 9 / 9 / 9 | 9 | Not proved; lower bound 8 |
| Chain 5, target 8 | 9 / 9 / 9 | 8 | 8, proved |
| Chain 5, target 10 | 11 / 11 / 11 | 10 | 10, proved |
| Chain 6, target 10 | 11 / 11 / 11 | 11 | Not proved; lower bound 10 |

The three source pools come from original, 200-column and complementary column selection. Combining them adds no new pricing. All selected routes pass individual replay. Some source MIP searches remain unfinished, so these recoveries do not establish that combining columns was necessary. Proofs concern the saved pools, under baseline physics without shared charger capacity or a terminal SOC floor.

### **Small previous-k seeds: completed comparison**

**At 21:13 EDT, this comparison is complete: 36 CG endpoints (35 certified, one time limit) and 36 MIPs. Nine MIPs match target; thirteen prove that their saved pools require extra buses; fourteen retain open fleet gaps. Both bus-count columns below are final integer results—the seed labels describe which previous-k routes were supplied to CG.**


| Case | Integer-seed buses | LP-seed buses | Pool fleet bounds: integer / LP | CG minutes: integer / LP |
| ----- | ----- | ----- | ----- | ----- |
| C1, k=8 | 9 | 9 | 9 / 9 | 74.0 / 74.2 |
| C1, k=10 | 11 | 12 | 11 / 10 | 101.6 / 96.7 |
| C1, k=15 | 16 | 19 | 15 / 15 | 185.7 / 239.0 |
| C2, k=8 | 8 | 9 | 8 / 9 | 17.9 / 13.1 |
| C2, k=10 | 11 | 11 | 10 / 10 | 38.8 / 51.4 |
| C2, k=15 | 15 | 18 | 15 / 15 | 219.3 / 238.7 |
| C3, k=8 | 9 | 9 | 9 / 9 | 14.8 / 12.2 |
| C3, k=10 | 11 | 11 | 10 / 10 | 28.6 / 27.7 |
| C3, k=15 | 15 | 17 | 15 / 15 | 66.7 / 70.3 |
| C4, k=8 | 9 | 9 | 9 / 9 | 52.6 / 49.6 |
| C4, k=10 | 11 | 11 | 11 / 11 | 56.2 / 67.5 |
| C4, k=15 | 16 | 18 | 15 / 15 | 228.7 / 223.7 |
| C5, k=8 | 8 | 9 | 8 / 9 | 26.4 / 26.3 |
| C5, k=10 | 10 | 11 | 10 / 11 | 29.4 / 40.0 |
| C5, k=15 | 15 | 16 | 15 / 15 | 142.4 / 181.8 |
| C6, k=8 | 8 | 8 | 8 / 8 | 9.1 / 10.2 |
| C6, k=10 | 10 | 11 | 10 / 11 | 20.3 / 22.0 |
| C6, k=15 | 16 | 18 | 15 / 15 | 209.5 / 188.9 |

For chain 5 at target 8, both CG runs certify the same weighted objective, 800,431.772270. Yet their generated pools have different integer optima: 8 and 9 buses. This shows that matching the LP objective does not ensure equally useful integer columns. Both methods imported seven previous-k sequences, but those sequences covered 156 versus 100 parent trips. The comparison tests the two selection methods; it does not isolate integrality from trip coverage.

  All eighteen pairs are complete: integer-route seeds find fewer buses in eleven and tie in seven. Integer-route seeds match eight of eighteen targets; LP-selected seeds match one. A fleet minimum is proved when the buses found equal the corresponding pool fleet bound. For example, C3 k=15 reaches 15 with integer-route seeds; the LP-seeded search found 17 with a bound of 15, so a 15-bus solution in that pool remains possible. C1 k=8, C4 k=8 and C4 k=10 miss their targets with both seed methods, with the extra bus proved necessary in each selected pool. Small seeds therefore do not consistently reproduce the full-pool inheritance results. Trip coverage differs between seed methods, and earlier computation must still be counted. The only uncertified seed CG is C1 k=15 with LP-weight seeds: it stopped after 239.0 minutes with minimum reduced cost −1.060160. Chain 3 k=8 now requires nine buses with either seed method, proved within each pool. New paired results: C1 k=10 uses 11 buses with integer seeds and 12 with LP seeds; only the 11-bus pool optimum is proved. C5 k=15 uses 15 versus 16, with the LP-seeded pool bound still 15\. The C6 k=15 pair finishes at 16 versus 18, both with bound 15\. At k=15, integer-seed / LP-seed fleets are C1:16/19, C2:15/18, C3:15/17, C4:16/18, C5:15/16 and C6:16/18.

 [Complete source tables and proof scopes.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_parallel_20260914/status_20260915T010508Z/README.md)

### **Completed longer searches: 12:33 EDT**

| Saved pool | Earlier buses | New buses | Minutes to fleet proof |
| ----- | ----- | ----- | ----- |
| Chain 2, target 25 | 26 | 25 | 87.2 |
| Chain 4, target 22 | 23 | 22 | 32.7 |
| Chain 6, target 23 | 24 | 23 | 19.3 |
| Chain 4, target 19; continued CG pool | 20 | 19 | 46.8 |

Each pair uses exactly the same ordered columns and input data. The new fleets are proved best within those pools, and selected routes pass individual replay. The last row is a separate continued-CG treatment; the original chain 4 k=19 already matched 19\.

| Saved pool that cannot reach target | Target buses | Minimum buses proved in pool |
| ----- | ----- | ----- |
| Chain 4; 200 columns per iteration | 8 | 9 |
| Chain 4; complementary selection | 8 | 9 |
| Chain 6; complementary selection | 10 | 11 |

These three pools require different columns to reach the target. Seven other column-selection searches still have an open integer gap. None of these ten searches recovered its target. These results do not establish impossibility in the full model.

Fleet search allowed up to three hours within 3½ hours total; charging used the remaining time. Chain 6 k=23 proved its fleet in 19.3 minutes, within the original 30-minute allowance, so extra allocated time alone cannot explain every recovery. Hardware and parallel search remain uncontrolled. [Fourteen-case comparison and source hashes.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/parallel_followup_20260914/status_20260914T162902Z/README.md)

### **k=2 controls recover two buses but exceed station capacity**

Results collected at 10:35 EDT. These are four settings on the same E1-short k=2 trip set, with flat prices and shared charging capacity disabled. All four CGs have pricing certificates; all four matched one-hour MIPs prove fleet two and their charging objective within the saved pool.

| Battery / minimum SOC | PARX charging power | CG minutes | Integer buses | Charging-related cost |
| ----- | ----- | ----- | ----- | ----- |
| 240 kWh / 0% | 240 kW | 64.3 | 2 | 79.272 |
| 240 kWh / 0% | 60 kW | 65.9 | 2 | 79.272 |
| 236.44 kWh / 15% | 240 kW | 30.6 | 2 | 91.712 |
| 236.44 kWh / 15% | 60 kW | 42.8 | 2 | 91.712 |

**All four schedules use two charging connections simultaneously at station 2190L, which has one documented charger.** They meet their tested model, but fail this shared-capacity check. The four capacity-enforced k=2 runs have now stopped at their 220-minute pricing limit without a certificate; details are below. Slower depot charging and the battery/reserve treatment alone do not prevent two buses on this input.

Charging-related cost includes electricity and the fee per charging start. The 236.44-kWh treatment changes battery size and reserve together. There is no 65% terminal target. Individual route feasibility in this dedicated solver is by construction, not a separate continuous replay audit. All four k=1 capacity tests now recover one bus and pass their shared-capacity checks. The two flat-price runs have CG certificates. Under the noon-peak price, both CGs stopped at the 110-minute pricing deadline without a certificate. The reference version completed 24 iterations and the cached version 32; both ended at the same restricted LP objective and found charging-related cost 61.715. Their matched MIPs prove that cost within the respective saved pools, not across every possible route. [Earlier twelve-setting comparison and source hashes.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/strict_capacity_parallel_20260914/status_20260914T152928Z/README.md)

### **Capacity-enforced k=2 results, collected 12:59 EDT**

| Battery / reserve | PARX power | Completed CG iterations | Columns saved | Buses in matched one-hour MIP |
| ----- | ----- | ----- | ----- | ----- |
| 240 kWh / 0% | 240 kW | 4 | 27 | 3 |
| 240 kWh / 0% | 60 kW | 4 | 27 | 3 |
| 236.44 kWh / 15% | 240 kW | 3 | 26 | 12 |
| 236.44 kWh / 15% | 60 kW | 3 | 26 | 12 |

All four CG runs reached the 220-minute pricing deadline. None proved that no improving route remains. Their small saved pools are proved to require the fleets shown, and the selected schedules pass the shared-charger check. These are pool limits after unfinished pricing, not proof that the full model needs 3 or 12 buses. Route feasibility is by construction, not independent continuous replay.

This points to pricing as the immediate bottleneck: only three or four iterations finished. The no-capacity controls completed 115–130 iterations and certified. The matched one-hour follow-ups now confirm the same fleets (3 or 12\) and prove charging cost within these small pools. All twelve matched MIPs are complete. CG budgets still differ between capacity and no-capacity settings, so the comparison is a feasibility diagnostic. Extra MIP time cannot replace the routes missing from unfinished pricing. Tonight’s small pricing tests will locate which duty structures trigger the slowdown. [Verified endpoint audit and source records.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/capacity_pricing_boundary_20260914/strict_capacity_k2_audit_20260914T1659Z.md)

| New independent work | Jobs | Question |
| ----- | ----- | ----- |
| Repeat original one-hour MIP settings on nine inherited pools | 27: three repeats per pool | How reliably does the original 30-minute fleet search recover a target solution now known to exist? |
| Longer MIPs on C1 k19, C3 k22 and C3 k23 | 3 | Can the unchanged pools close the three remaining gaps from the 04:28 result snapshot? |

**All 27 one-hour repeats finished: 21 target matches, each proved within its saved pool. All three repeats agree within each case.** The three longer MIPs have now finished: C1 k19 uses 19 buses, C3 k22 uses 22, and C3 k23 uses 23\. Each fleet is proved best in its unchanged saved pool and its selected routes pass individual replay. Fleet proof took 22.3, 16.1 and 24.1 minutes, respectively. Thus these reruns recovered the targets within the original 30-minute fleet allowance, even though they had more time available. Extra allocated time alone does not explain the earlier misses.

| Case | Target | Original buses | Repeat 1 | Repeat 2 | Repeat 3 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| C2 k17 | 17 | 18 | 17 | 17 | 17 |
| C2 k18 | 18 | 19 | 18 | 18 | 18 |
| C2 k19 | 19 | 20 | 19 | 19 | 19 |
| C2 k20 | 20 | 21 | 20 | 20 | 20 |
| C3 k19 | 19 | 20 | 20 | 20 | 20 |
| C3 k20 | 20 | 21 | 20 | 20 | 20 |
| C3 k21 | 21 | 22 | 22 | 22 | 22 |
| C5 k19 | 19 | 20 | 19 | 19 | 19 |
| C6 k18 | 18 | 19 | 18 | 18 | 18 |

The repeats keep the same ordered pool, inputs, default seed, eight threads and initializer settings. They test nine selected pools, not 27 new datasets. C3 k19 and k21 miss in every short repeat, although longer searches recovered both targets from the same pools: these are incomplete integer searches. The seven other recoveries show that extra allocated time alone does not explain the earlier misses; hardware and parallel-search timing are not isolated.

All selected routes pass individual replay. Charging optimality remains a separate question. [Repeat results, times and source hashes.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/mip_repeatability_20260914/status_20260914T124947Z/README.md)

 [Frozen cases, settings and jobs 186672–186701.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/mip_repeatability_20260914/README.md)

| New comparison | Jobs | What it tests |
| ----- | ----- | ----- |
| Longer search on unchanged pools | 23 MIPs; up to 3 hours for fleet search, 3½ hours total | Is the target already in the saved columns but harder for Gurobi to find? |
| Add up to 200 columns per CG iteration | 18 independent CGs, then 18 one-hour MIPs | Can a larger batch of improving routes produce a better integer pool? |
| Select 30 less-overlapping columns per iteration | 18 independent CGs, then 18 one-hour MIPs | Can a more complementary mix of routes improve the integer pool? |
| Continue chains 4 and 5 at k=19 | 2 CGs and 2 MIPs submitted; checkpoint check passed | Does increasing cumulative CG time from four to eight hours reach convergence? |

The CG comparisons use the same inputs, physics, objective, code and cumulative time allowances as their fresh controls. Only the column-selection treatment changes. Four fresh pools already proved to require an extra bus receive new CG treatments, not longer searches on unchanged pools. All jobs use the default partition and exclude the reserved GPU computer. These are selected difficult cases, not a random sample for estimating overall success rates. [Overnight plan, exact cases and job records.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_diagnostics_20260914/README.md)

## **Longer integer searches: what the saved pools contain**

All 23 longer MIPs finished by the 04:28 EDT collection. Every original/rerun pair has identical input and ordered pool hashes. These searches add no CG columns.

| Saved pool source | Cases tested | Target fleet found | Extra bus proved necessary in pool | Fleet gap still open |
| ----- | ----- | ----- | ----- | ----- |
| Inherited chain pools | 9 | 9 | 0 | 0 |
| Fresh CG pools | 14 | 0 | 3 | 11 |

The nine inherited pools already contained target solutions. Seven reruns proved the target within 30 minutes, although the original 30-minute fleet searches missed. Extra elapsed time alone therefore does not explain every recovery. The 27 completed one-hour repetitions now recover targets in all three repeats for seven of these nine pools; C3 k19 and k21 remain above target in every short repeat.

Fresh C3 k8 proves that its pool needs 9 buses; fresh C5 k10 and C6 k10 each prove 11\. More search on these unchanged pools cannot meet the target. The other eleven fresh gaps remain unresolved. This is a selected set of prior misses, with different instance sizes in the two groups; it is not an overall warm-versus-fresh success rate.

Longer runs allow up to three hours for fleet search and 3½ hours total. All selected routes pass individual replay. Proofs concern the saved pool; shared charger capacity and terminal-SOC constraints remain absent. [Case-by-case results, times and source hashes.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_diagnostics_20260914/status_20260914T082509Z/LONGER_MIP_RESULTS.md)

## **First result: choosing columns matters**

Chain 5, target five buses; results collected 14 September, 01:26 EDT. All three CG runs have pricing certificates and reach weighted LP objective 500,276.192, with fractional route count five.

| Column selection | CG minutes | Columns in final MIP pool | Integer buses found |
| ----- | ----- | ----- | ----- |
| Original: 30 by reduced cost | 8.0 | 14,175 | 6 |
| 200 by reduced cost | 26.1 | 45,105 | 6 |
| 30 complementary columns | 21.0 | 12,029 | 5 |

Each integer fleet is proved best within its own saved pool and passes individual-route replay. The complementary selection recovers five buses; simply adding more columns does not in this case. The pools contain different routes: the larger pool need not contain the useful routes in the smaller one. This is one selected difficult case, not a general success rate. Both new CG treatments took longer here. Baseline physics and objective are unchanged.

**Longer CG at k=19:** chain 4 certified after 267.0 cumulative minutes, 27.9 extra, improving the weighted LP objective by about 0.00033. Chain 5 now also certified after 314.0 cumulative minutes, 74.7 extra, improving its objective by 0.03250. Both original four-hour endpoints remain uncertified; these are separate continuation treatments.

Chain 4’s continuation MIP found 20 buses with a pool fleet bound of 19; the earlier MIP had found and proved 19\. The new solve started from its ordinary greedy initializer (181 buses, accepted by Gurobi), rather than importing the earlier 19-bus integer solution. A separate longer MIP now finds and proves 19 buses from this same continued pool, so the target was present: the earlier 20-bus result was an incomplete integer search. This does not establish that every route from the earlier original solution survived. Chain 5’s continuation MIP now finds 19 buses and proves that fleet within its saved pool, with individual-route replay. Its original four-hour treatment found 20 with a bound of 19\. This recovery belongs to the separate longer-CG treatment. A longer MIP on the unchanged original chain 5 k=19 pool now also finds and proves 19 buses. Extra CG was therefore not necessary to establish that a 19-bus solution was already available. [Results and source hashes.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_diagnostics_20260914/status_20260914T082509Z/README.md)

## **Column-selection results so far: improvements are case dependent**

Results at 08:53 EDT. Every case with at least one completed treatment MIP is shown. Numbers are buses actually found; pending means no published integer result yet. All use the original one-hour MIP budget.

| Chain / target buses | Original selection: buses found | 200 columns: buses found | Complementary: buses found |
| ----- | ----- | ----- | ----- |
| 1 / 8 | 9 | 9 | 9 |
| 1 / 10 | 11 | 11 | 12 |
| 1 / 15 | 18 | pending | 18 |
| 2 / 8 | 9 | 9 | 9 |
| 2 / 10 | 12 | 11 | 12 |
| 2 / 15 | 17 | 19 | 19 |
| 3 / 8 | 9 | 8 | 8 |
| 3 / 10 | 11 | 11 | 11 |
| 3 / 15 | 18 | 17 | 17 |
| 4 / 8 | 9 | 9 | 9 |
| 4 / 10 | 11 | 12 | 11 |
| 4 / 15 | 19 | 20 | 19 |
| 5 / 5 | 6 | 6 | 5 |
| 5 / 8 | 9 | 9 | 9 |
| 5 / 10 | 11 | 11 | 11 |
| 5 / 15 | 16 | 17 | 17 |
| 6 / 10 | 11 | 11 | 12 |
| 6 / 15 | 20 | 19 | 19 |

The latest 200-column MIPs at k=15 find 19, 20, 17 and 19 buses for chains 2, 4, 5 and 6\. All have a fleet bound of 15 and no fleet proof. Their selected routes pass individual replay. More columns per iteration therefore show no consistent integer improvement. The C1 k15 200-column CG certified after 765.0 minutes. Its one-hour MIP finished with 18 buses and a pool fleet bound of 15, leaving the fleet gap open. These fresh-run treatments are separate from the inherited chains that already match k=15.

Both new treatments recover eight buses on chain 3 at target eight, with fleet proofs within their saved pools and individual-route replay. Other cases still miss. Chain 6 at target ten has a worse complementary incumbent, twelve versus eleven; its pool bound is ten, so the extra buses are not proved necessary. Chain 5 at target eight is different: the complementary pool is proved to require nine. New results also reduce chain 2 at target ten from twelve to eleven buses with the 200-column treatment, and chain 3 at target fifteen from eighteen to seventeen with either treatment. Their pool bounds remain ten and fifteen, so neither target is recovered yet. Chain 1 at target eight is now proved to need nine in the complementary pool. New chain 1 k10 complementary and chain 4 k10 200-column results each use twelve buses versus eleven originally. Those are timed incumbents; they do not prove twelve necessary. At k=15, complementary selection now finds 19, 17 and 19 buses on chains 4, 5 and 6, versus 19, 16 and 20 originally. All three bounds are 15 and none proves its fleet optimal. The treatment still shows no consistent improvement. The early completions are selected cases, not an unbiased success-rate sample. [Original one-hour results, bounds and timing.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_diagnostics_20260914/status_20260914T124947Z/README.md)

## **Six larger chains: original k=16–25 and continuation to k=28**

**15 September, 00:16 EDT: the largest individual matches in one-hour MIPs are 25, 26, 27, 23, 24 and 26 across chains 1–6. This includes the original k=16–25 campaign and its separate k=26–28 continuation. Each listed fleet is proved within its saved pool and passes individual-route replay. The separate longer C2 k25 search still matches 25 buses.** Largest-target matches do not imply uninterrupted recovery at every smaller k. Charging optimality remains separate. One-hour results across both continuation campaigns at 00:16 EDT:

| Chain | Largest target matched | Integer buses | CG minutes at this target |
| ----- | ----- | ----- | ----- |
| 1 | 25 | 25 | 239.4 |
| 2 | 26 | 26 | 239.6 |
| 3 | 27 | 27 | 118.3 |
| 4 | 23 | 23 | 219.6 |
| 5 | 24 | 24 | 239.6 |
| 6 | 26 | 26 | 119.9 |

CG minutes count this k’s import and CG; earlier k values, graph construction and MIP are separate. The original extension has 60 CG endpoints: 45 pricing certificates and 15 time limits. Chain 1 k25 stopped after 239.4 minutes with reduced cost −0.020743. Its MIP now matches 25 buses and proves fleet optimal within the 237,601-column pool after 10.9 minutes. The separate chain 2 k26 CG stopped at its limit after 239.6 minutes, with reduced cost −0.012524 and no pricing certificate. Its MIP now matches 26 buses, with a pool fleet proof after 7.2 minutes in 180,809 columns. Both new integer solutions pass individual-route replay; charging optimality and duplicate-removal validation remain open. C5 k=25 newly stops after 239.4 minutes without a certificate; its last reduced cost is −0.146677. C1 k=24 matches 24 buses although its CG stopped without a certificate. Separately, C3 k=26 certified in 113.2 minutes, with weighted objective 2,601,094.794385 and fractional route weight 26\. Its MIP now finds 26 buses and proves fleet optimal within the 148,580-column pool in 28.1 minutes; the full two-stage MIP takes 60.1 minutes. Charging cost remains unproved. Its covering solution contains 50 duplicated trip assignments: individual routes pass replay, but duplicate removal was not separately validated. Shared capacity and a terminal-SOC floor remain absent. C6 k=26 now also matches 26 buses, with a pool fleet proof after 28.1 minutes in its 201,073-column pool. Total MIP time is 60.1 minutes; charging optimality remains open. Its selected covering routes contain 57 duplicated trip assignments, with individual-route replay but no separate duplicate-removal validation. C3 k=27 certifies CG after 118.3 minutes and now matches 27 buses. Fleet optimality is proved within its 156,516-column pool after 23.8 minutes; total MIP time is 60.2 minutes. Charging optimality remains open. Individual routes pass replay; 57 duplicated trip assignments have no separate removal validation.

All 60 original one-hour MIPs are complete: 35 target matches and 25 misses. Separate longer searches now recover 24 of those 25 misses. Chain 5 k25 remains the only target unresolved across the original results and their separate longer reruns. The separate continuation contributes four collected MIPs: C2 matches 26, C3 matches 26 and 27, and C6 matches 26\. C5 k=25 newly finds 26 buses with bound 25, leaving its fleet optimum open. The original chain 4 k25 MIP found 26 buses with bound 25; its separate longer rerun now proves 25 within the same pool. Longer-budget results remain separate from the original controls.

[Latest chain results, CG stopping reasons and unresolved integer gaps.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_next_20260914/status_20260915T040735Z/README.md)

**The original chains 1, 4 and 5 k=19 runs stopped before CG convergence**, after about 239 minutes. Their last minimum reduced costs were −0.065741, −0.003185 and −0.088463, below the −0.0001 threshold. These restricted-master objectives lack a pricing certificate. Chain 4 nevertheless found 19 integer buses and proved that fleet within its saved pool. Chains 1 and 5 each found 20, with pool fleet bound 19 and an open gap. The separate longer continuations for chains 4 and 5 now have pricing certificates; neither changes the original endpoint. Two more CG runs reached their four-hour limits: chain 1 k=20 and chain 4 k=21. Their last reduced costs were −0.008585 and −0.007782, so neither has a pricing certificate. Their saved pools still feed the scheduled MIPs. Chain 5 k=21 also reached its four-hour CG limit: 239.6 minutes, 496 iterations, last reduced cost −0.058611. It has no pricing certificate. Its MIP now finds 21 buses, proves that fleet within its saved pool and passes individual-route replay. This integer result does not create a CG pricing certificate.

At 12:33 EDT, longer-search reruns have recovered fifteen original gaps. Six targets remain unmatched across the collected extension results: chain 1 k=20 and k=22; chain 2 k=23 and k=24; chain 3 k=25; and chain 4 k=21. C2 k25 succeeding does not erase its earlier k23–24 gaps. New C2 k24 finds 25 buses with bound 24 and no fleet proof. New one-hour results are C1 k20: 21 buses, bound 20; C2 k23: 25, bound 23; C3 k25: 26, bound 25; C4 k21: 22, bound 21\. All pass individual-route replay, but none proves its extra buses necessary. The C1 k19 and C3 k22–23 longer-search reruns are complete and all match their targets. The new parallel work is listed at the top of this document. [Latest longer-search results and remaining-gap map.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/parallel_followup_20260914/status_20260914T162902Z/README.md)

| Stage | How it starts | Time budget per case |
| ----- | ----- | ----- |
| **Prepare the larger trip graph** | **Independent jobs; up to 50 at once** | **12 hours** |
| **Import saved routes, then run CG** | **Wait for its graph and the preceding k in the same chain** | **4 hours total** |
| **Two-stage integer solve** | **Wait only for this case’s CG; does not hold up the next k** | **1 hour total** |

Each step adds one randomly selected unused GIRO duty, preserves the existing trips, and checks every eligible sequence from the preceding saved pool for reuse. The random order is frozen before solving. We keep the successful baseline: covering, 240 kWh batteries, 240 kW charging, flat prices and a fee of 5 per charging start. This expansion does not add station capacity or a return-SOC floor. The frozen duty order extends through k=40. The original k=16–25 campaign remains intact; a separate continuation at k=26–28 has now been submitted, preserving that same order.

We will report graph time, import/pricing/LP time, CG stopping reason and certificate, and integer fleet/bound separately. Large caches go in the shared data directory. The native prelaunch check passed, including the unrestricted Gurobi license, unchanged graph bytes with progress logging, inherited CG, and physical replay of the final MIP. [Launch records and exact settings](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/chain_extension_20260913/README.md).

**Earlier full-parent CG attempt is paused; the new pool-combination MIPs above are separate.** The shared 750-trip graph build hit its 12½-hour limit before CG started. No cache was saved; all 20 dependent CG/MIP jobs cancelled before starting. This was not preemption or out-of-memory (25.4 GiB peak against 128 GiB requested). That decomposition batch is inactive. The new k=16–25 chain expansion above is running; held historical jobs remain untouched. Graph preparation needs profiling before a retry. [Execution evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/queue_recovery_20260912/status_20260913T093758Z/README.md). Completed scientific results below are unchanged.

## **Reference comparison at k=15**

**All six chains now reach k=15. The 37 repaired CG runs have pricing certificates; their 37 MIPs match the target, with fleet proofs within each saved pool and individual-route replay.**

**How to read this table.** Each chain is a different nested selection of GIRO duties. Every row here has target k=15: the trips originally assigned to 15 GIRO buses. The two fleet columns give the integer buses actually found by our MIP, using either all eligible saved sequences or the earlier 512 limit. CG minutes measure this k=15 run only, including import and CG; they exclude earlier k values, original graph construction and the final MIP.

**What “512” means.** It was a temporary limit on saved trip sequences checked for reuse, with a separate 15-minute replay budget. The importer first keeps the cheapest saved route for each trip set, then prioritizes sequences with more trips, breaking ties by lower cost per trip and stable trip IDs. Thus 512 was an engineering cutoff, not a bus count, trip count or limit on columns CG could later generate.

**What full inheritance does.** At the next k, we take every eligible representative from the previous run’s saved pool, rebuild a feasible charging schedule for its ordered trips on the new graph, and retain the routes that pass validation. New trips get single-trip initializers; ordinary CG then adds more routes. We do not inherit the LP basis, dual values or proof. “Full” means the saved pool, not every mathematically possible route, and it does not inject GIRO solution routes.

| Chain | Buses found: all saved sequences | Buses found: earlier 512 limit | CG minutes |
| ----- | ----- | ----- | ----- |
| 1 | 15 | 18 | 156.3 |
| 2 | 15 | 17 | 122.7 |
| 3 | 15 | 17 | 33.1 |
| 4 | 15 | 18 | 64.4 |
| 5 | 15 | 17 | 52.3 |
| 6 | 15 | 18 | 64.0 |

These are covering runs with 240 kWh batteries, 240 kW charging, no shared charger capacity and no return-SOC floor. Duplicate-trip removal has not been validated. CG time includes inherited-route import but excludes original graph construction. Earlier 512-route entries are timed incumbents; later inherited pools differ, so this table alone does not isolate a single code change. [All 37 results and exact proof scopes](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/queue_recovery_20260912/status_20260913T063519Z/README.md).

## 

## **Completed comparison: count the work behind each warm start**

**Completed results — 13 September, 20:49 EDT.** All 24 fresh CG runs converged before their cumulative allowance expired. All 24 fresh MIPs are complete: six match the target, four prove that their saved pool needs more buses, and fourteen leave the fleet gap open. All 24 matched warm-pool MIPs meet their targets. The table below retains selected k=5 examples; “converged” means no route below reduced cost −0.0001 in the tested graph. More CG allowance alone did not reproduce the warm pools’ integer results.

| Chain / target | Cumulative CG allowance | Fresh CG time | Fresh integer fleet | Warm integer fleet |
| ----- | ----- | ----- | ----- | ----- |
| Chain 3 / 5 | 51.0 min | 5.1 min; converged | 5; proved in pool | 5; proved in pool |
| Chain 5 / 5 | 110.0 min | 8.0 min; converged | 6; proved in pool | 5; proved in pool |
| Chain 6 / 5 | 15.0 min | 6.1 min; converged | 5; proved in pool | 5; proved in pool |

**Why chain 5 misses:** CG converged in 8.0 minutes, with fractional route weight 5 and weighted objective 500,276.192. Gurobi then proved that its 14,175-column pool needs 6 buses, in 36 seconds. The warm pool has 16,870 columns and supports 5 buses. More MIP time on the unchanged fresh pool cannot recover five; different columns are needed. An LP pricing certificate does not guarantee a good integer column pool.

At k=8, fresh chains 1, 4 and 5 prove that their pools require 9 buses; their warm pools use 8\. Fresh chain 6 matches 8\. These completed proofs identify a limitation of the fresh column pools.

Fourteen fresh MIPs retain an open fleet gap after one hour. At k=8, chains 2 and 3 have 9 buses with a bound of 8\. At k=10, chain 2 has 12 buses and the other five chains have 11, all with a bound of 10\. At k=15, chains 1–6 have 18, 17, 18, 19, 16 and 20 buses respectively, all with a bound of 15\. These gaps do not prove that the target is absent from those pools.

These are baseline runs without shared station capacity or a terminal-SOC floor; individual-route replay passes. Historical code and hardware variation limits timing comparisons. [Completed comparison and source evidence, 20:49 EDT.](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/cumulative_budget_20260913/status_20260914T004739Z/README.md)

Launched 13 September, verified 14:20 EDT. A warm k=8 run uses columns built at k=2,…,7. We now give a fresh k=8 run the sum of those CG times plus the k=8 CG time. The same test covers all six chains at k=5,8,10,15. Fresh starts use single-trip routes, with no inherited solution columns.

## **Fresh CG budget in hours**—measured accumulated warm-chain time, not a prediction of how long the fresh run will need:

| Chain | Target 5 | Target 8 | Target 10 | Target 15 |
| ----- | ----- | ----- | ----- | ----- |
| 1 | 3.39 | 8.94 | 9.68 | 15.87 |
| 2 | 0.57 | 10.16 | 10.97 | 16.09 |
| 3 | 0.85 | 6.30 | 14.77 | 16.59 |
| 4 | 0.34 | 6.45 | 11.94 | 15.26 |
| 5 | 1.83 | 10.51 | 20.16 | 23.14 |
| 6 | 0.25 | 1.74 | 5.89 | 10.54 |

## The first allowance includes graph loading, import and CG. A second allowance also credits construction of the smaller graphs; the target graph is common to both methods. Unrecorded historical serialization/setup costs remain a limitation. If fresh CG certifies early, it stops; if it hits the first time limit, it resumes its own checkpoint for the larger allowance.

## Every distinct fresh pool and each original warm target pool get the same one-hour, two-stage MIP. Intermediate warm-chain MIPs are excluded from the budget because they supplied no inherited columns. The comparison separates LP certificates, actual buses, finite-pool proof gaps, elapsed time and measured CPU use. Historical code and hardware varied, so this tests accumulated-budget performance rather than isolating a pure code speedup. [Exact method, inputs, jobs and results](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/cumulative_budget_20260913/README.md).

## **Controlled tests: which changes help?**

**These three changes are already implemented and tested on the cluster.** Indexed replay builds a lookup for graph connections, so checking each saved sequence avoids repeated full scans. The LP change skips constructing a matrix that the Gurobi path did not use. Full inheritance supplies more previous-run routes to CG and the final MIP.

All 24 paired allocations are complete: three frozen inputs, both execution orders, one change per comparison. Timings include loading, import and CG, excluding the final MIP. Do not add the percentage reductions.

**Capacity pricing is a separate workstream.** Its faster charging-window lookup and station-specific tariff accounting fix are implemented and tested locally. Three real-data comparison pairs still hit their three-hour limits without a pricing certificate, so we have not demonstrated faster convergence there. The next step is to retain distinct charging schedules when inheriting capacity-constrained routes, then test the combined settings. Updating only changed LP entries and improving stalled route generation remain proposals.

| Change tested | Reduction in total CG time | What else changed? |
| ----- | ----- | ----- |
| Indexed replay, same 512 routes | 12.4–18.1% | Same imported pool and certified LP endpoint |
| Remove unused LP setup, same full pool | 9.3–14.6% | Same imported pool and certified LP endpoint |
| Reuse all eligible saved trip sequences, removing the 512-sequence limit | 40.6–64.5% | More columns; better integer fleets |

For the implementation-only comparisons, imported sequence order and pool hashes, iteration counts, final column counts and certified weighted LP objectives agree. These are descriptive results on three inputs, not a population-wide speed guarantee.

| Input | CG iterations: 512 → full | MIP buses: 512 → full |
| ----- | ----- | ----- |
| Chain 1, k=8 | 950 → 177 | 9 proved → 8 proved |
| Chain 4, k=10 | 945 → 222 | 11 proved → 10 proved |
| Chain 3, k=15 | 891 → 280 | 17 timed → 15 proved |

Both execution orders give these fleet outcomes. All proof labels concern the supplied route pool. For chain 1 k8 and chain 4 k10, the smaller pools provably require 9 and 11 buses, while the richer pools use 8 and 10 at the same certified weighted LP optimum. This directly demonstrates a restricted-column bottleneck. The 17-bus result is unproved, so we cannot rule out 15 buses in that smaller pool.

Original arc scanning did not finish importing the full pool within two hours in any of six runs; no CG iteration completed and the MIPs were intentionally skipped. Indexed full-pool CG finished in 9.1–20.8 minutes in their paired counterparts. These capped runs are not execution failures. [Comparison details, paired timings and hashes](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/controlled_comparison_20260913/status_20260913T063519Z/README.md).

## 

## **Charging-start fee: 5 versus 0**

**All 36 MIPs finished; 35 match their target fleet.** All 36 selected solutions pass individual-route physical replay; 35 have finite-pool fleet proofs. Chain 1 k15 with fee 0 found 16 buses with a pool bound of 15\. CG has 34 certificates; both chain 1 k15 arms reached their two-hour budgets with negative reduced costs remaining.

| Target k | Fee 5 matches (of 6\) | Fee 0 matches (of 6\) |
| ----- | ----- | ----- |
| 5 | 6 | 6 |
| 10 | 6 | 6 |
| 15 | 6 | 5 |

The 16-bus result does not establish a physical need for another bus: the fee-5 schedule uses 15 buses and remains feasible with its start fee set to zero. Its routes may be absent from the fee-0 pool. The fee-0 MIP reached its time limits without closing the 15–16 fleet gap. Both fees start from the same frozen trip sequences, with charging reoptimized. Chain 1 k15 uses its frozen k14 pool; the other 17 inputs use same-k pools. Each run gets two hours of CG and one hour of MIP. [All paired outcomes and stopping reasons](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_charge_start_fee_20260913/status_20260913T083655Z/README.md).

**Completed k5 detail:** every row uses five buses under both fees. Arrows mean fee 5 → fee 0\.

| Chain (k=5) | Charging starts | Electricity cost | End energy (kWh) |
| ----- | ----- | ----- | ----- |
| 1 | 16 → 49 | 150.60 → 145.37 | 38.60 → 11.07 |
| 2 | 10 → 35 | 143.42 → 113.85 | 54.51 → 41.95 |
| 3 | 7 → 27 | 91.56 → 89.11 | 41.95 → 17.17 |
| 4 | 12 → 51 | 166.52 → 151.77 | 43.91 → 11.11 |
| 5 | 17 → 60 | 213.90 → 191.17 | 33.80 → 7.45 |
| 6 | 7 → 34 | 100.93 → 98.14 | 36.81 → 10.86 |

Across all 18 pairs, charging starts increased 2.0–4.9 times (median 3.0). Fee-0 CG took a median 2.9 times as long across 17 certified pairs (range 1.2–15.0); concurrent host load also affects timing. Return energy differs, so electricity differences are not savings at equal energy.

## 

## **GIRO electricity costs with no start fee**

**All six fee/tariff comparisons are complete** (checked 01:41 EDT). The table shows fee 0, electricity only. All schedules use five buses. Optimized schedules return at least the original GIRO total of 280.7833 kWh.

| Price peak | Original GIRO | Fixed duties, charging optimized | Joint route pool |
| ----- | ----- | ----- | ----- |
| 08:00 | 230.29–230.98 | 128.29 | 128.29 |
| 12:00 | 289.59–290.60 | 164.23 | 164.23 |
| 18:00 | 223.45–223.72 | 95.29 | 95.29 |

Optimizing charging on the fixed duties reduces electricity cost by about 43–57% versus the repriced original. The joint pool finds the same cost here: this experiment shows a gain from changing charging, with no additional gain from regrouping trips. GIRO has 52 charging starts; the zero-fee optimized schedules have 46, 43 and 42 starts respectively.

This separate cohort has 62 trips, 240 kWh batteries and 350 kW charging, without shared-station capacity constraints; the six chains use 240 kW. Optimized returning energy is 281.17, 281.17 and 282.90 kWh. Original costs are intervals because the exact within-window power trace is unavailable. Physical replay, trip coverage, returning-energy checks and completion-file hashes passed. The joint MIPs prove five buses within the supplied pools; charging gaps in the grid objective are at most 0.0088%. The table reports continuous replay costs, not a continuous-model optimality proof.

[Both fees: electricity, starts, total cost and returning energy](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_charge_start_fee_20260913/status_20260913T053436Z/giro_costs.csv). [Experiment settings and job records](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/zero_charge_start_fee_20260913/README.md).

## **GIRO comparison: which constraints are included?**

**We have not yet matched all documented GIRO rules in one experiment.** The six-chain k=15 result uses simplified physics. The original Notes, schedule PDFs and workbooks support the distinctions below; the source audit covers all eight recovered attachments.

| Partille requirement | Six-chain baseline | Stricter tests completed |
| ----- | ----- | ----- |
| Usable battery: 236.44 kWh for 18E1; about 239.01 for 18E2. Separate compatible work by group. | 240 kWh, homogeneous fleet | Actual group profiles in small diagnostics |
| SOC always at least 15%; original tasks start full | Full start, zero reserve | 15% reserve in new-physics k1–3 diagnostics |
| PARX depot 60 kW; remote power varies with SOC, roughly 120–371.5 kW | 240 kW everywhere | Depot60 and nonlinear curves in separate pilots |
| Charger counts: 2190L=1, 4808=1, 3127L=2, 7880C=1, JON\_A=1; PARX unlimited | No shared limits | Capacity pilots; original GIRO overlaps fit these counts |
| 65% is a recharge target; no documented 65% end-of-duty floor | No terminal floor | k5 cost test matches observed aggregate return energy |
| Charging setup/minimum duration, idle draw, route-specific layovers | Not all enforced | Some physics/setup included; full operational checks remain |
| Departure-platform blocking, FIFO and time-dependent directional deadheads | Not established | Still need implementation or explicit exclusion from the academic model |

**The reserve matters more than the small battery approximation.** With a 15% floor, approximately 201–203 kWh is available above reserve, versus 240 kWh in the baseline. A shifted zero-SOC convention is equivalent only if we also transform the starting energy and charging-curve thresholds.

**Return SOC correction:** all 42 literal Partille tasks finish below 65% in the original workbook (15.096–61.692%). The 65% entry applies to Recharge, not the end of a duty. Our k5 aggregate-energy constraint is a research comparison condition; it permits energy to be redistributed between buses. A per-duty return-energy comparison is a useful next sensitivity.

Further source details matter: route21 needs at least four minutes and 10% of trip duration as layover; 2190L/JON\_A charging can block departures, and 4808 requires FIFO. Opportunity charging has a minimum-duration convention and 18E2 setup of 45 seconds. Frölunda is a separate family, including 358.68-kWh vehicles, KEX60, a 60% recharge target and a 20% SOC floor for duties longer than 20 hours. Full crew rules, feeder limits and exact power/interpolation conventions are not supplied. [Source matrix, page/sheet references and remaining unknowns](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/model_fairness_audit_20260913/giro_requirements_audit.md).

## **What the stricter small tests actually achieved**

These eight diagnostics combine actual Partille battery profiles, 15% reserve, nonlinear remote charging, PARX60 and shared charger counts. “Fewer/more trips” identifies the duty-selection subsets within each vehicle group. Green matches the target; red uses extra buses.

| Group and subset | Target buses | Trips | Buses found |
| ----- | ----- | ----- | ----- |
| 18E1, fewer trips | 2 | 23 | 3 |
| 18E1, fewer trips | 3 | 35 | 3 |
| 18E1, more trips | 2 | 34 | 2 |
| 18E1, more trips | 3 | 51 | 5 |
| 18E2, fewer trips | 2 | 22 | 2 |
| 18E2, fewer trips | 3 | 37 | 4 |
| 18E2, more trips | 2 | 104 | 4 |
| 18E2, more trips | 3 | 150 | 18 |

**All eight fleets are proved only within their generated pools. None has a full-model pricing certificate.** Charging is restricted to full windows and pricing has search guards. The 18-bus result therefore does not prove that the operation needs 18 buses. Duplicate-service repair and full operational replay remain incomplete. These are not the six baseline chains.

A separate exact-event test of duty13408 uses one bus with PARX60 and charger counts: CG certified in 2.8 minutes, fleet proved in pool, and route/capacity replay passed. It still uses the baseline battery and zero reserve. [Completed constraint combinations and algorithm status](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/model_fairness_audit_20260913/constraint_results_audit.md).

## **How to interpret progress over time**

| Period or comparison | What changed | What the evidence shows |
| ----- | ----- | ----- |
| April–May | Heuristic DP, covering, older 300-kWh/300-kW assumptions; different seeds and inputs | April175 trips: 12 buses. May193 trips: 11–12 without GIRO seeds, 10 with them. Different datasets; incomplete proof/validation. |
| July–August | Safer SOC handling, exact expanded/event pricing, delayed charging, replay and saved certificates | Stronger correctness and LP proof support; more than a memory optimization. June work is not documented by the available git history. |
| 9 September: same175-trip pool, 240/240, same one-hour MIP budget | Only final trip rows change: covering versus partitioning | Cover: 10 buses, bound10. Partition: 34 buses, bound10, unproved. Cover has31 overcovered trips; operational removal still needs validation. |
| 12–13 September | Indexed import, less LP setup, full inheritance | All six baseline chains reach15. Controlled richer-pool results improve9→8,11→10,17→15. |
| Stricter GIRO studies | Reserve, depot speed, curves and counts tested separately and in small combinations | The k1–3 results above; no completed chain ladder with every documented constraint. |

These are different experiments, not a controlled curve of performance as constraints accumulate. Reserve, capacity and depot limits tighten feasibility; the real remote charging curve can be faster at low SOC and slower near full SOC. We should not attribute the whole history to battery size or an algorithm regression.

## **Next implementation and experiments**

1. **Verify the model first:** replay all original Partille duty variants under a frozen specification, testing each added rule separately. Compare observed return energy per duty as a declared sensitivity; do not impose an invented 65% terminal rule.  
2. **Bring full inheritance to capacity pricing:** retain different charging times for routes serving the same trips. The dedicated capacity drivers already distinguish occupancy; the generic baseline loader does not. Validate pricing, MIP accounting and shared-capacity replay together, then repeat the same k2/k3 inputs with one changed condition per arm.  
3. **Measure the remaining bottlenecks:** use the certified k1 capacity case for a controlled charging-window speed comparison; profile pricing and the large graph constructor. Updating only changed LP entries and alternate-dual route generation remain proposals.

This audit submits no new jobs. The launch plan will retain independent default-partition parallelism and genuine previous-k dependencies. [Full audit, definitions, experiment sources and next-test design](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/model_fairness_audit_20260913/README.md).

Figures remain in [Figures with explanations](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.ts4vwph3s99i) and [CG curves and bus schedules](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.h5h2ivyiprly).

## **Historical appendix — earlier runs and queue snapshots**

These tables preserve earlier experiments for comparison. Their missing cells and queued-job descriptions are dated history; current completed results and the GIRO assumptions audit are above.

## **Historical chain table — before the 12 September repairs**

**The dashes below are old interruptions, not unfinished jobs today.** The repaired runs now match every target from k=7–15 in chain 1, k=9–15 in chain 2, and k=10–15 in chain 4\. In particular, all three now use 10 buses at k=10, with CG times of 17.6, 16.4 and 18.8 minutes respectively. Each has a CG certificate and a fleet proof within its saved pool.

The table below preserves the earlier snapshot: each cell is the integer fleet found then; green matched the target, red used extra buses, and a dash meant interrupted CG. Use the current k=15 table above for today’s result.

| Target buses | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 2 | 2 | 2 | 2 | 2 | 3 | 2 |
| 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| 4 | 4 | 4 | 4 | 4 | 4 | 4 |
| 5 | 5 | 5 | 5 | 5 | 5 | 5 |
| 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| 7 | — | 7 | 7 | 7 | 7 | 7 |
| 8 | — | 8 | 8 | 8 | 8 | 8 |
| 9 | — | — | 9 | 9 | 9 | 9 |
| 10 | — | — | 10 | — | 10 | 10 |

**This table preserves the earlier full-pool runs. The separate 512-route extension is shown below as a comparison. For the repaired full-pool runs through k=15, use the latest table at the top; an old dash here does not mean the case is still blocked.**

k=2 starts from single-trip routes; later sizes reuse validated previous-k columns. Chain 3 k=10 also includes routes from a fresh solver solution. These are pool proofs. Chain 2 k10 (71 buses) had zero pricing iterations and no CG certificate. Individual route replay passed; duplicate-trip removal and shared charger capacity are not established by this table.

## 

## **Fresh starts on the same chains**

Each size starts independently. Both tables use covering and 240-kWh batteries with 240-kW charging. Fresh fleet proof status varies by cell.

| Target buses | Chain 1 | Chain 2 | Chain 3 | Chain 4 | Chain 5 | Chain 6 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5 | 5 | 5 | 5 | 5 | 6 | 5 |
| 8 | 9 | 9 | 9 | 9 | 9 | 8 |
| 10 | 11 | 11 | 11 | 11 | 11 | 11 |
| 15 | 18 | 17 | 18 | 19 | 16 | 19 |

Inherited columns improve the available integer pool: chain 3 at k=8 improves from a proved nine-bus pool optimum to eight; chain 5 at k=5 improves from a proved six-bus pool optimum to five.

## **Earlier bottlenecks and what changed**

**Saved-route checking: the earlier workaround, now superseded by indexed full inheritance.** Before starting CG on a larger trip set, we check that routes saved from the previous size are still valid. In chain 1 at k=7, chain 2 at k=9 and chain 4 at k=10, this checking used up the available time. The first workaround checked at most 512 saved routes for 15 minutes, then started CG. The current recovery instead uses an index to check all inherited sequences efficiently; its results are at the top. That earlier bounded experiment found 8, 10 and 11 buses, respectively; only the first and third were proved minimums within those saved pools. Using fewer saved routes may affect the final integer solution, so this is a separate experiment.

**Charger limits made the search for new routes very slow.** In one three-bus test, the LP solved in 0.006 seconds, but one search for a new route took 7.15 hours. The run ran out of time before it could prove that no improving route remained. Its 16-bus integer solution uses only the routes found so far; it does not show that 16 buses are necessary. The timing figures explain these two different delays.

| Case | Baseline | PARX 60 kW | Station capacity | Both |
| ----- | ----- | ----- | ----- | ----- |
| k=1, two duties tested | 1 each | 1 each | 1 each | 1 each |
| k=2, 23 trips | 2 | 2 | 3 | 3 |
| k=3, 35 trips | 3 | 3 | 16 | 16 |

Sixteen is the result in a small generated pool, not a proved physical requirement. Capacity-constrained schedules passed station sweeps. This pilot has no 65% return-SOC floor and is separate from the six baseline chains.

## 

## **Historical extension: at most 512 inherited sequences**

16:48 EDT collection: 66/87 original CG certificates, plus the certified chain 2 k14 retry. There are now 69 MIP results under the original extension campaign (34 warm, 35 component), plus the recovered chain 2 k14 result described below. The table below retains the earlier 34 warm results; the new k14 result is 16 buses, pool bound 14, fleet unproved, CG 159.5 minutes, weighted LP 1,400,558.356.

**Historical 512-limit runs.** At most 512 saved sequences were selected for replay before CG. All CG results below except chain 4 k=15 have pricing certificates. Times include preparation. Chain 4 k=15 has only a restricted LP objective, not a certified full-model lower bound.

**Buses:** the integer fleet actually found. **Proved fleet lower bound (pool):** Gurobi’s lower bound when restricted to that run’s saved routes. **Smallest fleet in pool proved?** Yes means the integer search established that no smaller fleet can be built from those routes; No means that question remained open when the search stopped.

**Example:** 16 buses found with lower bound 14 leaves 14, 15 or 16 possible as the pool optimum. Eight buses found with lower bound eight proves the pool optimum is eight. A different pool may allow fewer buses. This MIP bound is separate from the CG LP objective, which includes bus, electricity and charging-start costs.

| Chain and target | Buses | Proved fleet lower bound (pool) | Smallest fleet in pool proved? | CG minutes | Weighted LP objective |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Chain 1, k=07 | 8 | 8 | Yes | 53.6 | 700313.457 |
| Chain 1, k=08 | 9 | 9 | Yes | 49.5 | 800383.688 |
| Chain 1, k=09 | 10 | 9 | No | 67.9 | 900453.496 |
| Chain 1, k=10 | 11 | 11 | Yes | 91.2 | 1000511.205 |
| Chain 1, k=11 | 13 | 11 | No | 114.1 | 1100537.061 |
| Chain 1, k=12 | 13 | 12 | No | 95.6 | 1200608.942 |
| Chain 1, k=13 | 15 | 13 | No | 152.2 | 1300648.369 |
| Chain 1, k=14 | 16 | 14 | No | 147.9 | 1400650.097 |
| Chain 2, k=09 | 10 | 9 | No | 64.0 | 900301.353 |
| Chain 2, k=10 | 11 | 10 | No | 55.7 | 1000371.146 |
| Chain 2, k=11 | 12 | 11 | No | 65.7 | 1100404.250 |
| Chain 2, k=12 | 13 | 12 | No | 85.6 | 1200441.362 |
| Chain 2, k=13 | 15 | 13 | No | 98.4 | 1300493.195 |
| Chain 3, k=11 | 12 | 11 | No | 27.7 | 1100375.859 |
| Chain 3, k=12 | 14 | 12 | No | 32.8 | 1200427.901 |
| Chain 3, k=13 | 14 | 13 | No | 52.6 | 1300474.559 |
| Chain 3, k=14 | 15 | 14 | No | 57.9 | 1400485.597 |
| Chain 3, k=15 | 17 | 15 | No | 59.7 | 1500507.420 |
| Chain 4, k=10 | 11 | 11 | Yes | 42.0 | 1000426.271 |
| Chain 4, k=11 | 12 | 11 | No | 59.4 | 1100462.865 |
| Chain 4, k=12 | 13 | 12 | No | 68.2 | 1200505.601 |
| Chain 4, k=13 | 15 | 13 | No | 88.6 | 1300530.332 |
| Chain 4, k=14 | 18 | 14 | No | 204.6 | 1400590.723 |
| Chain 4, k=15 | 18 | 15 | No | 239.1 | 1500637.979 |
| Chain 5, k=11 | 12 | 11 | No | 70.5 | 1100522.228 |
| Chain 5, k=12 | 13 | 12 | No | 109.2 | 1200576.644 |
| Chain 5, k=13 | 14 | 13 | No | 134.6 | 1300625.367 |
| Chain 5, k=14 | 16 | 14 | No | 140.1 | 1400639.061 |
| Chain 5, k=15 | 17 | 15 | No | 209.7 | 1500669.159 |
| Chain 6, k=11 | 11 | 11 | Yes | 32.5 | 1100399.029 |
| Chain 6, k=12 | 13 | 12 | No | 64.6 | 1200464.548 |
| Chain 6, k=13 | 15 | 13 | No | 117.1 | 1300532.041 |
| Chain 6, k=14 | 16 | 14 | No | 136.9 | 1400578.400 |
| Chain 6, k=15 | 18 | 15 | No | 176.3 | 1500606.338 |

Pool fleet bounds concern saved columns only. A proved fleet above target requires different columns to reach target. An unproved gap does not distinguish incomplete integer search from missing useful columns.

### **Historical outcomes under the 512 limit**

**Chain 1 at k=14:** CG finished with a pricing certificate after 147.9 minutes and 1,301 iterations. Its weighted LP objective is 1,400,650.097. The completed MIP found 16 buses, with a saved-pool fleet bound of 14; the fleet is unproved. Individual routes passed replay; duplicate removal and shared capacity remain unverified.

**Chain 4 at k=15:** the MIP found 18 buses with a saved-pool fleet bound of 15; neither fleet optimality nor full-model LP optimality is proved. CG had stopped at its time limit after 239.1 minutes, with last reduced cost −0.1998 against tolerance 0.0001. Individual routes passed replay; duplicate removal and shared capacity remain unverified.

**Chain 2 at k=14, updated 12 September:** CG reached a pricing certificate after 159.5 minutes and 1,484 iterations, with weighted LP objective 1,400,558.356. The subsequent one-hour MIP found **16 buses** and a **pool fleet bound of 14**; it did not prove the best fleet. Individual-route replay passed, but duplicate-trip removal and shared-station capacity were not validated. This is the earlier bounded512 warm start. The new full-pool chain is a separate experiment. That bounded k15 follow-up has since finished: 17 buses, pool lower bound 15, fleet unproved. The separate repaired full-pool run found 15 and proved the pool fleet optimum; neither job is still queued.

## **Decomposition of 32 duties**

| Grouping | Four component fleets | Combined buses | Target |
| ----- | ----- | ----- | ----- |
| join01 | 9 \+ 9 \+ 9 \+ 8 | 35 | 32 |
| join02 | 9 \+ 9 \+ 9 \+ 8 | 35 | 32 |
| join03 | 9 \+ 9 \+ 8 \+ 10 | 36 | 32 |
| join08 | 9 \+ 9 \+ 10 \+ 9 | 37 | 32 |
| join09 | 8 \+ 9 \+ 9 \+ 9 | 35 | 32 |

These saved constructions combine disjoint trip groups chosen using GIRO duty membership; no GIRO route columns were supplied. Component routes passed replay. Shared charger capacity and duplicate removal are not certified. **All five joined-parent attempts timed out before producing a parent CG result.** Their saved decomposed constructions remain available, but there is no joint improvement or parent optimality proof.

## **Completed speed comparisons**

| Case | Reference minutes | Optimized minutes | Observation |
| ----- | ----- | ----- | ----- |
| d00\_g0 | 22.3 | 23.6 | 5.8% slower |
| d00\_g1 | 52.2 | 52.1 | Essentially unchanged |
| w1\_k08 | 41.1 | 34.4 | 16.2% less time |
| w1\_k08\_repeat | 31.1 | 24.1 | 22.4% less time |
| w4\_k11 | 58.8 | 49.1 | 16.5% less time |
| w6\_k12 | 67.2 | 54.6 | 18.6% less time |

All six pairs reached matching certified LP objectives. Four warm comparisons used 16–22% less time; inherited-route checking fell from 366–447 seconds to 5–7 seconds. Common graph-cache preparation is excluded. CG iteration counts differed, so the total savings are descriptive, not an isolated causal estimate. Fresh comparisons show no speed improvement. No integer improvement is claimed from these timing tests.

## **Capacity comparisons at the time limit**

| Capacity case | Iterations ref / opt | Final restricted LP objective | Fractional route weight | Pricing certified ref / opt |
| ----- | ----- | ----- | ----- | ----- |
| cap\_k1 | 13 / 12 | 100056.952000 | 1.0 | No / No |
| cap\_k1\_combined\_peak12 | 34 / 37 | 100074.693539 | 1.0 | No / No |
| cap\_k2 | 4 / 4 | 300094.976000 | 3.0 | No / No |

Every arm stopped after three hours. Both methods reached the same restricted-LP objective in each case; none proved that no improving route remained. Fractional route weight is not an integer fleet result. These tests do not demonstrate a convergence speedup.

**Cluster queue — 12 September**

At 16:57 EDT, 17 EVSP–DR jobs and 9 separate V2G jobs were running, with no invalid dependencies. The first new full-pool CG jobs for chain 3 k11 and chain 4 k10 completed with pricing certificates, and their successors are releasing automatically. The table below is the earlier 16:48 snapshot.

| Work | Running | Waiting for input data |
| ----- | ----- | ----- |
| Full-pool warm chains through k=15 | 6 | 68 |
| Decomposition: shared graph preparation | 2 | 22 |
| Earlier bounded warm chains | 2 | 2 |
| Recovered MIPs with ready CG outputs | 4 | 0 |
| Separate V2G project | 9 | 0 |

We removed 43 obsolete queued entries and recovered nine MIPs whose data were already ready. Five have finished: each found 8 buses for its 8-bus subset, proved the fleet optimal within its saved column pool, and passed individual-route replay.

The remaining waits are genuine: the next k needs the previous k’s routes; a MIP needs its CG output; the larger decomposition runs need their component solutions and shared graph. No dependency is marked impossible.

On Unicorn, run **\~/ladder-lite/drq** for this grouped view, or add **\--details** to see each prerequisite. [Job map and recovery evidence](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/queue_recovery_20260912/README.md).

**Methods and evidence.** CG route cost \= 100,000 \+ electricity \+ 5 per charging start. MIP stage 1 minimizes buses. Stage 2 constrains buses ≤ the validated stage-1 incumbent and minimizes electricity plus start fees.

A pricing certificate establishes the LP result only in its represented graph and tolerance. A MIP proof concerns its saved columns. Fleet attainment, physical replay and shared-capacity validation are separate checks.

[Experiment register](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/research_register) · [Dated results and source tables](https://github.com/ndandnd/EVSP-DR/blob/codex/parallel-research-20260911/outputs/overnight_extension_20260912/RESULTS_20260912T193423Z.md) · [Experiment settings](https://github.com/ndandnd/EVSP-DR/tree/codex/parallel-research-20260911/outputs/overnight_extension_20260912) · [Historical document and figures](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit) · [Local prototype tests and charging-cost bug](https://github.com/ndandnd/EVSP-DR/blob/b142f360f8a9018c2f50e3419f0dddcaf87a489d/outputs/algorithm_benchmarks_20260912/README.md)

**Storage, 12 September:** 156 cold files archived losslessly; 133.74 GB freed. Active research and V2G data preserved. [Restore instructions](https://github.com/ndandnd/EVSP-DR/tree/b1129bf4/outputs/storage_cleanup_20260912).

**Blocked decomposition case d00\_g3:** the original run timed out before CG, while constructing the graph. Its MIP remains blocked. A separate diagnostic crashed after 366 seconds (SIGSEGV), before its watchdog limit. Partial samples show JSON serialization while adding charging arcs; A local timer-only test also hung, so the asynchronous sampler has been retired. The exact cluster crash remains unresolved. [Diagnostic evidence](https://github.com/ndandnd/EVSP-DR/blob/57f663eb/outputs/d00_g3_startup_review_20260912/TERMINAL_REVIEW.md). No long retry has been launched.