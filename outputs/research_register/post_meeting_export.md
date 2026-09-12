# **After the meeting — 10 September**

## **How charging moves when the price changes**

These are the schedules behind the three-way cost comparison: original GIRO, charging optimized with the GIRO trip sequences fixed, and joint route \+ charging optimization. All three comparisons use the same 62 trips and five buses, 240 kWh batteries and 350 kW charging, with no shared station-capacity constraints.

The three panels below use price peaks at 08:00, 12:00 and 18:00. The orange and green curves move charging away from expensive hours. The original GIRO charging windows and energy are known, but the exact power within each window is not; its shaded range and dashed reference line make that uncertainty explicit.

**Important qualification:** GIRO ends with about 281 kWh of battery energy, while the optimized schedules end with about 8–13 kWh. The cost savings therefore include using more of the initial battery energy, not only shifting charging. An equal-terminal-energy comparison is needed to isolate the value of timing changes.

![][image1]

**Costs behind these schedules** (electricity plus 5 per charging start; bus purchase cost excluded):

| Price peak | Original GIRO repriced | Fixed GIRO duties | Joint optimization |
| ----- | ----- | ----- | ----- |
| 08:00 | 490.29–490.98 | 267.34 | 242.55 |
| 12:00 | 549.59–550.60 | 320.09 | 304.78 |
| 18:00 | 483.45–483.72 | 211.83 | 182.35 |

## **Can geography help us decompose the problem?**

The original 40-duty GIRO day contains two vehicle groups: route 21 has 14 duties and 203 trips; the local Partille services have 26 duties and 745 trips. They use different opportunity charging sites, but share the PARX depot.

The map joins historical GIRO passenger-trip endpoints with straight lines; these are not driven road paths. Squares mark charging locations using current OpenStreetMap stop positions. It covers 838 of 948 trips. PARX and 110 trips with unmapped endpoints are omitted pending coordinate verification. Background and coordinates: © OpenStreetMap contributors.

![][image2]

**What the graph check tells us:** an optimistic timetable screen finds 23,959 candidate trip connections with at most 30 minutes waiting after travel; 7,561 (31.56%) cross the two GIRO groups. All 948 trips remain connected when edge direction is ignored. This screen ignores battery and charging feasibility, so it does not establish that those connections are usable. It does show that geographic independence has not been proved.

**Next decomposition test:** generate routes within groups, import them into one global master, then allow cross-group pricing. Measure whether crossing routes improve fleet or charging cost. Keep shared charger constraints in the global master. A proof for the unrestricted LP requires pricing every permitted route family to the stated reduced-cost tolerance.

## **Why the dependent MIP stopped**

Job 740390 read the 35,495-column pool but selected Gurobi’s bundled restricted license. The log says “Model too large for size-limited license.” Earlier full-size runs used the cluster license, so there is no contradiction. This was a job-launch configuration error; it was not a failed search for an integer solution. Updating a script after submission does not update Slurm’s saved copy.

A separate saved-start serialization error affected two other runs after optimization. Their selected 9-bus and 11-bus solutions have now been reconstructed from the saved column journals and passed route validation without rerunning optimization. The replacement warm-k10 MIP reuses the saved CG pool and validates the production license before the real solve.

## **Controlled station-capacity and charging-speed tests**

Four matched arms: baseline; documented station counts only; PARX at 60 kW only; both changes. Opportunity charging remains 240 kW in these pilot arms. Battery, starting SOC and terminal requirements stay fixed across arms; the nonlinear opportunity-charging curve is a separate future experiment. Heden has two ports, the other four opportunity sites one each, and PARX is uncapped. MIPs exclude scaglione-compute-01.

The stochastic EVSP–V2G review can live in a separate research task. Share a short methods/evidence table with this project, especially the information structure, recourse policy, out-of-sample validation and consequences of approximate pricing for optimality certificates.

## **Charging windows by bus — noon price peak**

The top panel shows original GIRO charging windows. The middle keeps each GIRO trip sequence and optimizes charging by dynamic programming. The bottom shows the jointly optimized routes. Colors identify the two charging sites. The bottom rows are new route labels, not a one-to-one match to the original duty IDs. Trip driving and waiting are omitted here so the charging changes are easier to see. The terminal-energy qualification above still applies.

![][image3]

## **Launch status — 10 September, 22:08 EDT snapshot**

| Work | Jobs | Verified state |
| ----- | ----- | ----- |
| Warm chain 3, k=10 MIP | 772009 | 10 buses, bound 10; proved within the augmented pool. Charging optimization ended with a 14.39% gap. All selected routes passed physical replay. A final-file collision caused Slurm FAILED after optimization; the validated result was recovered without rerunning. |
| Capacity × depot-speed CG | 772080 | The live array throttle is now 50\. This pilot has 16 cases total. Future independent default-partition CG arrays request 50 concurrent tasks unless a documented resource or policy restriction requires less. |
| Dependent pilot MIPs | 773334 | Scaglione; two concurrent; 25-minute solver budget. Fleet first, then charging with fleet ≤ best validated incumbent. |
| Matched terminal-energy comparison | 778801 → 778802 | All three fixed-duty frontiers completed. All three joint MIPs stopped before optimization because older singleton columns lack terminal-energy metadata. Deterministic route replay reconstructs it; repair tests and real-pool checks pass. See the experiment register for recovery status. |
| Second inherited chain: random chain 5 | CG 779026–779035; MIPs 779057–779073 | At 22:08 EDT: targets k=2–6 use 3, 3, 4, 5 and 6 buses, respectively, proved within their saved pools. k=3–6 match their targets; k=2 remains unmatched. k=7 CG is running. |

All CPU jobs exclude **scaglione-compute-01** for GPU users. Held historical jobs remain held. The hourly monitor reports meaningful results, errors, and lost Unicorn access.

## **Fresh versus inherited columns**

The 10-bus result is **random chain 3, with 205 trips**. The easy ordering by trip count has 127 trips at k=10. Fresh CG starts each k from single-trip columns. Inherited CG imports the previous k’s columns and adds single-trip columns for the new trips; duals, LP bases and LP certificates are not inherited.

| CG start / random chain | Buses at k=5 | Buses at k=8 | Buses at k=10 |
| ----- | ----- | ----- | ----- |
| Fresh / chain 1 | 5 | 9 | 11 |
| Fresh / chain 3 | 5 | 9 | 11 (bound 10\) |
| Fresh / chain 5 | 6 | 9 | 11 |
| Inherited / chain 3 | 5 | 8 | 10 |

All results use set covering. Fleet is proved within each saved pool except fresh chain 3 at k=10. Inherited chain 3 matches every k from 2 through 10\. Its k=10 pool also includes a validated fresh-run covering start: solver-generated routes, not original GIRO routes. Shared charger capacity was outside these runs’ scope. Some selected trips are covered more than once; duplicate removal has not been validated.

**Clean evidence at k=8.** Both chain-3 runs reached LP objective 800,212.68747, fractional fleet 8, zero artificial coverage and the same reduced-cost stopping tolerance. The fresh pool was proved to require 9 buses; the inherited pool was proved to require 8\. Pool contents therefore limit integer quality in this case even after LP convergence. More MIP time on the unchanged fresh pool cannot produce eight buses.

**Runtime still needs work.** Warm k=10 took 292.57 minutes: 282.59 minutes importing and validating predecessor routes, then 9.98 minutes for the remaining work. Fresh CG took 48.94 minutes. Inheritance has not improved total runtime.

## **Second warm-chain replication**

Random chain 5 now repeats k=2 through 10 with the same previous-k inheritance and set-covering approach. CG jobs: 779026, 779027, 779028, 779030, 779031, 779032, 779033, 779034, 779035\. Dependent MIPs: 779057, 779059, 779061, 779063, 779065, 779067, 779069, 779071, 779073\.

First result, 17:42 EDT: k=2 (32 trips) finished CG in 100.12 seconds and 87 iterations. Weighted LP objective: 200,154.736; fractional route count: 2.000; no artificials; minimum reduced cost: −1.47×10⁻¹⁰ (tolerance 10⁻⁴). The 1,500-column MIP proved three buses and charging cost 128.992 optimal within that pool. All three routes passed physical replay; two trips are overcovered. Two buses remain unresolved in the full route model. This k=2 case starts from singletons; inheritance begins at k=3, which accepted all 1,500 columns in 120.24 seconds. By 22:08 EDT, k=3–6 had matched their targets with 3, 4, 5 and 6 buses; k=7 CG was running. The k=5 and k=6 results and import-time breakdowns are recorded below. MIPs remain sequential on Scaglione, with one hour for optimization and two scheduler hours; no external routes are injected. Evidence: monitor/20260910T214204Z\_changes.json.

## **Fair charging comparison — frontiers complete; MIP repair**

For the five-duty, 62-trip tariff cohort, both optimized alternatives now require **aggregate terminal energy ≥ 280.7833253 kWh**, the recorded GIRO total. Starting energy, tariff, charging power and other physics stay unchanged. This is a common minimum; actual ending energy will also be reported. We did not find a universal return-SOC policy establishing a 65% floor.

Array 778801 completed all three fixed-duty frontiers. Array 778802 failed before optimization because older singleton records lack terminal-energy metadata. The repair recomputes this energy from each saved route and checks existing metadata; it does not invent an SOC value. Retry provenance is recorded in the experiment register. A future joint result concerns the saved CG pool plus fixed-duty alternatives; the earlier CG certificate does not cover the new terminal-energy dual.

## **Completed capacity/speed pilot results — snapshot at 16:41 EDT**

| Cohort / completed arms | Integer buses | Charging cost in discrete model | Evidence |
| ----- | ----- | ----- | ----- |
| Duty 13408 / all four arms | 1 in each | 36.536 in each | CG certified; both MIP stages optimal in each saved pool. |
| Duty 13406 / baseline only | 1 | 56.952 | CG: 50 iterations, 11.0 minutes; both MIP stages optimal in the saved pool. |

Five of 16 pilot cells had completed at this snapshot. Duty 13406 covers all 14 trips exactly once and charges four times at Eketrägatan, avoiding PARX. These partial results do not yet establish the effect of depot power or shared station capacity. CG certificates concern the conservative event/SOC graph; station-overlap audits pass.

The pilot’s initial weighted-objective array 772082 was canceled; its completed cell remains a separate legacy control. Corrected two-stage code passes 62 targeted tests, including an unproved fleet incumbent, the ≤ fleet cap, and infeasible-result reporting. CG uses commit 7d38efd; MIP uses 9bf3f75.

**Evidence:** outputs/post\_meeting\_20260910/README.md links the warm-chain tables, terminal-energy experiment, P5 launch records and cluster snapshots. The register snapshot is monitor/20260911T020803Z.json (10 September, 22:08 EDT). The Experiment register tab gives the locations of the editable workbook and source evidence. No separate Astra literature-review chat was launched; only STOCHASTIC\_REVIEW\_HANDOFF.md was prepared for the user’s future task.

## **Chain 5, k=5: inherited columns recover the target**

Verified at 20:06 EDT, 10 September. The fresh and inherited runs use the same input hashes, 240 kWh / 240 kW, flat tariff and set covering. No external GIRO routes were added.

| Measurement | Verified result |
| ----- | ----- |
| Fresh pool | 14,175 columns; six buses proved necessary within that pool. |
| Inherited pool | 16,870 columns; five buses proved necessary within that pool, matching the GIRO target. |
| Final CG values | Weighted objective 500,276.192; fractional buses 5.000; zero artificials. |
| Why CG stopped | 148 iterations; minimum reduced cost −4.69×10⁻¹⁰ meets tolerance 10⁻⁴ in the conservative event/SOC model. |
| CG time | 62.13 minutes total: 59.12 importing and validating 13,670 predecessor columns; 3.01 for the remaining work. |
| Two-stage MIP | About 20 seconds. Fleet five; grid charging-related cost 307.456. Both stages have zero reported gap within the pool. |
| Physical checks | Five routes pass individual replay and cover all 102 trips. Six trips are overcovered; duplicate removal and shared charger capacity are not validated. |

**Interpretation:** another matched case where the inherited pool supports a better integer solution. Importing columns remains expensive. This does not establish a general speed improvement or a full-model integer proof.

Evidence: monitor/20260911T000630Z.json; MIP result SHA-256 7f9521afe16bcddb986d28a42d7a0e93a6c021d6b26aad2008b7020f423d766b. k=6 is running; no jobs were changed by this monitor check.

## **Chain 5, k=6: target matched**

| Measurement | Verified result — 22:08 EDT |
| ----- | ----- |
| Integer result | Six buses, fleet bound six, in a 25,366-column pool. Both MIP stages optimal within this pool. |
| CG certificate | 366 iterations; weighted objective 600,330.271027; fractional buses 6.000; zero artificials; minimum reduced cost −2.42×10⁻¹⁰ meets tolerance 10⁻⁴ in the conservative event/SOC model. |
| CG runtime | 101.81 minutes total; 91.08 importing and validating all 16,870 predecessor columns; 10.73 remaining. |
| MIP runtime and cost | 115.92 seconds. Stage 2 uses fleet ≤6; grid charging-related cost and bound both 383.336. |
| Physical checks | Six routes pass individual replay and cover all 139 trips. Seven trips are overcovered; duplicate removal and shared charger capacity are not validated. |

Evidence: monitor/20260911T020803Z.json; result SHA-256 e219e2bbf23679615cb12b164463de0312b19f23cbb87f7f2d81106cecf75763. This is a finite-pool integer proof, not a full-model fleet proof. k=7 is running. No jobs were changed by this check.