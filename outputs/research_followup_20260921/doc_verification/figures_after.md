# **Current research figures**

## **21 September Follow chain 1 as the target grows**

Blue is a fresh solve at the final target. Orange counts all preceding sequential CG work. The integer panel shows extra buses after the same one-hour MIP allowance. The LP curves coincide.

![][image1]

The right panel subtracts the common 100,000 × k fleet-cost term to show the charging-related part of the weighted LP objective. It is not a violation or reduced cost. All 24 paired objectives agree within 0.0000010617 cost units; certificates use reduced-cost tolerance 0.0001 on the conservative event grid. Graph construction, MIP and queue time are excluded from CG minutes. Baseline physics: 240 kWh / 240 kW, zero reserve, no shared capacity or terminal floor, start fee 5\.

[All six individual figures, exact table and certificate checks](https://github.com/ndandnd/EVSP-DR/blob/3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce/outputs/research_followup_20260921/chain_comparison/README.md)

## 

## **21 September The same comparison on all six chains**

Columns are chains; rows are CG time, extra integer buses and magnified LP cost. Points exist only at k=5,8,10,15. Faint integer segments show pool bound to incumbent, not uncertainty intervals.

![][image2]

Sequential CG spends more accumulated time and gives better integer route pools. Fresh CG reaches the same numerical LP optimum much sooner in this historical panel; hardware and code differ, so elapsed times are descriptive.

## **21 September Duty 13309 and actual algorithm counterparts**

Every panel below includes morning and afternoon. Fee0 and fee5 use the same 79-trip C6 k5 input and five-bus fleets. The fee0 route shares all 22 original trips; fee5 shares 11 and serves 10 other trips. Orange rings identify used chargers.

![][image3]

These are historical algorithm schedules, with zero reserve and constant 240 kW charging. They are not matched-physics savings comparisons. Both selected routes pass a capacity-only reduction to 239.01 kWh; other GIRO requirements were not added by that check. L labels are chronological passenger legs, M labels empty moves. 13722 is unlocated and placed schematically.

[Larger individual diagrams, charging clocks, original trip identities and complete itineraries](https://github.com/ndandnd/EVSP-DR/blob/3f2228605e0c9e1cc46140a3dbd03fd6c1f6e3ce/outputs/research_followup_20260921/duty13309/README.md)

## 

## **21 September Fractional solutions and integer routes**

Both methods certify the weighted LP in all 24 cases.

 Their one-hour pool MIPs match 6 of 24 GIRO targets from fresh pools and 24 of 24 from sequential pools. The upper panels count earlier smaller-instance CG work; the lower panels show extra integer buses.  
![][image4]  
Vertical segments run from the pool fleet bound to its incumbent; they are not statistical error bars. Fleet search receives up to 30 minutes; charging uses the time remaining within the one-hour total budget. Baseline: 240 kWh and 240 kW, covering, zero reserve, no shared capacity or terminal floor. Historical code and hardware differences prevent a causal runtime claim.  
[Editable values including exact LP objectives](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/figure1_paired_budget.csv) · [All five paper figures and full captions](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/RESULTS_PREVIEW.md)

## 

## **21 September Routes that complete an integer fleet**

Left: four fresh pools prove nine buses are needed, then reach eight after adding known sequential routes. C2 was open at nine before enrichment. Right: directed pricing generates useful routes without those known witnesses; the original pilot succeeds in three of four final MIPs.  
![][image5]  
The purple point is a later C1 follow-up that transfers its own dive incumbent to the final MIP. It is additional computation, not a fourth original-pilot success. These selected diagnostics establish a route-pool mechanism, not a general success rate. The new balanced comparison uses four cases, two seeds and paired control/treatment runs.  
[Witness values and proof lines](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/figure2_k8_witness.csv) · [Original pilot timing and results](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/figure2_k8_pilot.csv)

## 

## **21 September Longer integer searches still leave a gap**

Twelve 12-hour fleet searches on six fresh k15 pools return 16–19 buses. Every fleet bound remains 15\. More time has not found the target, but these searches do not prove that a 15-bus solution is absent.

![][image6]

Each segment runs from the saved-pool fleet bound to its incumbent. Plain and stronger-heuristic searches reuse the same six pools; they are not twelve independent instances. All charging stages also finished. [Editable values, proof lines and full-log paths](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/figure3_k15_12h.csv).

## 

## **21 September Which implementation changes helped**

Controlled tests compare indexed route replay, removal of unused LP setup, and full inheritance against the earlier 512-route limit. Points represent three cases tested in two execution orders.

![][image7]

Replay and setup changes preserve the pool and certified endpoint. Full inheritance changes the initial columns and CG trajectory; it saves time and improves the three tested fleets from 9 to 8, 11 to 10, and 17 to 15\. Times cover the target CG step, not all earlier chain work. Six earlier full-scan import timeouts remain in the source table and are excluded from speedup percentages. [All 24 allocations and settings](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/figure4_controlled_algorithms.csv).

## 

## **21 September Smaller graph storage and faster pricing**

On one 26-trip strict-physics case, packed graph storage gives faster construction, lower peak memory and faster pricing. All variants run on the same node and event lattice.

![][image8]

Five fixed dual vectors give identical minimum reduced costs; all 15 returned routes pass physical replay. There are no shared-capacity duals. This is one implementation benchmark, not five instances or a full-CG fleet certificate. Larger benchmarks are now running. [Exact metrics](https://github.com/ndandnd/EVSP-DR/blob/e07c877f0adc23bb8d64b1db1c3d104b0ceefc19/outputs/research_management_20260921/paper_results/figure5_packed_benchmark.csv).

## 

## **21 September What the charging-start fee changes**

All nine fee pairs now have validated schedules: three fixed five-bus assignments × three tariffs. Within each line, only the start fee changes. Trips, station paths, battery physics, charger limits and ending-energy floors stay fixed.

![][image9]

Adding the fee gives 8–22 fewer starts in every paired incumbent, while electricity cost rises. At a common fee of 5, total cost falls by 24.2–73.3 synthetic units. Twelve searches reach the 0.01% gap target; six time out with 0.75–3.83% gaps. Bounds establish strictly fewer starts at optimality in three of the nine restricted models; plotted counts need not be unique.

This is fixed-path charging optimization, not fresh CG. Each gap allows one charge within one tariff hour. “Original” means original trips with recovered station paths, not the recorded charging schedule. One raw witness missed a terminal floor by 0.028 Wh; a separately saved 6 ms charging extension passes the unchanged validator, with a recomputed upper bound and unchanged solver lower bound. [Editable tables, cost definitions and numerical repair](https://github.com/ndandnd/EVSP-DR/blob/d2a25746d160e0c181eca3c9ed09069f23fbeede/outputs/research_management_20260921/charging_fee_factorial/RESULTS.md) · [Gurobi logs](https://github.com/ndandnd/EVSP-DR/blob/d2a25746d160e0c181eca3c9ed09069f23fbeede/outputs/research_management_20260921/charging_fee_factorial/log_excerpts.md).

## 

## **21 September A day with three charging sites**

GIRO duty 13309 connects five areas: PARX, Partille centrum, Heden, Jons väg and Gamlestads Torg. The two panels follow the same bus before and after its midday depot stop. All 22 passenger trips remain; blue arrows show passenger legs, grey arrows show empty movements, and orange marks charging.

![][image10]

The bus leaves PARX at 05:19, returns at 10:42, charges there from 10:45 to 12:30, leaves again at 12:43, and finishes at 19:02. It also charges at Heden twice and Jons väg once. This example shows how short opportunity charges and a long depot recharge fit different parts of one day.

This is the recorded GIRO schedule, checked against the original workbook. It is not a new optimized result; no validated fee-0/fee-5 counterpart was established for this duty. Platform codes are grouped using the documented reference-area mapping; exact platforms, times and prepared trip IDs remain in the itinerary.

Full resolution: [graph](https://github.com/ndandnd/EVSP-DR/blob/343354a87389ee3f35848afed309e7d959b93001/outputs/week_20260921/complex_route_graphs/duty_13309_graph.png) · [complete day and itinerary (PDF)](https://github.com/ndandnd/EVSP-DR/blob/343354a87389ee3f35848afed309e7d959b93001/outputs/week_20260921/complex_route_graphs/duty_13309_daybook.pdf) · [editable tables and source checks](https://github.com/ndandnd/EVSP-DR/blob/343354a87389ee3f35848afed309e7d959b93001/outputs/week_20260921/complex_route_graphs/README.md).

## 

## **21 September — A bus day as a graph**

Five examples compare recorded GIRO duties with saved fee-0 and fee-5 schedules after charging reoptimization. Blue arrows are passenger trips; grey arrows leave or return to PARX; orange marks charging. The overview keeps approximate geographic positions. The detailed companion separates repeat visits and lists every wait, charge and movement in order.

![][image11]

Duty 13414 is the closest comparison: original and fee 5 serve the same 12 trips, with 9 versus 6 charging starts. Fee 0 serves 13 trips. All five buses together cover the same 62 trips in each arm; differing individual trip assignments mean these are not fee-only causal comparisons.

L1, L2, … indicate the order of passenger legs. Trip numbers in the detailed view are stable labels from our prepared full input, not GIRO journey numbers. The schedules use the corrected battery, charging power, reserve, return energy and five-bus capacity checks; known movement approximations remain listed below.

[All five comparisons and complete day sequences (PDF)](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/all_comparisons.pdf) · [Source files, editable tables and scope](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/README.md)

| Choose a bus comparison | Original charging starts | Fee 0 charging starts | Fee 5 charging starts |
| ----- | ----- | ----- | ----- |
| [13414 graph](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13414_graph.png) · [day sequence](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13414_itinerary.pdf) | 9 | 9 | 6 |
| [13403 graph](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13403_graph.png) · [day sequence](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13403_itinerary.pdf) | 12 | 8 | 5 |
| [13405 graph](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13405_graph.png) · [day sequence](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13405_itinerary.pdf) | 12 | 9 | 6 |
| [13401 graph](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13401_graph.png) · [day sequence](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13401_itinerary.pdf) | 10 | 8 | 6 |
| [13408 graph](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13408_graph.png) · [day sequence](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/duty_13408_itinerary.pdf) | 9 | 8 | 7 |

13414 best isolates the same passenger trips for original versus fee 5\. 13403 shows the largest displayed drop in starts, alongside an earlier depot return and changed trip assignments. The other examples show how the choice of trips also changes the day.

These are charging reoptimizations on saved trip sequences, not fresh CG under the corrected physics. [Validation](https://github.com/ndandnd/EVSP-DR/blob/0cf29322f12be23d83bc333473d282ba8812da05/outputs/week_20260921/spatial_schedule_graphs/extraction_validation.json) covers each five-bus fleet; full-day background traffic at chargers is outside this comparison.

## 

## **21 September Geography of the k5 example**

The bus in slide 12 repeatedly serves Eketrägatan and Merkuriusgatan, starting and ending at the Partille depot. Lines show connections, not road paths; charger markers use stop or depot-address locations rather than surveyed equipment positions.

![][image12]

Original duty 13414 leaves the depot at 06:43 and returns at 21:37. The fee-0 example runs 05:18–20:08 with a different 13-trip assignment; the fee-5 example has the original 12 trips and runs 06:44–21:37. All five buses together still cover the same 62 trips.

### 

### **Travel times and energy**

| From → to | Movement | Minutes | kWh |
| ----- | ----- | ----- | ----- |
| PARX → 4808 | Empty depot departure | 7 | 7.8 |
| 4808 → 2190 | Passenger service | 51–60 | 42.50 |
| 2190 → 4808 | Passenger service | 55–63 | 41.73 |
| 2190 → 2190L | Platform to charger | 0\* | 0\* |
| 2190L → 2190 | Charger to platform | 0 | 0 |
| 4808 → PARX | Empty depot return | 7 | 7.8 |
| 2190 → PARX | Empty depot return | 19 | 28.0 |

These are the times and energies used by the saved schedules. Passenger service includes scheduled stops. A direct empty connection between the terminals is 22 minutes / 37.6 kWh; it is not the service duration.

Two known approximations remain: GIRO uses 8 minutes for the original 06:43 depot departure, versus the static model’s 7\. \*GIRO also uses 1 minute / 0.4 kWh from platform 2190 to charger 2190L, which the model collapses to zero. The reverse move is zero in the source. ET\_R is an unlocated layover point, not a confirmed charger; it is not assigned an invented geographic position.

Network context: grey markers are other documented chargers, unused by these selected 18E1 buses. Their locations are stop-area proxies.

![][image13]

Full resolution: [example-bus map and terminal detail](https://github.com/ndandnd/EVSP-DR/blob/bc5391af975db7de61a1df804a57883b821c2868/outputs/week_20260921/geography_map/k5_geographic_context.png) · [all-charger overview](https://github.com/ndandnd/EVSP-DR/blob/bc5391af975db7de61a1df804a57883b821c2868/outputs/week_20260921/geography_map/k5_charger_network_context.png) · [travel data](https://github.com/ndandnd/EVSP-DR/blob/bc5391af975db7de61a1df804a57883b821c2868/outputs/week_20260921/geography_map/travel_table.csv).

Sources: [Eketrägatan stop L](https://www.openstreetmap.org/node/241780711), [Merkuriusgatan](https://www.openstreetmap.org/node/648217190), and [Transdev’s Partille site address](https://transdev.se/wp-content/uploads/2024/12/Certifikat_Transdev-Sverige-AB_ms_2024-12-03-2026-05-31-002.pdf). The depot-address match is inferred; the exact historical gate is unverified. Earlier figures below retain their dated assumptions.

## 

## **Earlier runs spent hours checking saved routes**

We saved routes from the previous, smaller problem and checked every one before restarting CG. The large part of each bar is that checking time. For example, chain 3 at k=10 spent about 283 minutes checking saved routes and 10 minutes on the remaining CG work. These completed runs show why we limited how many saved routes the new experiments check.

![][image14]

Source: completed full-pool warm runs, chains 3 and 5\. “Other CG work” includes optimization and remaining overhead; final MIP time is excluded.

## **Charging savings with matched return energy**

Five buses, 62 trips, 240 kWh batteries, 350 kW charging. Both optimized cases return at least 280.7833 kWh in total. Shared charger capacity is not imposed.

![][image15]

| Price peak | GIRO repriced ($) | Fixed duties ($) | Joint ($) |
| ----- | ----- | ----- | ----- |
| 08:00 | 490.29–490.98 | 275.87 | 257.69 |
| 12:00 | 549.59–550.60 | 328.55 | 328.36 |
| 18:00 | 483.45–483.72 | 230.78 | 215.76 |

Fixed duties: preserve trips on each bus and use DP to optimize charging. Joint: also change trip assignment using the saved route pool. Costs include electricity and 5 units per charging start, measured by physical replay.

The original range reflects unknown power within charging windows. At noon the discrete objectives tie; the small replay difference is not an improvement in that objective. Optimality claims apply only to the saved pool.

## **Geography suggests groups, but they still interact**

The full GIRO input has 40 duties and 948 trips. This map shows 838 trips as straight lines between known endpoints, not road paths. PARX and 110 trips with unmapped endpoints are omitted. Coordinates: OpenStreetMap.

![][image16]

Route 21 has 14 duties / 203 trips; local services have 26 duties / 745 trips. An optimistic timetable screen finds 31.6% of connections cross the groups. It ignores battery and charging feasibility, so geography does not establish independence.

The current decomposition pilot uses a 32-duty, 750-trip subset split into four groups of eight. Shared capacity and exchanges between groups must be checked when recombining solutions.

Earlier figures and discussions: [research archive](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit) · [slide archive](https://docs.google.com/presentation/d/1RAzaiZSh7DRf_By32mQXPCcwT1PvPDsk0S2xzxMOnDQ/edit)

## **Checking fewer saved routes lets CG get started**

Each new run checked 512 saved routes in about 5–8 minutes, then continued searching for new routes. The limit is 512 routes or 15 minutes, whichever comes first. Checking a route means rebuilding and validating it for the larger trip set. The bars separate that checking from the rest of the run. All seven shown here completed CG by the 01:27 EDT snapshot on 12 September.

![][image17]

The three previously blocked cases are chain 1 at k=7, chain 2 at k=9 and chain 4 at k=10. Their new integer solutions were still pending at this snapshot. Fewer starting routes can change the integer pool, so we must check solution quality as well as runtime.

## **With charger limits the slow step was finding a new route**

This compares two steps in the same iteration of a three-bus capacity test. Solving the LP took 0.006 seconds. Searching for a new route took about 7 hours 9 minutes. The axis is logarithmic because the times differ by several million times. The delay is inside the route search, unlike the saved-route checking in the previous figure.

![][image18]The run stopped before it could prove that no improving route remained. It generated only three new routes in eight hours. The final MIP found 16 buses among the saved routes; this is not evidence that the full problem requires 16 buses.