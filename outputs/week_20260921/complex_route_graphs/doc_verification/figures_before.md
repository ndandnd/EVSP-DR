# **Current research figures**

## **21 September — A bus day as a graph**

Five examples compare recorded GIRO duties with saved fee-0 and fee-5 schedules after charging reoptimization. Blue arrows are passenger trips; grey arrows leave or return to PARX; orange marks charging. The overview keeps approximate geographic positions. The detailed companion separates repeat visits and lists every wait, charge and movement in order.

![][image1]

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

![][image2]

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

![][image3]

Full resolution: [example-bus map and terminal detail](https://github.com/ndandnd/EVSP-DR/blob/bc5391af975db7de61a1df804a57883b821c2868/outputs/week_20260921/geography_map/k5_geographic_context.png) · [all-charger overview](https://github.com/ndandnd/EVSP-DR/blob/bc5391af975db7de61a1df804a57883b821c2868/outputs/week_20260921/geography_map/k5_charger_network_context.png) · [travel data](https://github.com/ndandnd/EVSP-DR/blob/bc5391af975db7de61a1df804a57883b821c2868/outputs/week_20260921/geography_map/travel_table.csv).

Sources: [Eketrägatan stop L](https://www.openstreetmap.org/node/241780711), [Merkuriusgatan](https://www.openstreetmap.org/node/648217190), and [Transdev’s Partille site address](https://transdev.se/wp-content/uploads/2024/12/Certifikat_Transdev-Sverige-AB_ms_2024-12-03-2026-05-31-002.pdf). The depot-address match is inferred; the exact historical gate is unverified. Earlier figures below retain their dated assumptions.

## 

## **Earlier runs spent hours checking saved routes**

We saved routes from the previous, smaller problem and checked every one before restarting CG. The large part of each bar is that checking time. For example, chain 3 at k=10 spent about 283 minutes checking saved routes and 10 minutes on the remaining CG work. These completed runs show why we limited how many saved routes the new experiments check.

![][image4]

Source: completed full-pool warm runs, chains 3 and 5\. “Other CG work” includes optimization and remaining overhead; final MIP time is excluded.

## **Charging savings with matched return energy**

Five buses, 62 trips, 240 kWh batteries, 350 kW charging. Both optimized cases return at least 280.7833 kWh in total. Shared charger capacity is not imposed.

![][image5]

| Price peak | GIRO repriced ($) | Fixed duties ($) | Joint ($) |
| ----- | ----- | ----- | ----- |
| 08:00 | 490.29–490.98 | 275.87 | 257.69 |
| 12:00 | 549.59–550.60 | 328.55 | 328.36 |
| 18:00 | 483.45–483.72 | 230.78 | 215.76 |

Fixed duties: preserve trips on each bus and use DP to optimize charging. Joint: also change trip assignment using the saved route pool. Costs include electricity and 5 units per charging start, measured by physical replay.

The original range reflects unknown power within charging windows. At noon the discrete objectives tie; the small replay difference is not an improvement in that objective. Optimality claims apply only to the saved pool.

## **Geography suggests groups, but they still interact**

The full GIRO input has 40 duties and 948 trips. This map shows 838 trips as straight lines between known endpoints, not road paths. PARX and 110 trips with unmapped endpoints are omitted. Coordinates: OpenStreetMap.

![][image6]

Route 21 has 14 duties / 203 trips; local services have 26 duties / 745 trips. An optimistic timetable screen finds 31.6% of connections cross the groups. It ignores battery and charging feasibility, so geography does not establish independence.

The current decomposition pilot uses a 32-duty, 750-trip subset split into four groups of eight. Shared capacity and exchanges between groups must be checked when recombining solutions.

Earlier figures and discussions: [research archive](https://docs.google.com/document/d/1f0orWtM1-_VWAqjj6GnWP78webCOnvoc01x2VQvaA_k/edit) · [slide archive](https://docs.google.com/presentation/d/1RAzaiZSh7DRf_By32mQXPCcwT1PvPDsk0S2xzxMOnDQ/edit)

## **Checking fewer saved routes lets CG get started**

Each new run checked 512 saved routes in about 5–8 minutes, then continued searching for new routes. The limit is 512 routes or 15 minutes, whichever comes first. Checking a route means rebuilding and validating it for the larger trip set. The bars separate that checking from the rest of the run. All seven shown here completed CG by the 01:27 EDT snapshot on 12 September.

![][image7]

The three previously blocked cases are chain 1 at k=7, chain 2 at k=9 and chain 4 at k=10. Their new integer solutions were still pending at this snapshot. Fewer starting routes can change the integer pool, so we must check solution quality as well as runtime.

## **With charger limits the slow step was finding a new route**

This compares two steps in the same iteration of a three-bus capacity test. Solving the LP took 0.006 seconds. Searching for a new route took about 7 hours 9 minutes. The axis is logarithmic because the times differ by several million times. The delay is inside the route search, unlike the saved-route checking in the previous figure.

![][image8]The run stopped before it could prove that no improving route remained. It generated only three new routes in eight hours. The final MIP found 16 buses among the saved routes; this is not evidence that the full problem requires 16 buses.