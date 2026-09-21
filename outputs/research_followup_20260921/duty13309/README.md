# Duty 13309 — the entire day and actual saved algorithm counterparts

[**Whole-day comparison: original / fee 0 / fee 5**](comparison_same_cohort.png) · [PDF](comparison_same_cohort.pdf) · [Editable SVG](comparison_same_cohort.svg) · [Complete chronological itineraries](complete_itineraries.pdf) · [Editable event table](events.csv)

The morning and afternoon are combined in every graph. Each panel in the comparison represents a different complete schedule, never half a day. Repeated directed connections are grouped to keep the five-place overview readable; L labels retain passenger-leg order and the separate itinerary preserves every source trip, platform, clock, charge, wait and empty movement. M labels count all empty moves, including same-area moves retained only in the itinerary, so some M numbers are absent from the spatial graph.

| Displayed schedule | Underlying input / saved fleet | Trips on displayed route | Shared with original | Jaccard | Starts on displayed route | Figure |
|---|---|---:|---:|---:|---:|---|
| Recorded GIRO duty 13309 | Original duty / one bus day | 22 | 22/22 | 22/22 = 1 | 4 | [PNG](original_13309_full_day.png), [PDF](original_13309_full_day.pdf) |
| C6 k5 fee 0, saved route 4 | 79 trips / 5 routes | 22 | 22/22 | 22/22 = 1 | 7 | [PNG](w6_k05_fee0_full_day.png), [PDF](w6_k05_fee0_full_day.pdf) |
| C6 k5 fee 5, saved route 3 | Same 79 trips / 5 routes | 21 | 11/22 | 11/32 = 0.34375 | 1 | [PNG](w6_k05_fee5_full_day.png), [PDF](w6_k05_fee5_full_day.pdf) |
| Optional C6 k15 fee 5, saved route 2 | Different 373 trips / 15 routes | 19 | 15/22 | 15/26 = 0.57692 | 1 | [PNG](w6_k15_fee5_full_day.png), [PDF](w6_k15_fee5_full_day.pdf) |

Route numbers in this table are one-based; the source ranking retains zero-based selected-route indices. Coverage uses the original 22 source trips as denominator. Jaccard uses the union of original and candidate source-trip sets. The primary fee-0 and fee-5 arms have the same exact input SHA256. Both entire k5 fleets have 79 trips covered exactly once, with 34 and 7 charge starts, respectively. Selecting their most overlapping individual routes does not hold that route's assignment constant: the fee-5 route has eleven original trips and ten other trips. These graphs show existing algorithm outcomes; they do not establish a fee-only effect on one fixed duty.

## Physics and interpretation

| Item | Original 13309 | Both saved C6 k5 arms |
|---|---|---|
| Battery / initial energy basis | Recorded 239.01 kWh 18E2; starts at 100% SOC | 240 kWh capacity and initial energy |
| Charging | Recorded recharge and SOC changes; opportunity/depot behavior differs | Historical constant 240 kW opportunity model; exact saved continuous-realized kWh/windows retained |
| Reserve | Recorded SOC; no newly imposed reserve | Zero reserve |
| Terminal energy | Original recorded end-of-day SOC | No original terminal-energy floor |
| Charger capacity | No new physical certificate | Shared station capacity omitted and not validated |
| Deadhead clocks | Original recorded directional activity clocks | Pinned static reference deadheads; chronology reconstructed between fixed saved trip/charge anchors |
| Optimization scope | Source record | Inherited sequences replayed under destination fee; cover master; conservative-grid pricing and finite-pool MIP |

The original charges at Heden, Jons väg and PARX, including 10:45–12:30 at PARX (+105 kWh). The exact-trip fee-0 saved route charges seven times at Heden and Jons väg, without the original midday depot return. Its lower total charge is **not** a fair original-energy saving claim: initial capacity, charging physics, terminal energy and other settings differ.

Both saved k5 fleets have zero overcovered trips and their source reports validate individual continuous-realized routes under historical physics. The source pricing certificate concerns the conservative expanded-grid model, not continuous realized electricity-cost optimality. Fleet-five proof is a finite-pool MIP result, not a new full-model integer certificate. No optimizer, dispatch cleanup or new shared-capacity audit was run here. A separate [capacity-only frozen-schedule replay](../battery_rounding/duty13309_counterparts.csv) reports that reducing the selected two routes' capacity and initial energy from 240 to the relevant 239.01 kWh leaves minimum/terminal energy 0.6670007 kWh (fee0) and 12.5310003 kWh (fee5); that does not add reserve, taper, terminal-floor or full operational equivalence.

The optional stronger-overlap fee-5 route is deliberately separate: C6 k15 has a different 373-trip universe, and its saved fleet overcovers 18 trips. It is an individually replayed historical route, not a clean operational dispatch or a primary matched-cohort comparison.

## Source identity, search coverage and geographic limits

The target IDs are `226,229,234,586,589,901,792,793,796,393,396,922,626,629,632,635,638,641,447,451,456,459`. These are prepared `Ordered_Trip_ID` labels, not GIRO-supplied journey numbers. Their ordered endpoints, clocks and energy match the original source exactly in every candidate universe. Other duties can have published endpoint variants; each candidate's full route is resolved through its own pinned instance, never the master solely by numeric ID.

[Search coverage](search_coverage.csv) checks all 18 universes of the saved 13-September start-fee campaign: six chains × k5/k10/k15. Six universes contain all 22 target trips (C3 and C6 at each size); the other twelve have zero overlap. [Ranking](overlap_ranking.csv) evaluates all 120 selected routes in the twelve corresponding fee arms by Jaccard, then original-trip coverage. C6 k5 fee0 is the exact match; C6 k15 fee5 is the highest-overlap fee5 overall; C6 k5 fee5 is the highest-overlap fee5 in the primary matched universe. Existing recent k3/k5 and controlled charging-factorial figures use the distinct route21/134xx cohort and do not contain this duty. This is a bounded saved-result search, not a claim to enumerate every result ever produced.

[Read-only cluster snapshot](sources/campaign_snapshot.json) preserves remote paths, exact input/result hashes and complete saved payloads. [Manifest](figure_manifest.json) pins local source files, code commit, objective, initialization, proof scope and output hashes. The source campaign's [manifest](../../zero_charge_start_fee_20260913/manifest.json) retains resources, parent pools and source commit `06b5cb86d6c24df0ec0a5ca7189fa9552f527dd0`. This figure work uses local plotting and read-only retrieval only: no cluster submission, new dependency, Slides edit or live Doc edit.

The four located passenger areas preserve approximate geographic relative positions. PARX is displaced for clarity; its actual proxy remains in the [prior coordinate source](../../week_20260921/complex_route_graphs/coordinates.csv). The additional 13722 area is unlocated and placed schematically, visibly labeled as such. Arrows are activity connections, not roads. Saved station suffix `_0` denotes the modeled station node; exact station codes remain in the ledger. Same-reference platform transfers retain their original recorded gaps or the model's zero-time convention, not invented surveyed paths. Geographic proxies © OpenStreetMap contributors.

Reproduce with `python3 build_figures.py` from this folder or any directory. The script consumes frozen artifacts and never accesses a solver. [Editable caption](figure_caption.txt), [comparison data](comparison.csv), [all events](events.csv), [display layout](display_layout.csv) and [schedule payloads](schedules.json) accompany PNG, PDF and SVG figures. Original source artifacts remain untouched.
