# All five spatial schedule comparisons

The source extraction contains **15 complete day schedules**: five recorded GIRO duties and their one-to-one saved fee-0/fee-5 counterparts, after the previously validated charging reoptimization. No new solver run was performed. The plotting entry point is [schedules.json](schedules.json); [schedule_metrics.csv](schedule_metrics.csv), [events.csv](events.csv), [visits.csv](visits.csv) and [edges.csv](edges.csv) provide flat editable tables.

## Pairing and useful examples

Indices below are zero-based source indices, preserved from the solver witnesses. Each row is a three-panel comparison.

| Priority | Original duty | Fee-0 index | Fee-5 index | Original/fee-0/fee-5 charge starts | Original trip overlap, fee 0 / fee 5 |
|---|---|---|---|---|---|
| 1 | 13414 | 1 | 0 | 9 / 9 / 6 | 6 of 12 / 12 of 12 |
| 2 | 13403 | 0 | 3 | 12 / 8 / 5 | 7 of 14 / 6 of 14 |
| 3 | 13405 | 2 | 2 | 12 / 9 / 6 | 9 of 13 / 4 of 13 |
| 4 | 13401 | 3 | 1 | 10 / 8 / 6 | 4 of 12 / 4 of 12 |
| 5 | 13408 | 4 | 4 | 9 / 8 / 7 | 3 of 11 / 6 of 11 |

13414 is the strongest first example: the original and fee-5 arm serve the same twelve passenger trips, while their charging starts differ from nine to six. Its fee-0 arm serves thirteen trips, six shared with the original. 13403 highlights a larger charge-count difference and earlier returns caused by changed trip assignments. 13405 shows later modeled depot returns. The remaining two pairs illustrate substantial trip reassignment.

All three fleets cover the same **62 source trips exactly once**. Most individual paired buses have different trip sets; every fee-0 versus fee-5 pair differs. Therefore these are not controlled fee-only causal comparisons.

Pairing preserves the maximum-total-trip-overlap assignment already used to assign terminal-energy floors. Independent enumeration of all 120 permutations confirms it is optimal, but **not unique**: fee 0 has two optimal assignments with 29 shared trips in total; fee 5 has four with 32. The pair label is a comparison convention, not evidence of the same physical bus. [pairings.csv](pairings.csv) lists added/removed source trip IDs and visual-interest reasons.

## How to draw the graph

Each schedule has `events`, `visits`, and `edges`. A visit is one occurrence at a location; repeated visits remain separate nodes. Its `arrival_min` and `departure_min` bound its dwell. `charge_indices`, `wait_indices`, and `prep_indices` refer to the schedule's event list. Every service/deadhead edge identifies its endpoint visits and timed movement; passenger edges preserve both the source trip ID and internal model ID.

**Trip-ID provenance:** “Trip 23” means `Ordered_Trip_ID=23` in our prepared full input, not a GIRO-supplied journey number. The raw GIRO workbook `Par_VehicleDetails.xlsx`, `Data!A1:X2205`, has 24 columns and no individual journey-ID or `Ordered_Trip_ID` field. Preparation added 987 unique regular-trip labels, 0–986; the k5 subset preserves them and separately uses local `count_trip_id` indices 0–61. For example, prepared ID23/local ID9 corresponds to recorded duty13414, 4808→2190, 06:51–07:49, worksheet row2170. The JSON/CSV name `source_trip_id` means the prepared source-file label.

`charge` events use actual saved intervals and kWh. `wait` events fill the entire noncharging idle interval. Original preparation activities are retained; model preparation activities are not invented. A zero-duration reference-node transfer such as 2190→2190L remains an explicit logical edge. No road-route geometry is inferred from straight diagram edges.

Original movements follow recorded GIRO clocks. Saved model arcs provide travel duration and idle energy but do not uniquely prescribe every deadhead departure. The extraction uses a feasible explicit convention: travel immediately after service, wait at the destination, and leave the fixed charging site at the next-trip deadline. Actual optimized charge times are unchanged. These reconstructed deadhead clocks must not be labeled uniquely optimized departure choices.

ET_R has no documented coordinate. Show it only as an explicitly unlocated auxiliary node, or omit its geographic placement while retaining its itinerary. 2190 is the identified terminal vicinity, not a precisely established passenger platform. Existing coordinate records include their proxy/uncertainty labels. The static model's 2190→2190L transfer is zero minutes; original GIRO uses one minute. Do not hide this distinction by merging those logical nodes.

## Verification and scope

Extraction checks passed for all 751 events and 405 visits: continuous time and location, preserved trip order/IDs, all charge intervals and counts, and energy accounting to below 2×10⁻¹³ kWh. Fleet starts are 52 / 42 / 30. Each matched terminal floor equals its original duty's final energy. Existing nonlinear SOC and five-bus shared-capacity validation is included by reference and hash in [extraction_validation.json](extraction_validation.json).

Common battery/charging parameters are 236.44 kWh, 15% reserve, 0.1 kW idle, documented 18E1 opportunity taper, PARX 60 kW, and a three-minute active charging minimum. Comparison limits remain: static model deadheads, no platform/FIFO/crew validation, and charger capacity checked only within the five-bus cohort. Original costs are reconstructed under the synthetic tariff, not observed invoices.

[extract_schedules.py](extract_schedules.py) regenerates the tables from immutable corrected k5 artifacts, the original GIRO CSV, and existing geographic provenance. The JSON records absolute source locations, row/pointer references, and SHA256 hashes. No source artifact was edited.
