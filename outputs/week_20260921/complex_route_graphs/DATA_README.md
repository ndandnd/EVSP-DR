# A real bus day across five places

**Use recorded GIRO duty13309 as the primary example.** Its full day visits exactly five documented reference areas: Heden, Partille centrum, Gamlestads Torg, Jons väg and the PARX depot. It has22 passenger trips and four recorded charging connections at three distinct sites—two opportunity sites and the depot. Twelve raw platform/depot codes are retained in the data; grouping uses the existing `Ref_dict.csv`, not an invented merger of unrelated places.

| Recorded activity | Time | Recharge |
|---|---|---:|
| First depot departure | 05:19 | — |
| Heden,3127L | 06:42–06:56 | 40.594kWh |
| Jons väg,JON_A | 09:15–09:25 | 44.832kWh |
| Midday depot return | 10:42 | — |
| Depot,PARX | 10:45–12:30 | 105.000kWh |
| Second depot departure | 12:43 | — |
| Heden,3127L | 16:22–16:49 | 130.728kWh |
| Final depot return | 19:02 | — |

The full recorded schedule includes preparation05:12–05:19 and19:02–19:05. The22 service edges and five inter-area deadhead edges cover the entire day, including the midday depot visit. Total recorded recharge is321.1537248kWh. Energy values were copied directly from the GIRO recharge cells; their SOC changes independently agree with the documented **239.01kWh18E2** usable capacity to within0.000016kWh. The236.44kWh18E1 parameters from the earlier route21 figure do not apply here.

The secondary extraction is duty13320:16 service trips, three charges at the same three sites, and six reference areas. Its extra location13722 remains unlocated. It is retained as an alternative, not needed for the primary figure. [candidates.csv](candidates.csv) ranks ten original schedules by the requested area count and then simplicity; [saved_candidates.csv](saved_candidates.csv) records the bounded review of saved model routes.

## Files and interpretation

[schedules.json](schedules.json) contains both complete original days. Its schedules contain `events`, area-level `visits`, and inter-area `edges`. Each event preserves exact raw `from_code`/`to_code`, grouped `from_ref`/`to_ref`, times, original duty, recorded SOC/energy and original workbook row pointer. Flat tables: [events.csv](events.csv), [visits.csv](visits.csv), [edges.csv](edges.csv), [charging.csv](charging.csv), and [areas.csv](areas.csv).

Adjacent GIRO activities sometimes switch platforms within the same reference area without a separate movement row. The extraction explicitly calls these `same_reference_gap` events, including zero-minute changes; it does not invent a physical path, movement duration, or energy. Same-place intervals are `wait` events. Waiting at the depot does not automatically imply the bus consumes idle energy: no replacement energy model is applied.

Trip numbers are prepared-input `Ordered_Trip_ID` labels, not GIRO-supplied journey numbers. Original workbook row pointers identify the recorded activities. Geographic points are documented location proxies, not surveyed platform/charger coordinates; straight graph edges are schedule connections, not road geometry.

## Validation and comparison scope

All selected activity rows were checked directly against `Par_VehicleDetails.xlsx`, including exact duty, activity type, endpoints, times, energy, recharge and SOC. Both extracted schedules have continuous clocks and locations after explicitly retaining the unspecified within-area gaps, unique trip labels and no invented inter-area movement. [validation.json](validation.json) preserves checks and source hashes. This is an extraction of recorded schedules, not a new optimization or comprehensive physical-feasibility certificate.

These figures are **original GIRO only**. The existing matched fee0/fee5 k3 and k5 studies contain only route21 duties. A saved18E2 model fleet was also reviewed: it has duplicate service and fails documented shared charger counts, and its constant240kW opportunity model does not match the documented taper. It is therefore not presented as a corrected optimized counterpart. No ready validated three-arm comparison for13309 or13320 was established in this bounded search.

[extract_complex_originals.py](extract_complex_originals.py) regenerates the extraction. No solver was run and no source record was changed.
