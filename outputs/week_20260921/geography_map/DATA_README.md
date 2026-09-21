# Model geography for the three buses in the corrected k5 figure

This folder's extraction files describe the exact buses in `cleanup_physics/one_bus_k5_joint_matched.png`. They contain no guessed coordinates or geographically inferred travel times.

- Original: GIRO duty **13414**, source trip IDs 23,36,49,68,81,100,113,126,139,154,165,178. Leaves PARX **06:43**, first service **06:51**, last service ends **21:30**, returns **21:37**. Prep-out is06:36–06:43; prep-in21:37–21:40. Twelve service trips and nine charges.
- Matched fee0: `saved_joint_fee0.json`, bus index **1** (second bus), source IDs6,22,35,48,61,74,87,100,113,126,139,154,165. Leaves PARX **05:18**, first service **05:25**, last service ends **19:49** at Eketragatan, returns **20:08**. Thirteen service trips and nine charges.
- Matched fee5: `saved_joint_fee5.json`, bus index **0** (first bus), same twelve source trip IDs as original duty13414. Leaves PARX **06:44**, first service **06:51**, last service ends **21:30**, returns **21:37**. Twelve service trips and six charges.

The original and fee5 displayed buses have the same passenger trips. Across the full five-bus arms, trip assignments and fixed station paths differ, so the fleet comparison remains noncausal for the fee alone.

## Suggested uncluttered map

Use three geographical areas: **Partille garage (PARX)**, **Merkuriusgatan (4808)**, and **Eketragatan (2190 platform /2190L charger)**. Original duty13414 additionally visits **ET_R**, a nearby layover code with one-minute source deadheads; its formal name and exact position remain unverified. Show it only if geocoding supports it, or as an explicitly unlocated local inset.

Show the passenger corridor separately from depot deadheads. Every displayed service trip is route5021 (route21), between4808 and2190. Service lasts51–63minutes across these plotted trip sets and uses41.73–42.50kWh. The **empty** reference-DHD arc between those endpoints is22minutes/37.6kWh; it is not the duration of the passenger trip and is not traversed as a separate intertrip deadhead by these three displayed buses.

The distinct terminal/charger codes must not imply distinct invented coordinates. The model maps2190L and2190 to one reference node and assigns0minutes/0kWh between them. The raw directional DHD source instead has2190→2190L=1minute/0.2km, while2190L→2190=0. This is a concrete known modeling simplification. Original duty13414 actually executes the one-minute movement. Similarly, PARX→4808 is7minutes in the static model but8 in the raw peak band and in GIRO duty13414's06:43 departure. Map line labels should identify model values, with these exceptions in an editable note/table.

## Files

- `model_geography.json`: all selected buses, ordered trips, movement legs, charging events, location metadata, raw directional DHD comparisons, source SHA-256 hashes and row/pointer provenance.
- `compact_movement_table.csv`: seven editable rows for the document/slides; passenger service and empty movements are explicitly different.
- `locations.csv`: eight documented location codes/names/roles. Coordinates are null pending separately verified geocoding. The additional opportunity sites3127L(Heden),7880C(ÖstraSjukhuset),JON_A(Jonsväg) are not visited by these buses and should normally be omitted from this map.
- `service_trips.csv`:37 service-trip appearances across the three panels, preserving source and internal IDs, endpoints, departure/arrival times, scheduled duration and energy.
- `movements.csv`: original recorded movements and model-inherited deadheads, including zero reference arcs. Model movement clocks use immediate post-trip travel and departure from station at the next trip's deadline, consistently with the reconstructed path. They do not include charging time in travel time.
- `charging_events.csv`:24 actual charging intervals across the three displayed buses. Fee0/5 data come from final capacity-rescheduled solutions, not the earlier overlapping diagnostic replay.
- `service_corridors.csv`: directional duration/energy ranges by bus.
- `model_deadhead_links.csv`: selected static model arcs and exact source reference-table rows; values are symmetric at the reference-pair level.
- `raw_directional_dhd.csv`: original `Par_DHD.xlsm`, sheetDeadhead, matching place pairs with base and interval-specific duration/distance and worksheet row numbers. These values are source evidence and were not substituted into the model run.

Travel-time conventions: raw DHD interval1 is06:30–08:40 and15:00–18:30; interval2 is08:41–09:30 and14:30–14:59; interval3 is00:00–05:59 and21:00–23:59; remaining departure times usebase. Values above24:00 use modulo24 for interval lookup. The plotted model uses a static minimum-duration symmetric reference lookup, not these departure-dependent directional values.

No documented GPS coordinates were found in the inspected source files. `data/bus_outputs/global_stops_hubs.html` contains abstract graph-layout x/y positions; it is **not geographical** and must not be used as latitude/longitude. Charger/place names are documented in the attachment audit; exact depot/charger/platform coordinates still require external geographic evidence.
