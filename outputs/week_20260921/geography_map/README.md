# Geographic context for the matched k5 bus figure

This is a geographic companion to the saved-trip-sequence charging comparison: original GIRO duty **13414**, fee0 source bus **1**, and fee5 source bus **0**. The map focuses on the original and fee0 buses. It uses actual latitude/longitude, road/water context, and stop/address proxies. The earlier spectral/Plotly layouts were schematic and are not used as geographic coordinates here.

## Figure and editable data

- [Geographic figure PNG](k5_geographic_context.png), [PDF](k5_geographic_context.pdf), [SVG](k5_geographic_context.svg).
- [Optional all-charger overview PNG](k5_charger_network_context.png), [PDF](k5_charger_network_context.pdf), [SVG](k5_charger_network_context.svg): Heden, Östra Sjukhuset and Jons väg appear in muted grey as network context; they are not used by these selected 18E1 buses. This adds no bus movements.
- [Travel comparison table](travel_table.csv): concise editable rows, including recorded-versus-model differences.
- [Coordinate ledger](coordinates.csv): source URLs, confidence, scope, and explicitly unlocated codes.
- [Exact selected-bus extraction](model_geography.json), [individual services](service_trips.csv), [movements](movements.csv), [charges](charging_events.csv), [model empty-driving links](model_deadhead_links.csv), [raw directional travel data](raw_directional_dhd.csv).
- [Data extraction explanation](DATA_README.md), [figure script](build_geography.py), and [file hashes](geography_manifest.json).

**Suggested caption, kept editable:** Geographic context for GIRO duty 13414 and the representative fee0 bus in the matched k5 charging figure. Blue links join passenger-service endpoints; scheduled trips take 53–63 min for the original bus and 51–63 min for fee0. Orange dashed links show model empty-driving connections: 7 min between PARX and Merkuriusgatan, and 19 min for the fee0 return from Eketrägatan. Connections are not driven road paths. The stop and depot markers are location proxies, not surveyed charger positions. ET_R is unlocated and omitted. At Eketrägatan, the model assigns 0 min/0 kWh between 2190 and 2190L, whereas the original outward movement takes 1 min/0.4 kWh; the passenger platform remains unresolved.

## What the travel numbers mean

The original bus makes 12 passenger trips (six each direction). The fee0 bus makes 13 (seven Merkuriusgatan→Eketrägatan, six in reverse), so it finishes at Eketrägatan and returns to PARX in 19 model minutes. The original and fee5 buses finish at Merkuriusgatan and return in 7 min. A departure clock such as 06:43 is not a travel duration.

Passenger service takes longer than empty driving between the same endpoint codes. For original duty 13414, 4808→2190 takes 53–59 min and 42.50 kWh; 2190→4808 takes 55–63 min and 41.73 kWh. The empty-driving reference between those endpoints is 22 min/37.6 kWh, a different service/path assumption. That 22 min reference is **not** the duration of route 21 passenger service and is not drawn as the blue connection's time.

The input's static symmetric reference table supplies 7 min/7.8 kWh for PARX↔4808 and 19 min/28kWh for PARX↔2190. Original duty 13414 instead has an 8 min morning pull-out from PARX to 4808 and a 7 min return. Its 2190→2190L transfer is 1 min/0.4 kWh, while 2190L→2190 is permitted at 0 min. Collapsing both codes to reference 2190 removes this outward movement from the reconstructed model. These are remaining movement-model mismatches; matching battery, charging power, reserve and shared charger capacity does not resolve them.

## Coordinate evidence and limits

| Code | Location used | Evidence and precision |
|---|---|---|
|2190L|57.7170023N,11.9103268E|[OSM node 241780711](https://www.openstreetmap.org/node/241780711), named Eketrägatan, stop L. High confidence for the stop; proxy for the charger.|
|4808|57.7620108N,12.0703756E|[OSM node 648217190](https://www.openstreetmap.org/node/648217190), named Merkuriusgatan. High confidence for the stop; proxy for charger location.|
|PARX|57.737614N,12.086805E|GIRO attachment audit identifies Partille garage. [Transdev's site certificate](https://transdev.se/wp-content/uploads/2024/12/Certifikat_Transdev-Sverige-AB_ms_2024-12-03-2026-05-31-002.pdf), PDF page 2, lists Järnringen 5, 433 30 Partille. [Hitta's address map](https://www.hitta.se/v%C3%A4stra%2Bg%C3%B6talands%2Bl%C3%A4n/partille/j%C3%A4rnringen%2B5/omr%C3%A5de/57.737614:12.086805) supplies the coordinate. The historical PARX-to-address binding is an inference; this is a depot-address proxy, not a verified garage gate/charger.|
|2190|No unique point assigned|The inset shows the vicinity of known Eketrägatan bus stops. The actual passenger platform in this duty is unresolved; no artificial coordinate is substituted.|
|ET_R|Unlocated|One-minute original GIRO movements associate it with Eketrägatan, but there is no verified coordinate. It is omitted from the map and retained in the source movements.|

The inset's shaded region bounds archived OSM bus-stop positions; it is an illustrative terminal vicinity, not a surveyed terminal boundary. Labels C–K are OSM stop-position references, not a claim that route 21 serves every platform. The optional network figure adds previously verified stop-area proxies for [Heden](https://www.openstreetmap.org/relation/2766184), [Östra Sjukhuset](https://www.openstreetmap.org/relation/2208859) and [Jons väg](https://www.openstreetmap.org/node/2097238852). Grey markers provide context only; they do not imply eligibility for these selected 18E1 buses.

Primary basemap and stop evidence: archived OSM JSON from 2026-09-10 plus an OSM API area extract from 2026-09-21. The geographic context is current, while the GIRO timetable is historical. Map data **© OpenStreetMap contributors**, available under [ODbL](https://www.openstreetmap.org/copyright). Hitta is the source of the address coordinate only. The original model tables contain no GPS; no road route or travel-time estimate is derived from this basemap. An attempted Nominatim address search resolved only the broader neighborhood, so those returned coordinates were rejected.

## Reproduce and scope

Run `python3 outputs/week_20260921/geography_map/build_geography.py` from the repository root with Matplotlib installed. Rendering was verified with Python 3.12/Matplotlib in the local environment. All map inputs are archived in `sources/`; rebuilding needs no network. Source paths and SHA256 hashes for the scientific movement extraction are recorded in `model_geography.json`; `geography_manifest.json` hashes the map package.

This is a geographic explanation of existing saved-trip-sequence recharging results. It is not a new CG solution, road-routing study, or proof of all GIRO operational constraints. Fee0 and fee5 use different saved trip assignments/station paths, so their aggregate charging-start difference is not a controlled fee-only causal experiment.
