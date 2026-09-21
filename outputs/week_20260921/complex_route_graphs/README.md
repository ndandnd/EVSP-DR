# Recorded GIRO duty 13309 across five places

[**Readable two-panel graph**](duty_13309_graph.png) · [Vector PDF](duty_13309_graph.pdf) · [SVG](duty_13309_graph.svg) · [**Five-page graph and full itinerary**](duty_13309_daybook.pdf) · [Interactive gallery with editable tables](gallery.html)

This actual bus day visits **Heden, Partille centrum, Gamlestads Torg, Jons väg and the PARX depot**. It serves 22 passenger trips and charges four times at three sites. Its most useful feature is the midday depot visit: **return at10:42, charge10:45–12:30 (+105kWh), leave again at12:43**. The two panels divide the day at that stop and retain identical node positions. This is the **recorded GIRO schedule**, not a newly optimized result or a claimed optimized comparison.

The four passenger-area nodes preserve approximate geographic relative positions. **PARX is displaced into clear space** so passenger arcs do not appear to serve the depot. [Actual coordinate proxies and their sources](coordinates.csv) and [display positions](display_layout.csv) are separately retained. Curved arrows show connections between recorded activities, not roads. Charger/platform details are not surveyed coordinates; the historical PARX address binding is explicitly an inference.

## Read the day

- **L1–L22** number passenger legs in traversal order; **M1–M5** number inter-area empty movements. These are separate from **Trip numbers**, which use stable labels from our prepared input, not GIRO-supplied journey numbers.
- **C1–C4** number charging connections; **V1–V28** number repeated area visits. Orange rings identify sites used for charging somewhere during the day, not necessarily in each panel.
- The [27-row leg key](diagram_leg_key.csv), [28-row visit key](diagram_visit_key.csv), [60-event ledger](diagram_event_key.csv) and [charge key](diagram_charging_key.csv) retain exact clocks, recorded platform codes, source trip labels and workbook row references. Preparation05:12–05:19 and19:02–19:05 is included. The 12 raw location codes are grouped only through the documented reference-area mapping.
- Same-area platform changes without a separate movement record remain explicit gaps; no physical path, travel time or energy is invented. All recorded within-area deadheads, waits and charges remain in the complete itinerary.

Open [gallery.html](gallery.html) locally for clickable arcs and native editable tables. GitHub displays its HTML source; download the folder and open the file for interactivity. The PNGs and PDFs are directly viewable on GitHub. The [single-panel full-day graph](duty_13309_full_day.png) is optional; the split view is clearer.

## Recorded charging and scope

| Connection | Site | Recorded interval | Recorded recharge |
|---|---|---|---:|
| C1 | Heden3127L | 06:42–06:56 | 40.594kWh |
| C2 | Jons vägJON_A | 09:15–09:25 | 44.832kWh |
| C3 | PARX depot | 10:45–12:30 | 105.000kWh |
| C4 | Heden3127L | 16:22–16:49 | 130.728kWh |

Recharge values are copied from GIRO, not recalculated using the earlier route21 assumptions. Their recorded SOC changes agree with this duty’s **239.01kWh18E2 capacity** within0.000016kWh. The18E1 capacity236.44kWh does not apply here. Exact clocks, endpoints, SOC and energy cells were cross-checked against the raw workbook. These extraction checks are not a fresh optimization or full physical feasibility certificate.

[schedules.json](schedules.json), [validation.json](validation.json) and [extraction notes](DATA_README.md) retain input hashes and pointers. The source selection also preserves original duty13320 as an alternative:16 trips and six areas, including unlocated13722. It is not geographically plotted. No ready validated optimized counterpart was established for either duty; existing matched three-arm examples concern route21.

## Reproduce and audit

1. `extract_complex_originals.py` regenerates the source extraction (requires original research inputs).
2. `plot_complex_routes.py` generates the graph PNG/PDF/SVG with Matplotlib.
3. `build_gallery.py` generates the interactive SVG, native HTML and editable CSV keys.
4. `build_daybook.py` combines the vector graph with native PDF tables using ReportLab/pypdf.

[Figure manifest](figure_manifest.json) records file hashes and visual/data checks. All five PDF pages were rendered and inspected; the primary diagram was reviewed after separating the depot from passenger arcs. The source original and existing geography/spatial comparison artifacts remain unchanged.

Geographic references: [Heden](https://www.openstreetmap.org/relation/2766184), [Jons väg](https://www.openstreetmap.org/node/2097238852), [Partille centrum](https://www.openstreetmap.org/relation/2849678), [Gamlestads Torg](https://www.openstreetmap.org/relation/11803527), and the [operator's site-address certificate](https://transdev.se/wp-content/uploads/2024/12/Certifikat_Transdev-Sverige-AB_ms_2024-12-03-2026-05-31-002.pdf) with the address geocode linked incoordinates.csv. Geographic data ©OpenStreetMap contributors; these are location proxies, not surveyed historical charger/platform positions.

[Updated Google Doc figure archive](doc_verification/figures_after.pdf) · [Preservation verification](doc_verification/README.md). The new first page preserves all prior figure text and eight images; no Slides or current-results tab was edited.
