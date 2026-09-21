# Five spatial bus-day comparisons — 21 September 2026

[Browsable gallery with editable keys](gallery.html) · [All five graphs and complete itineraries, 10-page PDF](all_comparisons.pdf) · [Contact sheet](contact_sheet.png) · [Comparison CSV](comparison_summary.csv) · [Leg key CSV](leg_key.csv)

The primary view is a **fixed spatial multigraph**, with approximate geographic positions, directed passenger legs, explicit depot departure/return and every charging interval. A companion visit-expanded graph preserves full chronology, local transfers, waiting, preparation and ET_R rest visits. Its vertical spacing is not elapsed time. Original geographic maps remain unchanged in [the earlier map package](../geography_map/README.md).

| Original duty | Charge starts: original / fee0 / fee5 | Spatial graph | Complete itinerary | Why inspect it |
|---|---|---|---|---|
|13414|9 / 9 / 6|[PNG](duty_13414_graph.png), [PDF](duty_13414_graph.pdf)|[PNG](duty_13414_itinerary.png), [PDF](duty_13414_itinerary.pdf)|Original and fee5 share all twelve passenger trips; fee0 has a different thirteen-trip assignment.|
|13403|12 / 8 / 5|[PNG](duty_13403_graph.png), [PDF](duty_13403_graph.pdf)|[PNG](duty_13403_itinerary.png), [PDF](duty_13403_itinerary.pdf)|Largest charging-count difference; shorter modeled days also reflect different passenger assignments.|
|13405|12 / 9 / 6|[PNG](duty_13405_graph.png), [PDF](duty_13405_graph.pdf)|[PNG](duty_13405_itinerary.png), [PDF](duty_13405_itinerary.pdf)|Longer modeled days and charging intervals.|
|13401|10 / 8 / 6|[PNG](duty_13401_graph.png), [PDF](duty_13401_graph.pdf)|[PNG](duty_13401_itinerary.png), [PDF](duty_13401_itinerary.pdf)|Different return times despite similar service counts.|
|13408|9 / 8 / 7|[PNG](duty_13408_graph.png), [PDF](duty_13408_graph.pdf)|[PNG](duty_13408_itinerary.png), [PDF](duty_13408_itinerary.pdf)|Substantial reassignment shows why a paired label is not a physical bus identity.|

**Labels:** L1, L2, … indicate chronological passenger-leg order. **Trip numbers use stable labels from our prepared input, not GIRO-supplied journey numbers.** C1, C2, … indicate charging-session order; V1, V2, … indicate visit order. Exact data remain in [schedules.json](schedules.json) and [editable event/visit/edge tables](DATA_README.md); display clocks round to the nearest second.

These are existing recharging results on saved trip sequences, not fresh CG. All three fleets cover the same 62 trips exactly once, with 52 / 42 / 30 starts. Every fee0/fee5 paired trip set differs; the figures do not isolate a fee-only causal effect. Pairing preserves the previous optimal, nonunique overlap assignment and terminal-energy mapping. [Pairing and physics details](DATA_README.md).

The spatial overview groups stop2190 and charger2190L at Eketrägatan; the itinerary separates their logical visits and retains the original one-minute outward transfer versus the model zero-minute transfer. ET_R is explicitly unlocated and placed schematically. Depot/station coordinates are proxies, not road-route geometry. Modeled empty-driving clocks use a feasible reconstruction convention, not uniquely optimized departure choices. Recorded preparation is preserved; no preparation is invented for model schedules. Zero-duration same-place bookkeeping is suppressed from dwell annotations, while the complete source events and distinct-location zero-time edges remain available.

[Independent figure QA](independent_figure_qa.md), [parent data checks](parent_data_checks.json) and [extraction validation](extraction_validation.json) verify the 15 schedules and all 124 charge windows. [Plotting script](plot_spatial_schedules.py) and [gallery builder](build_gallery.py) reproduce the views. The gallery contains native HTML tables that can be selected/edited/copied; it does not store edits back to source CSV/JSON.

The accompanying [k40 path-dependence audit](ladder_path_dependence.md) explains what ladder order can change. Within an identical final input/model, the complete weighted event-graph LP optimum is path independent; uncertified restricted masters, route weights and finite integer pools can differ. Parent routes are reoptimized on the child graph, so parent-only charging times do not enter. The six final inputs form three byte-identical groups, not one common instance. This is a source/theory audit, not a finished k40 result.
