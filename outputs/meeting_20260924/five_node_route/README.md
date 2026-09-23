# Duty 13309: meeting comparison pair

Added to the current weekly Google Slides as slides14–15 on23September2026. [Open the five-node graph](https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit?slide=id.g3fbe166cb3e_22_33).

The original liked graph was `outputs/week_20260921/complex_route_graphs/duty_13309_graph.png`. The saved algorithm counterpart comes from `outputs/research_followup_20260921/duty13309/`: C6k5 fee0, selected route4, among a79-trip five-route fleet. It serves the same22 passenger trips as recorded GIRO duty13309, with identical ordered IDs, endpoints and clocks. GIRO charges4times, including10:45–12:30 at PARX; the counterpart charges7times at Heden/Jons väg and skips that midday depot visit. This is distinct from the62-trip duty13414 example on existing slides10–13.

- [Trip and charging times](13309_time_comparison.png), with the stable prepared trip labels and charging-site strips.
- [Five-node spatial comparison](13309_node_comparison.png), with all cross-area activity connections and common node positions.
- [Figure data/source hashes](figure_manifest.json) and [publication checks](publication_verification.json).

These historical schedules use different physical assumptions. Recorded GIRO18E2 uses239.01kWh and recorded charging; saved algorithm uses240kWh/240kW, zero reserve, no terminal floor/shared capacity. No energy/cost saving or new physical certificate is claimed. The fee5 route from the same cohort shares only11/22 original trips and is not shown in this exact-trip pair. L labels mean passenger order; original prepared trip IDs remain unchanged. M labels retain the archived empty-move numbering; local moves stay in the source itinerary. PARX is displaced for clarity; arcs are schematic, not road paths.

The two new slides have native editable titles/captions, and full source/scope notes. All40 pre-existing slides retain visible text, except the slide2 pointer to the first chain figure changes15to17 after insertion. All original embedded media are preserved; final deck has42slides. Both new pages were exported to PDF and visually inspected. Before/after PPTX/PDF exports remain local. No solver, queue query, new cluster job, or Doc edit occurred; scheduler monitoring remains paused.

Run `python3 build_figures.py` with Matplotlib to reproduce these two scientific figures. It reads frozen schedules and reuses only the archived spatial draw function, without running that source script's extraction or write operations. No optimizer is invoked.
