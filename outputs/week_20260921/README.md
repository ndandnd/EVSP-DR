# Week of 21 September 2026 — EVSP–DR

**Five-place example:** [Recorded GIRO duty13309](complex_route_graphs/README.md), [morning/afternoon spatial graph](complex_route_graphs/duty_13309_graph.png), [five-page full itinerary](complex_route_graphs/duty_13309_daybook.pdf), and [interactive gallery](complex_route_graphs/gallery.html). It serves22 trips and charges at Heden, Jons väg and PARX, including a105-minute midday depot recharge. This is original-only; the depot is displaced in the diagram for clarity.

**Latest visual follow-up:** [Five spatial bus-day comparisons](spatial_schedule_graphs/README.md), [browsable gallery with editable keys](spatial_schedule_graphs/gallery.html), [10-page comparison PDF](spatial_schedule_graphs/all_comparisons.pdf), and [k40 ladder path-dependence audit](spatial_schedule_graphs/ladder_path_dependence.md). Complete itineraries preserve all visits, waits, charges and depot returns; the earlier geographic maps remain available below.

[Latest research document snapshot](spatial_schedule_graphs/doc_verification/current_after.pdf) · [Latest figures archive](spatial_schedule_graphs/doc_verification/figures_after.pdf) · [Verified preservation of prior text, tables, links and all seven earlier images](spatial_schedule_graphs/doc_verification/README.md).

[Live research document](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.79m3d3x4h45m) · [This week's 14-slide deck](https://docs.google.com/presentation/d/1F6udjgkiPH51vMUcku7ZT3PhZPAxsQwUCi3TK9vFRDM/edit)

[Deck PDF](slides/weekly_deck.pdf) · [Editable PowerPoint export](slides/weekly_deck.pptx) · [Initial weekly document snapshot](doc_verification/after.pdf) · [Publication and history-preservation checks](doc_verification/README.md)

[Concise research journal](RESEARCH_JOURNAL.md) · [Full solver evidence and log-reading guide](evidence/README.md) · [Capacity and strict-physics audit](capacity_strict/README.md) · [Matched k5 figure](cleanup_physics/one_bus_k5_joint_matched.png)

[Geographic companion and source audit](geography_map/README.md) · [Selected-bus map](geography_map/k5_geographic_context.png) · [All-charger context](geography_map/k5_charger_network_context.png) · [Editable travel comparison](geography_map/travel_table.csv)

[Updated research document snapshot](geography_map/doc_verification/current_after.pdf) · [Figures archive with both maps](geography_map/doc_verification/figures_after.pdf) · [Verified history preservation](geography_map/doc_verification/README.md). The geographic follow-up updates Docs; Slides retain the previously published 14-slide set.

[Six baseline chains extended through k40](chain_extension_40/README.md) · [Verified production launch](chain_extension_40/launch_verification.json) · [Exact job map](chain_extension_40/case_jobs.json)

The eight jobs from the previous update have finished. Six capacity CGs produced completed pool MIPs; the C1 incumbent-transfer follow-up now proves eight buses; strict k15 used its budget building the graph and completed zero pricing iterations. The [journal](RESEARCH_JOURNAL.md) gives the eight-job table, interpretation, algorithm work and concise dated history.

**Later on 21 September:** the six existing baseline chains now continue at k33–40: 44 distinct graph builds, 48 CGs and 48 MIPs. The verified initial snapshot has five graph tasks running and 39 resource-pending; all CG/MIP dependencies are intact. This preserves baseline scientific settings and full previous-k column inheritance. Larger graph allocations follow measured time/memory, while CG/MIP budgets stay unchanged. [Launch details and timing limits](chain_extension_40/README.md).

The [14:46:43 EDT snapshot](chain_extension_40/final_queue_snapshot.json) advances this to **14 new graph tasks running**, 29 waiting for resources and one scheduled requeue waiting for its eligible time; 96 CG/MIP stages remain dependent. The separate strict k16 CG also runs, making 15 running jobs in this scope. The original launch snapshot is preserved.

The main new evidence is auditable locally:

- **Fresh k8:** four finite pools provably require nine; appending eight sequential witness routes to each restores eight. [Complete logs and exact proof lines](evidence/LOG_EXCERPTS.md), [paired results](evidence/k8_witness_summary.csv).
- **Why:** the LP splits eight buses over 80–101 fractional routes, while integer scheduling needs compatible whole routes. [Dual-cost decomposition, route origins and source checks](evidence/README.md), [recomputed values](evidence/mechanism_summary.csv).
- **Fresh k15:** all twelve 12-hour fleet searches and subsequent charging stages are finished, with fleets 16–19 and fleet bound 15; none proves that 15 is absent. [Completed results and log locations](evidence/k15_12h_summary.csv).
- **Charging-aware duplicate removal:** a real k5 repair removes its repeated trip, preserves five buses and validates exactly-once service. [Result and before/after metrics](cleanup_physics/cleanup_result/summary.json), [validation](cleanup_physics/cleanup_result/validation.json).
- **Strict-physics algorithm benchmark:** on one matched 26-trip case, packed construction is 2.90× faster, peak memory 14.1× lower and mean pricing 530.8× faster, with five identical reduced costs and all15 route replays passing. This is a small-case benchmark, not large-case or full-CG evidence. [Verified metrics and scope](capacity_strict/README.md); k16 recovery646675 now runs.
- **Physics-matched k5:** the old figure was not fully matched. [Replacement figure](cleanup_physics/one_bus_k5_joint_matched.png), [data](cleanup_physics/one_bus_k5_joint_matched.csv), [validated constraints](cleanup_physics/joint_validation.json), [settings and input hashes](cleanup_physics/manifest.json). Post-hoc charging of saved trip sequences validates five buses and 62 exactly-once trips in both arms, including charger counts; fee0/fee5 use42/30 starts. The arms inherit different trip sequences and station paths, so this is a descriptive comparison, not a fee-only causal test. It is not fresh CG or a complete GIRO constraint validation. [Full scope and logs](cleanup_physics/README.md).
- **Geographic explanation:** actual stop/address proxies locate Eketrägatan, Merkuriusgatan and PARX; a second map shows other network chargers. Original passenger services take 53–63 minutes, while the separate empty-driving reference between their endpoints is 22 minutes. The audit also identifies remaining travel mismatches: the original outward 2190→2190L movement is 1 minute/0.4 kWh versus model 0/0, and the original morning PARX→4808 departure is 8 minutes versus model 7. PARX is an explicitly sourced address proxy; ET_R remains unlocated. [Maps, exact movements, coordinate sources and limits](geography_map/README.md).

For source retrieval, all 35 full Gurobi logs and 193 hashed copied source files are under `/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/`. [The source manifest](evidence/source_manifest.json) maps every file to its original Unicorn path and SHA-256. The [evidence builder](evidence/build_evidence.py) reproduces and checks these tables without rerunning optimization. Capacity artifacts and new recovery receipts are indexed separately in [capacity_strict](capacity_strict/README.md).

The historical experiment entry point remains [the research register](../research_register/README.md). New dated entries supersede specific conclusions/statuses while retaining original logs, failed attempts and experimental budgets. Scheduler success, pricing certificates, finite-pool proofs, physical validation and GIRO target attainment remain separate.
