# Independent spatial-figure QA — 21 September 2026

**PASS: no blocking finding.** Read-only independent inspection of the finalized plotting/extraction code, extracted data and primary rendered figure `duty_13414_graph.png`. No source or plotting files were changed. This receipt records checks already executed.

## Checks executed

- Rehashed every source listed in `schedules.json`; all SHA256 values matched.
- Checked all 15 schedules and all 124 charging intervals directly against the recorded GIRO/source witnesses. Charge sites, starts, ends and optimized kWh were preserved exactly; display clocks round to seconds.
- Checked passenger prepared trip IDs, locations and service times against the prepared full input. L1, L2, … represent chronological passenger-leg order, not prepared trip IDs.
- Verified continuous time and location through every extracted day schedule. Each of the three fleets covers the same 62 trips exactly once.
- Verified plotted outbound/return depot endpoints against schedule metrics. Recorded preparation events remain in the itinerary; modeled preparation events are not invented.
- Verified that every movement event is present in the visit-expanded itinerary, including zero-duration transfers between distinct logical nodes. All charge sites used by these schedules appear in the overview charging tables.
- Recomputed normalized geographic positions from the recorded coordinate proxies: 4808 ≈ (1.000, 0.527), PARX ≈ (1.103, 0.241), consistent with rounded plotting constants.
- Visually inspected the primary three-panel graph. Approximate geography, leg-order notation and ET_R’s unlocated status are legible. The separate itinerary explicitly states that vertical spacing is not elapsed time.

## Scope and interpretation

The figure is a faithful rendering of saved schedules, not a new optimization or an independent proof of full GIRO operational feasibility. Fee-0 and fee-5 inherit different trip sequences and station paths, so this comparison does not isolate a causal effect of the fee. Original13414 and its fee-5 counterpart share twelve service trips; other operational differences remain.

Keep adjacent editable captions explicit: `Ordered_Trip_ID` is a prepared label rather than a GIRO-supplied journey number; ET_R is unlocated; 2190/2190L are grouped only in the geographic overview and remain distinct in the itinerary; model deadhead departure clocks are reconstructed feasible placements; original preparation is retained where recorded. Geometry is schematic connectivity using approximate locations, not road-route geometry.

## Reviewed artifacts

Hashes below identify the files inspected; subsequent edits require a new or amended receipt.

- `plot_spatial_schedules.py`: `e5f2a025413b59388a7e23d4851fca0f45794fd15f6ba5a3123566ca3cb1a2b6`
- `extract_schedules.py`: `22ba7e06a8c8b04a0f6f04b425bcda4c83f27ac36178939acf6a87400efd8a77`
- `schedules.json`: `f9e52bd0fbf61bdb513ec5ef2bc64295c9ea1d37bb17453afc510090ecd11de1`
- `duty_13414_graph.png`: `7b1489622aa0eadef0a4728461d4e1ef0ba64c13b43f1c3706b4a2c5c6c0be2b`
