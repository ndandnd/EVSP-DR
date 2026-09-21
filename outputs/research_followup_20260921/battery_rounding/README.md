# Frozen-schedule battery-capacity replay — 21 September 2026

The saved 240 kWh schedules do **not** all remain feasible when only usable battery capacity and full initial energy are reduced. This is a deterministic posthoc energy replay, not a solver run or a claim that the same trip routes cannot be repaired.

The documented capacities are **236.44 kWh for 18E1** and **239.01 kWh for 18E2**, rather than a universal “237”. Battery energy (kWh), charging power (kW), and the earlier computer-memory number (GiB) are unrelated quantities. The profile definitions are in `.codex-work/review-strict-chain-20260916/src/giro_partille_physics.py`; the associated recovery README documents their interpretation.

## Fixed panel and findings

All 24 instances of the pinned paper panel (six chains × k=5,8,10,15), both fresh/base and sequential/warm final MIPs, were fetched read-only from Unicorn. All **48 result-file hashes** match the pinned 14 September monitor. Instance and static input hashes match the experiment records. There are **487 selected route occurrences**, representing **463 distinct saved schedules** under a fingerprint of source trip IDs, station sequence, and continuous charging blocks. Repeated schedules and routes within a fleet are not independent observations.

| Capacity-only scenario | Failed route occurrences | Distinct failed saved schedules | Solutions with ≥1 failure |
|---|---:|---:|---:|
| Saved 240 kWh control | 0 / 487 | 0 / 463 | 0 / 48 |
| Every bus 239.01 kWh (18E2 sensitivity) | 40 / 487 (8.2%) | 37 / 463 | 29 / 48 |
| Every bus 236.44 kWh (18E1 sensitivity) | 195 / 487 (40.0%) | 180 / 463 | 46 / 48 |

All failures are below the preserved zero SOC floor; **none is overcapacity**. Among failed route occurrences, deficits at 239.01 kWh have median 0.520, 90th percentile 0.990 and maximum 0.990 kWh; at 236.44 kWh, median 2.020, 90th percentile 3.090 and maximum 3.560 kWh. Percentiles use the nearest-lower ordered observation. Small absolute deficits do not establish repairability.

Fresh/base: 13/259 route occurrences fail at 239.01 (12/24 solutions), and 101/259 at 236.44 (23/24 solutions). Sequential/warm: 27/228 fail at 239.01 (17/24 solutions), and 94/228 at 236.44 (23/24 solutions).

| Chain | Arm | Route occurrences | Fail at 239.01 | Fail at 236.44 |
|---|---|---:|---:|---:|
| C1 | base | 43 | 2 | 22 |
| C1 | warm | 38 | 6 | 22 |
| C2 | base | 43 | 3 | 11 |
| C2 | warm | 38 | 7 | 12 |
| C3 | base | 43 | 3 | 10 |
| C3 | warm | 38 | 2 | 7 |
| C4 | base | 44 | 2 | 17 |
| C4 | warm | 38 | 5 | 17 |
| C5 | base | 42 | 1 | 26 |
| C5 | warm | 38 | 3 | 20 |
| C6 | base | 44 | 2 | 15 |
| C6 | warm | 38 | 4 | 16 |

## Source-group mapping is a separate scenario

Source trip IDs map to original GIRO vehicle groups using the 40-duty manifest plus the saved k1 duty CSVs, including alternate service-day variants. All trips map. This establishes original trip-group membership, **not a verified physical vehicle assignment for each new route**.

There are 112 homogeneous 18E1 route occurrences, 259 homogeneous 18E2, and 116 mixed-group occurrences. Assigning homogeneous routes their source group's capacity and assigning every mixed route the conservative 236.44 kWh capacity yields **184/487 failed occurrences across 46/48 solutions** (169 distinct schedules). Breakdown: 106/112 homogeneous E1, 3/259 homogeneous E2, and 75/116 mixed routes fail. Mixed routes have no uniquely implied battery; this scenario is explicitly conservative and cannot be presented as an observed actual fleet assignment. Uniform-capacity results above remain separate sensitivity bounds.

## What was replayed

The panel physics are saved as 240 kWh full initial battery, constant 240 kW charging, zero reserve, and no terminal floor. The study uses cover master sense, flat tariff, start fee 5, bus coefficient 100000, 2.5 kWh SOC grid, five-minute blocks, and singleton initialization for fresh runs; warm runs inherit genuine preceding-k pools. The pinned experiment provenance records the exact CG initialization/dependencies. The final-MIP execution commit is `871d057e1067411f09581e37d78f7c1ca43f68bb`; runtime settings and proof status remain in each immutable source result. Those preexisting resource requests were eight solver threads and a 3600-second MIP allowance. This audit uses one local Python process, no scheduler resources or dependencies, and no solver calls.

Every trip, station visit, charging window, **continuous realized charged kWh**, travel energy, power limit, zero reserve and absent terminal floor is frozen. We do not replay conservative grid charging amounts as though they were physical energy. Static reference deadheads follow the producer's symmetric reference lookup with shortest recorded duration. At the saved 240 capacity all reconstructed terminal energies agree with the persisted continuous realization within 1e-5 kWh; route chronology and block power constraints also pass. Between events, SOC is monotone for each consumption or charging action, so event-endpoint extrema suffice for this capacity test. No idle draw is added to this historical zero-idle model.

With frozen energy flows, reducing both capacity and full initial energy from 240 to B shifts the entire SOC trace by **B − 240**. Thus a route fails exactly when its saved minimum SOC is below the capacity reduction (0.99 or 3.56 kWh, tolerance 1e-5). Optimized traces often approach the zero floor, explaining why a small capacity difference can affect many frozen schedules. Reducing initial energy and capacity together cannot create an overfill that was absent before.

The documented charging curve is distinct from capacity: opportunity charging has successive 10%-SOC powers 371.5, 357, 342.5, 328, 313.5, 299, 284.5, 270, 150, 120 kW; depot charging is 60 kW. Changing 240 kW to that curve is **not tested in this capacity-only panel**. Neither are 15% reserve, setup, allowed-group sites, idle draw, charger count, FIFO, platforms or crew constraints. This replay neither changes nor extends any CG certificate, finite-pool MIP proof, whole-fleet dispatch validation, or GIRO target attainment.

## Already matched witnesses and the requested 13309 counterpart

The current fee0/fee5 witnesses in `outputs/week_20260921/cleanup_physics/saved_joint_fee{0,5}.json` already use the documented **236.44 kWh** 18E1 battery. All ten routes pass independent arithmetic replay with their 35.466 kWh reserve, saved per-route terminal floors, 0.1 kW idle draw and documented taper; no capacity approximation is present. See `matched_physics_checks.csv`.

Separately, the historical C6k5 fee0 route index 3 matched to duty 13309 and fee5 index 2 were checked from the counterpart agent's pinned campaign snapshot. At **239.01 kWh (13309's 18E2 capacity)** both selected counterparts remain energy-feasible with unchanged charging and preserved zero reserve: their minimum/terminal SOC values are **0.6670007** and **12.5310003 kWh**, respectively. This narrow success does not certify the documented 15% reserve or taper. Supplemental results for all ten C6k5 routes are in `duty13309_counterparts.csv`; they are **excluded from the 48-solution main-panel totals**.

## Practical implication and artifacts

A posthoc checker is useful, but “rarely infeasible” is not supported for these frozen optimized schedules. A subsequent backtracking/charging-repair stage could test whether affected routes can gain a little energy earlier without violating time, rate, capacity or coupling constraints. **No such repair or reoptimization was performed here**, so route repairability and same-fleet feasibility remain unknown. A negative trace means the exact saved schedule fails, not that its trip assignment is irreparable.

- `per_route.csv`: editable route/scenario metrics, group mapping, schedule fingerprint and source path.
- `per_solution.csv`: editable case/arm/scenario totals.
- `by_chain.csv`, `summary.json`: compact aggregates and failed-route deficit statistics.
- `baseline_checkpoints.csv`: reconstructed continuous 240 kWh event trace; other capacities are a constant shift.
- `replay.py`: deterministic standard-library main replay; run `python3 outputs/research_followup_20260921/battery_rounding/replay.py` from the repository root.
- `replay_counterparts.py`: supplemental historical C6k5 replay, with its separate provenance file.
- `sources/`: unchanged 48 result JSONs, 24 instance CSVs and static reference/manifest copies.
- `provenance.json`, `remote_sources.json`: source hashes, execution commit, source remote paths, resource/scope notes and output hashes.

No source branch changes, cluster submissions, Google Docs or Google Slides edits were made by this audit.
