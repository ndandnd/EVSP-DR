# Current model, GIRO comparison and next tests — 13 September 2026

**The baseline result is strong: all six chains reach 15 buses. We have not yet demonstrated those results under all documented GIRO vehicle and station rules.** This audit changes the explanation and source coverage, not the saved scientific results.

The live [Google Doc](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lumf8xm66fow) contains the table definitions and a concise model comparison. Its existing figures remain in their two figure tabs. Slides are unchanged.

## Reading the inheritance tables

- **Chain:** one nested selection of GIRO duties. Target k counts the GIRO duties supplying the trips, not our resulting fleet.
- **Buses found: all saved sequences:** the final integer fleet after checking all eligible previous-run route representatives for reuse, running CG and solving the saved-pool MIP.
- **Buses found: earlier 512 limit:** the earlier integer incumbent with a bounded inheritance treatment. It is not necessarily the smallest possible fleet in that earlier pool.
- **CG minutes:** this case's import and CG time, excluding previous k runs, original graph construction and the final MIP. See the campaign report for precise `wall_s` telemetry scope.

The importer first validates the previous journal and keeps the cheapest realization for each trip set. Under `--inherit-max-columns 512`, it sorts representatives by decreasing trip count, increasing source cost per trip, then stable trip-ID sequence, selecting at most 512. The separate 900-second replay budget starts after preparation/selection. A selected sequence is reoptimized on the child graph and physically validated; rejected or unprocessed sequences do not become inherited routes. The 512 limit does not limit later CG columns. Full inheritance removes this selection limit; it still uses only the saved representatives, not every conceivable route or every charging realization. New trips receive singleton initialization. No previous LP basis, dual or certificate is transferred, and these runs do not inject GIRO solution routes.

This description is verified in `src/exact_pricer_expanded.py`, `load_column_pool`, `inherit_event_pool` and `_replay_inherited_event_sequence` on the integrated baseline source lineage; implementation and input hashes are recorded by the campaign. Trip-set deduplication is valid only for the baseline row structure: it must not be copied unchanged into a station-capacity master, where charging occupancy distinguishes columns.

**Pool proof columns:** Buses is a feasible saved-pool integer incumbent B. The MIP's saved-pool fleet lower bound L gives L ≤ I_pool ≤ B. “Smallest fleet in pool proved?” means Gurobi has excluded a smaller integer fleet from that pool, with solver tolerances taken into account. For B=16 and L=14, the pool optimum could still be 14, 15 or 16. For B=L=8, eight is proved within that pool. Neither statement alone gives the full-route-space integer optimum. The fleet bound is not the weighted CG objective.

The repaired runs cover C1 k7–15, C2 k9–15, C3 k11–15, C4 k10–15, C5 k11–15 and C6 k11–15. All 37 match k with CG certificates and pool fleet proofs. C1/C2/C4 k10 use ten buses; their CG times are 17.6/16.4/18.8 minutes. Earlier dashes are historical interruptions. [Frozen result table](../queue_recovery_20260912/status_20260913T063519Z/integer_results.csv).

## What GIRO specifies

The [primary-source audit](giro_requirements_audit.md) covers all eight recovered Notes, schedule PDF, activity workbook and deadhead workbook attachments, with hashes and page/sheet references. Raw correspondence and attachments are not newly published by this audit. The [code and experiment audit](constraint_results_audit.md) checks implementation and completed results separately.

| Partille requirement | Current six-chain baseline | What has been tested separately |
|---|---|---|
| Usable battery: 236.44 kWh (18E1), approximately 239.01 kWh (18E2); vehicle groups have separate compatible work | Homogeneous 240 kWh; compatibility not established | Group-specific k1/k2/k3 diagnostics |
| SOC never below 15%; original tasks start full | Full start, zero reserve | 15% floor in the new-physics diagnostics |
| PARX depot: 60 kW; opportunity chargers: SOC-dependent curves, roughly 120–371.5 kW | Constant 240 kW everywhere | PARX60 exact-event pilots; nonlinear/group-specific charging in diagnostics |
| Shared chargers: 2190L1, 4808 1, 3127L2, 7880C1, JON_A1; PARX unlimited | No shared limits | Capacity pilots and continuous overlap checks |
| Recharge target 65%; no documented 65% terminal minimum | No terminal floor | k5 cost comparison matches observed aggregate returning energy, not 65% per bus |
| Minimum opportunity-charge duration, setup, idle draw, route-specific layovers and compatibility | Not all enforced | Some physics/setup in diagnostics; full operational validation missing |
| Platform blocking/FIFO and time-dependent directional deadheads | Not established; base-only/symmetric simplifications remain in relevant diagnostics | Need explicit implementation or declared academic exclusion |

The small battery approximation is not the main issue: at 15% reserve, energy available above the floor is about 201–203 kWh, versus 240 kWh in the baseline. Subtracting the reserve is an equivalent coordinate change only if initial/terminal energy and SOC-dependent charging curves are transformed consistently.

**65% is a recharge target, not a documented end-of-duty floor.** The original Partille workbook's 42 literal tasks all finish below 65% (15.096–61.692%). Source convention: 42 literal records include day variants; a service-day cohort uses 40. Original opportunity charging overlaps fit the documented counts on each admissible 40-duty variant combination. Charger counts alone do not enforce departure-platform interference or FIFO.

Frölunda is a separate instance family: it includes 358.68-kWh groups, KEX60, different opportunity curves/counts, a 60% recharge target and a 20% SOC floor for blocks longer than 20 hours. Do not apply the Partille approximation to all GIRO data. Curve interpolation and grid-side versus battery-side power are not established by the originals. Full crew rules and feeder/demand limits are unavailable; they may be declared outside the academic model rather than invented.

## Completed stricter experiments

[Editable eight-case results](strict_constraint_results.csv): actual 18E usable battery, 15% reserve, nonlinear remote charging, PARX60 and shared charger counts were combined in these small diagnostics. All eight integer outcomes have finite-pool fleet proofs. **None has a full-model pricing certificate:** charging is restricted to full windows and pricing is guarded. Duplicate-service repair and full platform/deadhead operational replay remain incomplete.

| Vehicle group / selected duties | Target buses | Trips | Buses found |
|---|---:|---:|---:|
| 18E1, fewer trips | 2 | 23 | 3 |
| 18E1, fewer trips | 3 | 35 | 3 |
| 18E1, more trips | 2 | 34 | 2 |
| 18E1, more trips | 3 | 51 | 5 |
| 18E2, fewer trips | 2 | 22 | 2 |
| 18E2, fewer trips | 3 | 37 | 4 |
| 18E2, more trips | 2 | 104 | 4 |
| 18E2, more trips | 3 | 150 | 18 |

Target matches are **two of four k2 inputs** and **one of four k3 inputs**. The 18-bus pool result is not a physical fleet lower bound.

A different exact-event k1 pilot, duty13408, combines PARX60 and charger counts: one bus proved in pool, CG certified in 2.8 minutes, route and shared-capacity replay passed. It still uses the baseline battery and zero reserve. A separate k5 tariff experiment matched GIRO's aggregate return energy using 240/350 physics without shared capacity. No experiment combines all source-supported constraints and validation checks.

## What changed over time

| Period / controlled comparison | What changed | What the evidence supports |
|---|---|---|
| April–May | Heuristic DP, covering, older 300-kWh/300-kW assumptions; different seeds/inputs | April 175-trip incumbent12; May different 193-trip input11–12 without GIRO seeds,10 with them. Not interchangeable datasets or full-model proofs. |
| July–August | Safer SOC handling, exact expanded/event-graph pricing, delayed-charge options, physical replay and durable proofs | More reliable feasibility and named-grid LP certificates; not simply memory reduction. The git record has a June gap. |
| September9 matched 175-trip MIPs | Same 240/240 saved pool and one-hour budget; only final coverage sense changes | Cover10, pool bound10; partition34, bound10, unproved. Cover has31 overcovered trips and needs operational duplicate-removal validation. This is stronger evidence about the apparent regression than changing battery size alone. |
| September12–13 | Indexed import, less LP setup, full previous-pool inheritance | All six baseline chains reach15; controlled 512-to-full cases improve9→8,11→10,17→15. |
| Stricter GIRO studies | Reserve, depot power, charging curves and counts added in separate pilots | Useful k1–3 results above, but no completed all-constraints chain ladder. |

This is a history of different experiments, not a controlled curve of performance as constraints accumulate. Adding reserve/capacity/depot limits tightens feasibility. Replacing a constant remote power with GIRO's curve can speed low-SOC charging and slow high-SOC charging, so its effect is not one-directional. Do not attribute chronological fleet changes to one cause.

Historical sources: `HISTORICAL_175_TRIP_REGRESSION_AUDIT.md`, `HISTORICAL_RND002_AUDIT.md`, `CHAIN3_NESTING_AND_DP_CHANGES.md` and `HISTORY_EVIDENCE.md` under `outputs/meeting_20260910/`; current normalized records for `historical_cover_controls` retain original per-result payloads and hashes. The matched one-hour result is distinct from the earlier eight-hour partition incumbent33.

## Improvements and next experiments

Already implemented and cluster-benchmarked: indexed inheritance (12.4–18.1% less total CG time), omitted unused LP setup (9.3–14.6%), full versus 512 inheritance (40.6–64.5%, with richer pools). Do not add percentages. The dedicated capacity path's station-specific cost bug is fixed (`550bc795`) and selector acceleration is implemented (`309d98d2`), but all three real-data comparison pairs reached three-hour limits without certificates. Local speedups do not establish faster production convergence.

Next work, in order:

1. Freeze a Partille model specification and validate all original duty variants under it. Test each added physical rule separately to distinguish source/model mismatch from optimization difficulty. Use the actual full start and 15% reserve. Use observed per-duty returning SOC for a strict matched-energy sensitivity, explicitly as a research comparison condition, not GIRO's supposed 65% floor. Keep the aggregate-energy sensitivity separate because it permits redistribution between buses.
2. Port full inheritance to the capacity path while preserving different station-time charging schedules for the same trips. Validate station-specific pricing, MIP cost and post-selection replay together. A regression should require two occupancy-distinct same-trip routes to preserve a feasible capacity combination. Then rerun the existing four homogeneous k2/k3 cohorts with one added condition per arm, fixed code/inputs/budgets, certificate/stop reason, pool bound, fleet and physical checks. Use the same duty sets instead of declaring the current eight bounded pools a performance threshold.
3. Resume a certified k1 capacity control and compare scan versus indexed charging-window lookup; profile pricing before extrapolating local speedups. In parallel, instrument parent-graph construction before repeating the 750-trip build. Persistent LP delta updates and alternate-dual route generation remain proposals, not explanations of completed results.

No new cluster jobs are submitted by this documentation/audit update. Before execution, read the Unicorn resource policy; independent default arrays use concurrency50 or all cases if fewer, with true previous-k dependencies, excluded `scaglione-compute-01`, immutable inputs/source hashes and separate attempt records. Held history and V2G data remain protected.
