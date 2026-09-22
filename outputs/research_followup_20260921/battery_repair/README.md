# Battery-only charging repair — 21 September 2026

**All 195 failing route occurrences can be repaired at 236.44 kWh without new column generation or changing any trip sequence or station visit.** Of those, 194 fit within the original charging intervals; one needs a charging end extended by approximately **0.150 seconds** at the same station. All 487 route occurrences in all 48 saved solutions pass a forward physical replay under the deliberately preserved historical physics.

This is a narrow, useful repair result: initial/full battery 236.44 kWh, constant 240 kW charging at every modeled charging site, zero reserve, no terminal floor, zero idle draw, static reference deadheads, and no shared charger capacities. In particular, it does **not** establish the documented GIRO taper, 60 kW depot rate, 15% reserve, allowed vehicle-group sites, shared chargers, FIFO, platforms or crew feasibility. Every bus is assigned 236.44 kWh as the requested sensitivity; this is not an inferred actual vehicle assignment for mixed-group routes.

## Measured result

The source is the same unfiltered panel as the preceding audit: six chains × k=5/8/10/15 × fresh/base and sequential/warm. Its 487 selected route occurrences represent **463 distinct saved schedules**; repeated schedules are not independent observations. Totals below include repeated occurrences, so they are accounting totals across experiments rather than one operating fleet.

| Result | Route occurrences | Distinct saved schedules |
|---|---:|---:|
| Already feasible at 236.44, unchanged | 292 | 283 |
| Repaired within existing charging intervals | 194 | 179 |
| Repaired by retiming at the same visits | 1 | 1 |
| Unresolved | **0** | **0** |
| Total | 487 | 463 |

Fresh/base: 100 interval repairs, one timing repair, 158 unchanged. Sequential/warm: 94 interval repairs, 134 unchanged. All **48/48** selected fleets have individually energy-feasible routes afterward; **46/48** had at least one failing route before repair. This does not validate cross-route charger capacity or exact-once passenger service.

| Perturbation | Measured value |
|---|---:|
| Net additional purchased energy across occurrences | 339.004837 kWh |
| Gross energy additions at charging visits | 352.144830 kWh |
| Energy reductions at other visits | 13.139993 kWh |
| Net addition per changed route: min / median / max | 0 / 2.019999 / 3.560000 kWh |
| Gross addition per changed route: min / median / max | 0.019999 / 2.019999 / 3.560000 kWh |
| Gross additions expressed as full-power charging time | 88.036208 minutes total |
| Actual charging-window extension outside original intervals | **0.002499375 minutes total** |
| Continuous electricity-cost increment across occurrences | 33.629280 synthetic tariff units |
| Per-solution cost increment: min / median / max | 0 / 0.661664 / 1.850377 units |
| Positive-energy charging visits before / after | 1,420 / 1,420 |
| Peak charging power | 240 kW |

Six failing routes can be fixed by reallocating energy between existing visits with **zero net extra purchase**. Start fees do not change. The reported electricity-cost increment therefore also equals the charging objective increment; the fleet coefficient is unchanged. These cost units are the saved synthetic flat tariff, not a verified operator currency or bill. We compare continuous realized costs, not the saved conservative grid objective.

The 88.04 full-power-equivalent minutes are **not additional dwell time**: almost all energy is delivered by using existing power slack inside the saved intervals. No source trip is removed, reordered or reassigned. All **11,400 trip occurrences** across these repeated experimental solutions and all original route nodes are preserved.

## Why existing intervals have slack

The selected schedules already carry a continuous realization of conservative event-grid columns. The producer retains charging windows but reduces their charged kWh to avoid overfilling when continuous SOC retains residual energy discarded by grid rounding. For this panel, the saved expanded-grid charging amount equals each window's duration × 240 kW. Across all selected occurrences it is 169,260 kWh, while continuously realized charging is 160,788.439673 kWh: **8,471.560327 kWh of aggregate unused interval power capacity**. Aggregate slack alone would not prove repairability, because it must occur at the right time and SOC; the LP plus chronological replay tests that condition route by route.

The original settings include a 2.5 kWh SOC step and five-minute event-grid blocks, but the repairs use continuous energy amounts/rates and, for one route, continuous timing. **Treat all repaired schedules as continuous witnesses outside the original grid proof.** Their validity does not create a new pricing certificate or establish a new globally optimal routing/charging objective. Original CG certificates and finite-pool MIP proofs remain attached to their original model and artifacts.

## Two concrete repairs

- **C1 k8, sequential/warm, zero-based route 3:** one block at `3127L_0` remains exactly **21:00:00–21:02:30**. Its energy increases from **6.5909996 to 6.74399886 kWh**, adding 0.15299926 kWh. Its average rate remains only 161.856 kW, below 240 kW. Minimum SOC changes from −0.15299926 kWh to zero within numerical tolerance. Every other charge and all timing remain unchanged.
- **C3 k10, fresh/base, zero-based route 1:** the only route whose existing intervals cannot support an energy-only repair. The `2190L_0` visit beginning **12:39:00** increases from **148.459999 to 150.0099975 kWh**. Its old end is **13:16:30**; its repaired end is **13:16:30.1499625**, inside the physical gap before the next trip. Other charging stops are compacted at full power within their original windows; no start moves. The net addition is 1.5499985 kWh and its minimum SOC becomes zero. The original-interval LP explicitly reports infeasible, while the same-visit continuous timing LP succeeds.

The approximately 150 ms adjustment is a mathematical witness under continuous time. The task did not impose a minimum timing resolution or require rebuilding the original event lattice; no claim is made that this precise adjustment can be dispatched or admitted as an original-grid column without further processing.

## Method and independent replay

The first LP freezes each existing charging block's interval and station, with kWh bounded by duration × 240 kW. Linear SOC constraints are imposed before and after every service trip, deadhead and charging block. The objective minimizes the sum of absolute changes in charged kWh from the saved continuous schedule. It is a minimal energy perturbation LP, not a cost-optimal route search. All routes that were already feasible remain byte-for-byte equivalent in charging values/timing.

Only when that LP is infeasible does the fallback permit the existing charge visits to use the gap from arrival after the preceding trip and deadhead to the latest departure that reaches the next fixed trip on time. Trip order, station choice, deadheads and existing visits remain fixed. The fallback minimizes the same absolute energy perturbation, then places a contiguous full-power session as close as possible to its original start. No new visit or charging start is introduced.

The LP engine is local SciPy 1.13.1 / HiGHS. Each LP has a ten-second limit; the measured full pass took approximately **0.47 seconds** and required no cluster job, solver license or full CG. There are no new scheduler dependencies. Source MIPs retain their original execution commit, master sense (cover), initialization and dependency provenance in the preceding audit's immutable source files. This audit's execution commit and source/output hashes are in `provenance.json`.

An independent forward arithmetic routine, separate from the LP constraint construction, checks every occurrence. A second saved-artifact validation pass reloads all witnesses and checks them against original sources: identical trips/order/nodes, original intervals for 486 occurrences, arrival/departure chronology and travel time, deadhead and service energy, charging rate, SOC before and after all events, terminal SOC, all charge records consumed, and horizon. It also confirms all 1,420 positive-energy charging visits remain. No route-coupling constraints are asserted.

All serialized witnesses pass with tolerance 1e-6 kWh. Observed numerical SOC extrema are −3.52e−13 and 236.44000000000008 kWh; observed peak power is 240.00000000000023 kW. These are floating-point residuals.

## Reusable artifacts

- `per_route.csv`: editable occurrence-level before/after metrics, stage, energy, equivalent minutes, actual timing extension, costs and reasons.
- `per_solution.csv`: editable fleet-level repair results and cost increments.
- `repaired_schedules.json`: all 487 complete selected-route witnesses with charge windows/amounts, original trip/node sequences, LP outcomes and replay traces.
- `summary.json`, `independent_validation.json`: measured results and persisted-artifact validation.
- `repair.py`: rerun local repairs with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 outputs/research_followup_20260921/battery_repair/repair.py`.
- `verify_saved.py`: reload and validate with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 outputs/research_followup_20260921/battery_repair/verify_saved.py`.
- `provenance.json`: source hashes, execution commit, physical assumptions, resources, objective, dependencies and artifact hashes.

All original artifacts remain unchanged. No Google Doc or Slides edits were made by this repair audit.
