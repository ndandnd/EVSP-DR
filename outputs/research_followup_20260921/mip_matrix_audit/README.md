# Actual MIP matrix audit: fresh versus sequential cohort

All **48 endpoints / 24 pairs** were checked against their pinned result JSONs, native Gurobi 12.0.3 logs, execution-state receipts and exact solver source at commit `871d057e1067411f09581e37d78f7c1ca43f68bb`. All 577 consistency checks pass. No solver was run.

The final MIP matrix is a **trip-by-route 0–1 incidence matrix**, not the millions-of-arcs pricing network. Each admitted route is one binary variable; each trip is one covering row. Stage one minimizes the number of routes. Stage two retains the variables and covering rows, adds one dense `sum(a) <= stage1_buses` row, and changes the objective to charging-related route cost. Its matrix therefore has **one additional row and exactly one additional nonzero per column**.

## C1 original fleet-stage matrices

| Target | Pool | Trip rows | Binary route columns | Nonzeros | Density | Mean trips / column | Presolved columns |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | fresh | 104 | 28,549 | 596,747 | 20.099% | 20.903 | 28,549 |
| 5 | sequential | 104 | 27,340 | 623,626 | 21.933% | 22.810 | 23,289 |
| 8 | fresh | 194 | 39,940 | 1,092,654 | 14.102% | 27.357 | 39,742 |
| 8 | sequential | 194 | 78,053 | 2,107,909 | 13.921% | 27.006 | 77,236 |
| 10 | fresh | 225 | 50,585 | 1,329,898 | 11.685% | 26.290 | 50,585 |
| 10 | sequential | 225 | 84,209 | 2,237,466 | 11.809% | 26.570 | 82,193 |
| 15 | fresh | 364 | 79,611 | 2,322,459 | 8.014% | 29.173 | 79,566 |
| 15 | sequential | 364 | 130,468 | 3,541,122 | 7.457% | 27.142 | 129,896 |

Density is `nonzeros / (rows × columns)`; mean trips per route column is `fleet_nonzeros / binary_columns`. It is an arithmetic average over all admitted columns, not the number of trips in the selected bus schedules. The CSV records both stage matrices and each native presolved matrix. It does not infer the per-column minimum/median/maximum from an average.

## What changes before model construction

The pinned code first validates journal records (nonempty unique trip IDs, valid finite cost, no unknown trips), then for covering runs keeps the cheapest record per identical trip-incidence set. It replays/maps each resulting route under the physical gate, then deduplicates admitted routes again, retaining the cheapest incidence-equivalent route. This is application-level preprocessing before Gurobi presolve.

Across these 48 saved endpoints, the declared CG column count equals the pre-replay unique count, accepted count, post-replay unique count and final binary-variable count. There are **zero physical rejections, zero deterministic repairs, zero added GIRO columns and zero extra-route sources**. Thus no observed column reduction occurs between the physical-gate input and final MIP. Raw journals total 8.03 GB and were neither downloaded nor scanned; their independent raw record counts and any initial duplicate journal records are deliberately left unknown. Declared CG counts are labeled separately from those unknown raw counts.

## Presolve, sparsity and starts

- The application builds sparse incidence lists `trip_rows[t]` and adds only the route variables that cover each trip (`run_exact_pool_mip.py:2473–2483`). There is no dense Python matrix construction. Native logs show all matrix coefficients equal one and all variables binary.
- This pinned application does not explicitly set `Presolve`, `PreSparsify`, `Method`, `MIPFocus`, `Heuristics` or `Cuts`, and does not implement reduced-cost variable fixing. The native logs report only the time allowance and eight threads as non-default optimizer parameters. `MIPGap=0.0001` is explicitly assigned; the source records Seed 0 as the Gurobi default. An absence of application fixing does not mean Gurobi internally performs no reductions.
- All 48 endpoints already assign a complete validated greedy-pool-partition MIP start, accepted by Gurobi. Stage two explicitly assigns the validated stage-one solution as a new start; all 48 stage-two starts were accepted. Sequential describes the inherited column-generation pool, not an otherwise absent versus present MIP-start mechanism.
- Presolve dimensions are read directly from the native stage-specific logs, not inferred from the final result JSON. Fleet presolve removes 0–7,168 columns in fresh runs and 500–6,468 in sequential runs. Charging-stage presolve is separately recorded and can retain more columns because the objective and fleet-cap row differ.

## Interpretation

Across the cohort the original incidence density ranges **6.392%–21.933%** (approximately 78.067%–93.608% zeros), with **13.769–29.173 trips per route column on average**. The model is wide, with 79–385 trip rows and 10,006–130,468 route columns. Its sparsity is meaningful, but these are not matrices with only one or two nonzeros per route column.

Sequential models have **more columns in 20/24 pairs** and **more nonzeros in 19/24**, yet their fleet-stage optimizer time is lower in all 24 pairs. Matrix size alone therefore does not explain the timing advantage; this observation does not establish a causal effect of any presolve setting, basis, solver seed or particular structural feature. The pools can differ in column content and integer usefulness. Finite-pool proof, target attainment and physical dispatch validity remain separate.

## Files and provenance

- [Full editable 48-endpoint table](matrix_dimensions.csv) and [JSON](matrix_dimensions.json): original and presolved dimensions for both stages, density, mean coverage, preparation counts, MIP starts, fleet times and all endpoint/log source hashes.
- [Verification and aggregate summaries](verification.json): 577 checks, pinned source hashes and paired size comparisons.
- [Collection receipt](collection.json): exact remote log/state paths, hashes, CG metadata and original journal sizes. Only 1,196,101 bytes of logs/states were collected, plus small metadata.
- [Exact execution source](sources/run_exact_pool_mip.py), [collector](collect.py), [audit script](audit.py). All 48 full Gurobi logs and execution-state receipts are in `sources/`; the already-pinned MIP JSONs remain at their original paths.

No queue-wide polling, jobs, solver submissions, whole-journal transfers, live-document or Slides changes were made.
