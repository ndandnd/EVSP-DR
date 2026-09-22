# **Route columns and charging capacity**

22 September 2026 · A route column is one complete bus schedule: trips, deadheads and charging sessions. The master chooses schedules; charging times inside a saved column are already fixed.

| Part of a column | Entry | Requirement |
| ----- | ----- | ----- |
| Trip i | 1 if served; 0 otherwise | Ax ≥ 1: cover every trip |
| Station s, interval t | 1 if a plug is occupied; 0 otherwise | Bx ≤ C: respect plug capacity |
| Fleet cap (stage two) | 1 for each route | Σx ≤ best validated stage-one fleet |
| Objective (separate from A/B) | One bus, or charging-related cost | Minimize cᵀx |

The final MIP has binary x; CG uses fractional route weights. SOC/energy feasibility is checked in route generation and replay. Energy delivered is not another “1” in the master. The baseline chain matrices have trip rows only; capacity experiments add station/time rows.

## **A small example**

R1 serves A and C, charging at S from 08:00–08:20. R2 serves B and C, charging there from 08:10–08:30. S has one plug.

| Row | R1 | R2 | Requirement |
| ----- | ----- | ----- | ----- |
| Trip A | 1 | 0 | ≥ 1 |
| Trip B | 0 | 1 | ≥ 1 |
| Trip C | 1 | 1 | ≥ 1 |
| S, 08:00–08:10 | 1 | 0 | ≤ 1 |
| S, 08:10–08:20 | 1 | 1 | ≤ 1 |
| S, 08:20–08:30 | 0 | 1 | ≤ 1 |

Both routes cover the trips but conflict from 08:10–08:20. Total energy or charging-start counts would miss this overlap.

## **Compact times versus a compact matrix**

We already store station, charging times and energy. At constant delivered power, energy \= power × duration: 20 minutes gives 80 kWh at 240 kW, or 20 kWh at 60 kW. With tapering, arrival SOC and the charging curve matter too. Plug occupancy can include setup/disconnection time with no energy delivered. Two routes with the same trips but different charging times can be different columns when capacity is enforced. Plug counts are binary; a site-power limit would instead use kW coefficients.

**Option 1: merge repeated capacity rows.** For a fixed pool, many minute rows have identical route coefficients and capacity. Keeping one is exact. Preserve the original conservative minute rounding; replacing it with unrounded endpoints changes feasibility.

**Option 2: track occupancy at start/end events.** Let u be plugs occupied. Then uₑ − uₑ₋₁ \= Σᵣ(startsₑᵣ − endsₑᵣ)xᵣ, with 0 ≤ uₑ ≤ Cₑ and zero occupancy before the first event. Each session contributes \+1 at its start and −1 at its end. These extra variables are continuous. The LP projection is unchanged, not strengthened.

## **Measured reduction on a saved capacity pool**

Same 35 trips, 321 routes, fleet objective and original minute occupancy. Entries \= nonzero coefficients.

| Formulation | Rows | Binary vars | Extra continuous vars | Nonzeros |
| ----- | ----- | ----- | ----- | ----- |
| Minute capacity | 1,789 | 321 | 0 | 44,844 |
| Merge repeated rows | 303 | 321 | 0 | 10,232 |
| Start/end equations | 307 | 321 | 272 | 7,714 |

Row merging removes 83.1% of rows and 77.2% of nonzeros. The merged matrix is 89.48% zeros versus 92.19% originally: its dimensions shrink faster than its nonzero count. Percentage sparsity alone does not predict work. Exact coefficient reconstruction and 1,000 rational-weight checks passed. This proves equivalent matrices; solve-time improvement is a separate test. It is not new physical validation or faster pricing.

## **Controlled tests and evidence**

Five frozen pools: fresh C1 k8, C4 k8, C1 k15, C3 k15; sequential C1 k15. Each gets five 30-minute fleet searches: default, MIPFocus=1, MIPFocus=2, PreSparsify=1, and a better saved incumbent from the same pool. Seed, threads, ordering and scientific settings are fixed. The saved-start arm excludes its earlier acquisition cost: it diagnoses search, not end-to-end speed.

First four pool diagnostics: one connected block each, no duplicate columns or redundant rows, and only 1.1–6.9% safely dominated columns. Fleet reduced-cost screening removes zero columns in the three completed fresh cases, versus 8,458/130,468 (6.5%) in sequential C1 k15. No reductions are applied to the parameter trials. **Capacity pilot completed (729675).** All three LP values are 3.000 and all three MIPs prove three buses within this saved pool. Selected solutions pass the original trip and minute-capacity rows.

| Formulation | Build, s | MIP, s | Presolved nonzeros |
| ----- | ----- | ----- | ----- |
| Original minute rows | 0.207 | 0.338 | 5,009 |
| Merged rows | 0.028 | 0.058 | 5,596 |
| Start/end equations | 0.042 | 0.099 | 6,779 |

This single tiny, fixed-order run confirms equivalence. It does not establish a general speedup; native presolve already shrinks the original model substantially.

 Fleet tests: loading defect repaired; affected setup attempts retained separately. All five C4 k8 arms prove nine buses. Sequential C1 k15 with a saved incumbent proves 15 in 11.15 seconds, excluding its acquisition time. Pending trials are not results. Logs and recovery receipts are linked below.

Gurobi already uses sparse matrices and presolve; smaller input need not mean a smaller presolved model. These tests keep columns fixed. Inside CG, compression also requires correct updates to capacity rows and pricing duals when new columns arrive.

Sources: [Gurobi sparse matrix API](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.addMConstr) · [presolve](https://support.gurobi.com/hc/en-us/articles/360024738352-How-does-presolve-work) · [parameters](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#presparsify). Execution settings are checked against the installed version. Evidence: [charging-column formulations, input and logs](https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/research_management_20260922/charging_column_structure/README.md) · [controlled MIP tests, jobs and source hashes](https://github.com/ndandnd/EVSP-DR/blob/codex/week-evidence-20260921/outputs/research_management_20260922/mip_structure/README.md). 48-case audit: [Fleet times & matrix](https://docs.google.com/document/d/1hDSWYb2KG-8pnLFN9BdSKkbXOjhytm4bxQz_MsBw_BY/edit?tab=t.lt33xg84cn65).