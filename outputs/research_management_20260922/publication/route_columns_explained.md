# Route columns, charging capacity and compression — 22 September

A route column is one complete, already feasible bus schedule: its trips, deadheads and charging sessions. The master chooses schedules; it does not choose the charging start/end times inside a saved column.

| Part | Entry in route r's column | Constraint |
|---|---|---|
| Trip i | Aᵢᵣ = 1 if this route serves trip i; otherwise 0 | Ax ≥ 1: cover every trip |
| Station s, interval t | Bₛₜ,ᵣ = 1 if this route occupies a plug then; otherwise 0 | Bx ≤ C: respect available plugs |
| Fleet cap, in charging MIP | 1 for every route | Σx ≤ best validated fleet from stage one |
| Objective, stored separately | One bus in stage one; electricity plus the chosen start fee in stage two | Minimize cᵀx |

The final MIP has binary x. CG uses fractional route weights. Battery/SOC feasibility is checked when a route is generated or replayed. Delivered energy is not a separate “1” in the master. The baseline chain matrices have trip rows only; the capacity experiments add station/interval rows.

## A small example

Route R1 serves trips A and C and charges at S from 08:00 to 08:20. R2 serves B and C and charges there from 08:10 to 08:30. Suppose S has one plug.

| Row | R1 | R2 | Requirement |
|---|---:|---:|---|
| Trip A | 1 | 0 | ≥ 1 |
| Trip B | 0 | 1 | ≥ 1 |
| Trip C | 1 | 1 | ≥ 1 |
| S, 08:00–08:10 | 1 | 0 | ≤ 1 |
| S, 08:10–08:20 | 1 | 1 | ≤ 1 |
| S, 08:20–08:30 | 0 | 1 | ≤ 1 |

Selecting both routes covers the trips but violates the middle capacity row. Knowing total energy or counting charging starts would miss this overlap. A different route/schedule column is needed.

## Can start/end times replace all those entries?

We already store charging sessions compactly, with station, times and energy. Under constant delivered power P, energy = P × duration: 20 minutes gives 80 kWh at 240 kW, or 20 kWh at 60 kW. The stricter tapering model also needs arrival SOC and its charging curve. Plug occupancy can include setup/disconnection time, during which no energy is delivered.

There are two exact matrix alternatives for a fixed pool:

1. Merge capacity rows with identical route coefficients and capacity. Occupancy changes only at interval endpoints, so many minute rows repeat. Preserve the existing conservative minute rounding; switching to unrounded endpoints would change feasibility.
2. Introduce occupancy u at station events: uₑ − uₑ₋₁ = Σᵣ(startsₑᵣ − endsₑᵣ)xᵣ, with 0 ≤ uₑ ≤ Cₑ. A session contributes +1 at its start and −1 at its end. These are continuous auxiliary variables; no new binary decisions are needed. This has the same LP projection, not a stronger relaxation.

## Measured matrix reduction: one saved capacity pool

Same 35 trips, 321 route columns, fleet objective and original minute occupancy in all rows below. “Entries” means nonzero coefficients, not bytes.

| Formulation | Rows | Binary variables | Continuous auxiliaries | Nonzero entries |
|---|---:|---:|---:|---:|
| Original minute capacity | 1,789 | 321 | 0 | 44,844 |
| Merge repeated capacity rows | 303 | 321 | 0 | 10,232 |
| Start/end occupancy equations | 307 | 321 | 272 | 7,714 |

Row merging removes 83.1% of rows and 77.2% of nonzeros. Exact coefficient reconstruction and 1,000 rational-weight checks passed. This establishes equivalence and smaller input matrices; solver timing is tested separately. It does not demonstrate faster pricing or new physical feasibility of the saved routes.

## What we are testing

Five frozen pools: fresh C1 k8, C4 k8, C1 k15, C3 k15, and sequential C1 k15. Each receives five 30-minute fleet searches: default, MIPFocus=1, MIPFocus=2, PreSparsify=1, and a better saved incumbent from that same pool. Seed, threads, column order and scientific settings are held fixed. The saved-incumbent arm excludes its earlier acquisition cost and is a diagnostic, not an end-to-end speed claim. Separate diagnostics check safe row/column dominance and certified reduced-cost screening without changing these five-arm matrices.

The small capacity pilot compares the three equivalent formulations above, including LP objectives, presolved size, construction time and MIP time. All results retain full logs and input/code hashes. See the experiment entry point for submitted jobs and verified endpoints; queued work is not a result.

Gurobi already applies presolve and accepts sparse matrices. PreSparsify can reduce coefficients; MIPFocus changes the balance between finding solutions and proving them. Smaller input matrices may presolve to the same model, and occupancy auxiliaries may be eliminated again. We will choose based on measured results. Merging rows during CG additionally needs correct dual handling as new routes split previously identical rows; these first tests are MIP-only frozen-pool tests.

Sources: [Gurobi sparse matrix API](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html#Model.addMConstr), [presolve](https://support.gurobi.com/hc/en-us/articles/360024738352-How-does-presolve-work), [parameters](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#presparsify). Execution uses Gurobi 12.0.3 where recorded; current online documentation describes a newer release, so supported settings are checked at runtime.

Evidence: `outputs/research_management_20260922/charging_column_structure/`; controlled MIPs: `outputs/research_management_20260922/mip_structure/`. Existing 48-case sparsity and timing tables remain in the Fleet times & matrix tab.
