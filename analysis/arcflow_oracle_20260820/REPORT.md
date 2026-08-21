# Independent arc-flow oracle — executed local results

Date: 2026-08-20. All runs were local. No cluster jobs were submitted.
Machine-readable rows are in `results.csv`.

## Formulation and proof scope

`src/arcflow_oracle.py` imports `ExpandedNetwork` from
`src/exact_pricer_expanded.py`; it does not reconstruct the SOC-by-time graph.
The model has flow conservation, exact trip coverage, source flow as fleet,
and the pricer's stored arc costs. Arcs outside every source-to-sink path are
elided as variables fixed to zero. The physics are 300 kWh, 300 kW, zero
reserve, and the flat historical tariff.

The direct all-arc-integer HiGHS solve exhausted the first 47 GiB local VM.
Subsequent witness searches imposed integrality on trip-service arcs, an
explicit relaxation of the requested all-arc-integer model. Every reported
integer witness nevertheless had **every arc value integral** and was
decomposed into source-to-sink routes. Therefore each is feasible for the full
all-arc-integer model. A certified set-partitioning LP lower bound whose
ceiling equals the witness fleet proves the full model's fleet optimum by
sandwiching; no claim relies on the service-integrality relaxation being exact.
G4 physically replayed every decomposed route and audited exact trip coverage.

## Gates

- **G1 passed** on all nine primary cells. Full node/arc counts exactly matched
  the pricer. Exact reachability presolve retained 138,181 to 883,745 arcs.
- The tracked repository contains no ladder column journals (generated pools
  are ignored). For **G2**, matching one-iteration exact-pricer journals were
  generated locally. A charged multi-trip route in every primary cell mapped
  back to one continuous DAG path at the identical stored grid cost.
- **G3 passed** on all nine primary cells. The arc-flow fleet LP equalled the
  certified set-partitioning fleet LP to solver tolerance in every cell.
- **G4 passed** for seven primary cells: all three k2 cells, all three k3
  cells, and k05_s2.
- **G5 passed** wherever an integer witness was found: no arc-flow fleet
  exceeded its RAW pool-MIP fleet.

The fine-grid k02_s1 network also passed G1 and G2: 576,816 nodes and
16,950,819 arcs, of which 8,186,924 survived reachability presolve. Its
interior-point LP returned no primal within 607.265 seconds. Per the gate
ordering, no fine-grid integer solve or later fine-grid cell was attempted.

## Primary-grid integer findings

| instance | LP lower bound | proven arc-flow fleet | RAW pool MIP | pool excess |
|---|---:|---:|---:|---:|
| k02_s1 | 2.181818182 | **3** | 4 | 1 |
| k02_s2 | 2.187500000 | **3** | 4 | 1 |
| k02_s3 | 2.274725275 | **3** | 7 | 4 |
| k03_s1 | 3.181818182 | **4** | 5 | 1 |
| k03_s2 | 3.404697987 | **4** | 10 | 6 |
| k03_s3 | 3.000000000 | **3** | 4 | 1 |
| k05_s1 | 5.323651452 | unresolved (6–11) | 11 | unresolved |
| k05_s2 | 5.000000000 | **5** | 6 | 1 |
| k05_s3 | 5.000000000 | unresolved (at least 5) | not given | unresolved |

For k02_s2, the three-bus witness had objective 300,070.592 and exact
coverage of all 29 trips. This independently agrees with corrected
branch-and-price, which proved three with a different replayed incumbent cost
of 300,148.744. The independent fleet agreement is exact; charging schedules
need not agree because the branch-and-price run did not optimize charging.

## Correction to the LP premise

The executed arc-flow LP is **not strictly below** the set-partitioning LP:
it equals it on all nine primary cells (for example, k02_s2 is 2.1875 in both).
This is structurally expected for this particular model. Any nonnegative flow
on the acyclic expanded network decomposes into source-to-sink path flows;
each path is exactly a route in the set-partitioning formulation. Conversely,
every route induces an arc flow. Decomposition preserves trip incidence,
fleet flow, and cost. With unlimited station capacity and no additional
cross-route coupling constraints, the two LPs are extended formulations of
the same path-flow relaxation. The non-strict ordering in the work order holds,
but its prediction of strict inequality does not.

## Required answers

**1. At k2 on the primary grid, the true discretized integer optimum is 3
buses, not 4.** This is proven independently for every k2 replicate by a
certified LP lower bound above two plus a physically replayed, fully integral
three-bus arc-flow witness. For k02_s2 it independently matches corrected
branch-and-price.

**2. Pool composition fails wherever a primary integer witness was found.**
The exact excess of the RAW pool MIP is 1, 1, and 4 buses for k02_s1/s2/s3;
1, 6, and 1 for k03_s1/s2/s3; and 1 for k05_s2. Thus the original k02_s1
2-to-4 result combines one bus from discretization (GIRO 2 versus model 3)
and one bus from the restricted pool (model 3 versus pool 4).
