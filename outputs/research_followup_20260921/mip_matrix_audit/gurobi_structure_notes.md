# Exploiting the final route-pool MIP matrix

Read-only audit, 22 September 2026. No optimizer was run, no solver setting or implementation changed. Recommendations below are proposed tests, not measured speedups.

**The useful question is what structure remains after existing preprocessing and native presolve, not whether to tell Gurobi that A is sparse.** The pinned implementation already supplies sparse incidence, deduplicates identical trip sets and provides MIP starts. A saved example already invokes sifting automatically. Highest-priority work is to measure residual combinatorial structure and the incumbent/bound bottleneck, then test one justified intervention at a time.

## Verified implementation and existing solver behavior

Inspected `src/run_exact_pool_mip.py` at execution commit `871d057e1067411f09581e37d78f7c1ca43f68bb` using `git show`; file SHA256 `bcb5a6b76040ff6ddfa932433d296a1f0f72207b28cbba738b1b4dd39f1eaac7`.

- Lines 200–280: loading and `deduplicate_pool` retain the lowest stored cost physically admitted route for each `frozenset(trips)`.
- Lines 2442–2483: one binary variable per route; `trip_rows` lists only containing routes, and `quicksum` creates `Ax >= 1` for cover or `Ax = 1` for partition. There is no dense m×n Python matrix.
- Lines 2538–2540: stage 1 minimizes fleet, `sum(x)`.
- Lines 2681–2694: stage 2 adds `sum(x) <= validated incumbent fleet`, supplies that solution as a start, and minimizes `route.cost − BUS_COST_KX`. The cap is not silently treated as a proved fleet optimum.
- Lines 2443–2446 and 2685–2687 already set binary `Start` values. Lines 2384–2386 set time/gap/threads; no custom presolve, cuts or MIPFocus setting is imposed there.

Concrete saved evidence: `sources/c3_k15_fresh_gurobi.log` starts with **302 rows, 38,130 binaries, 896,915 nonzeros**. Native presolve removes **2,665 columns**; automatic sifting solves the fleet root relaxation in **1.42 seconds**, versus an approximately 1,800-second stage budget. Later log summaries list MIR and flow-cover cuts. This example points toward the integer search, rather than missing sparse linear algebra, as the expensive part. It is an example, not a diagnosis of every instance.

The matrix audit agent also verified that stage 2 adds one dense fleet-cap row and exactly n additional nonzeros while retaining n route variables. The complete dimensions/presolve receipt is maintained alongside this note; do not interpret a dense extra row as a dense overall matrix.

Official Gurobi documentation describes native presolve equivalence transformations, including merging parallel columns. This is already solver functionality, not a novel application reduction. [Gurobi: how presolve works](https://support.gurobi.com/hc/en-us/articles/360024738352-How-does-presolve-work).

## Ranked, bounded next experiments

| Priority | Test | Reason and measurement |
|---|---|---|
| 1 | Audit residual exact structure without solving | Count identical row supports, dominated cover rows, cost-respecting column dominance, and connected components of the trip–route bipartite graph. Report additional reductions after the existing incidence deduplication. Never assume sparsity implies substantial reductions. |
| 2 | Improve/compare valid starts within the identical pool | Existing starts are already enabled. Compare their actual fleet/cost/acceptance with stronger validated in-pool witnesses. Distinguish a better start from adding new witness columns: the latter changes the pool. Measure incumbent-at-time curves and fleet-proof time separately. |
| 3 | Stage-specific reduced-cost screening | Use a valid dual bound and an incumbent for the **same objective, constraints and pool**; retain a removal certificate per variable. First report the safely removable fraction. Near-zero reduced costs or a wide gap can make the test ineffective. |
| 4 | Small matched Gurobi parameter comparison | If incumbents are the bottleneck, compare default with `MIPFocus=1`; if proof/bounds are the bottleneck, compare a justified proof/bound focus. Optionally test `PreSparsify=1` only against measured presolved nnz/runtime. Do not combine many changes or claim a guaranteed improvement. |
| 5 | Build/LP-specific changes only if profiling supports them | Compare sparse CSR `addMConstr` against current row building for construction time and memory, with identical ordering/coefficient hashes. Test root-LP method/sifting only on LP-dominated cases; sifting is already automatic in the example. |

Gurobi's `Start` attribute accepts complete or partial MIP starts; `RC` is an attribute of convex continuous models, not a reduced-cost certificate from the MIP incumbent. [Gurobi 12.0 variable attributes](https://docs.gurobi.com/projects/optimizer/en/12.0/reference/attributes/variable.html).

The parameter guide recommends distinguishing incumbent search from proof/bound improvement. It also identifies sifting for very wide LPs (roughly 100 variables per constraint or more), and says automatic selection is normally available. This makes forced sifting a low-priority novelty claim here. [Gurobi parameter guidelines](https://docs.gurobi.com/projects/optimizer/en/current/concepts/parameters/guidelines.html).

`Presolve`, `PreSparsify` and cut generation default to automatic choices. `PreSparsify` may reduce presolved nonzeros; it is not a guarantee of faster solving. The documented special benefit of aggressive clique cuts concerns some **set-partitioning** models; it should not be transplanted to our covering model based only on its name. Likewise, a `CoverCuts` switch is not evidence of a bespoke algorithm for this route set-cover instance. [Gurobi parameter reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html).

`Model.addMConstr` directly accepts a SciPy sparse matrix. Changing APIs can reduce Python construction overhead, but leaves the mathematical model unchanged and does not inherently strengthen its relaxation. [Gurobi Python Model reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html).

For a bounded benchmark, choose predeclared representative cases with easy versus difficult fleet/cost stages, freeze the exact pool/order/cost/start and solver version, and test one intervention at a time. Record preprocessing time, rows/columns/nnz before and after presolve, root-LP time/bound, node count/work, incumbent/bound traces, wall time and peak memory. Repeat relevant settings with controlled seeds; treat the best setting on the tuning sample as provisional until tested on held-out cases. Gurobi supplies a tuning facility, but using it is an experiment rather than proof of structural advantage. [Gurobi 12.0 tuning tool](https://docs.gurobi.com/projects/optimizer/en/12.0/features/tuning.html).

## Which reductions are actually safe?

The following are mathematical deductions for the explicitly stated model, not claims taken from solver documentation.

**Identical route columns.** With nonnegative route costs, binary variables, unit trip requirements and no distinguishing side constraints, keep the cheaper of identical full constraint columns. The current code already does incidence deduplication. Once station-time capacity, vehicle-group counts, route-specific constraints or a different objective are introduced, equal trip sets need not be equivalent; include all relevant coefficients and objective semantics in the comparison. Retain source witnesses and a mapping to removed variables.

**Cover column dominance.** In the pure covering model, if route j covers a subset of route k and k costs no more, j can be omitted while retaining an optimum: replace j by k, or delete j if k is already selected. This uses nonnegative costs and no adverse side-row effect. Equal-cost fleet stage 1 can admit more dominance than stage 2. Removing subsets using only fleet cost may lose the best charging solution; either enforce dominance in the secondary cost as well, or restore the full pool for stage 2. This superset replacement is generally **invalid for exact partitioning**, where extra covered trips violate equalities.

**Cover row dominance.** For unit `>=1` rows, if every route covering trip a also covers trip b, the a row implies the b row. The b row can be omitted from that fixed-pool model. This is not a physical statement that trip b need not be served; the selected full routes still traverse it. It need not hold after new columns are added. Do not apply the same rule mechanically to equality rows.

**Decomposition.** Disconnected components of the bipartite incidence graph permit independent pure-cover fleet minimization. Count them first. The stage-2 total fleet cap couples components unless proved component fleet minima determine the allocation, or the allocation is explicitly coordinated. Shared chargers or other side constraints may connect otherwise separate incidence components.

**No network/TU inference.** A column of A represents an entire route's service set, not a directed arc's one +1 and one −1 incidence. Arbitrary 0/1 route-set matrices can contain the three-column triangle with supports `{1,2}`, `{2,3}`, `{1,3}`; its determinant magnitude is 2, and its fleet LP has a 1.5 solution while the integer cover needs 2. Therefore sparsity and binary coefficients do not imply total unimodularity or an integral LP. This example does not assert that this particular minor was found in each saved A; a dataset-specific non-TU claim requires a witness. Pricing on a time-expanded acyclic graph does not transfer that graph's network matrix to the master.

## Reduced-cost fixing: prerequisites and exact scope

For the nonnegative-variable cover relaxation

`min c'x  subject to Ax >= 1, x >= 0`,

let `y >= 0` satisfy `A'y <= c`. Set `L = 1'y` and `r = c − A'y >= 0`. For any integer feasible solution selecting route j,

`c'x = y'Ax + r'x >= L + r_j`.

Given a feasible incumbent U for this **same** objective, if `L + r_j > U` by a defensible numerical margin, j cannot appear in a solution of value at most U. Removing it preserves at least one optimum. Use strict inequality to preserve equal-value optima. This is a sufficient screening condition; the LP solution need not be unique, and many degenerate columns may have r approximately zero.

Important consequences for this repository:

1. **Fleet objective:** set `c_j=1`. Then this cover-dual construction has `0 <= r_j <= 1`; if the certified fleet gap `U−L` is at least 1, this particular test cannot fix a route. A strong fleet incumbent and close fleet bound are essential. Weighted CG reduced costs for `100000 + charging + start fees` are not fleet reduced costs.
2. **Stage 2:** use charging cost and include the fleet-cap dual. For the additional constraint `1'x <= K`, a valid formulation uses multiplier `mu >= 0`, `r = c − A'y + mu*1 >= 0`, and `L = 1'y − mu*K`. Omitting this row gives the wrong certificate for that LP. General bound/side-row dual contributions must also be accounted for.
3. **Variable bounds:** the simple derivation above uses nonnegative variables without explicit upper-bound dual terms. A solver relaxation with `0<=x<=1` may report negative reduced costs for upper-bound variables. Do not assume all `RC` values are nonnegative or insert them blindly into the formula; either construct this valid dual explicitly or use a bound-aware fixing derivation.
4. **Pool versus full model:** checking `A'y<=c` on the saved columns certifies only that finite pool. A corresponding pricing certificate is required to cover all admissible routes of the same model/objective. A restricted-master objective alone is not a full-model lower bound. An approximate pricing tolerance requires a justified bound adjustment, not an unqualified exact-dual claim.
5. **Do not conflate proof scopes:** a valid incumbent supplies an upper bound; a fleet MIP bound concerns fleet; a charging bound under an unproved fleet cap concerns only that capped pool; a conservative event-grid certificate does not automatically certify continuously repaired costs or strict GIRO physics.

## Version and evidence notes

The saved runs are Gurobi **12.0.3**. Official 12.0 variable-attribute and tuning pages were accessible; the browsed current parameter/API pages now describe Gurobi 13.0. Only longstanding mechanisms were recommended, and any actual experiment must validate parameter availability/defaults in 12.0.3. In particular, this note does not propose the newly documented `PreSparsify=2` option; the conservative test is default versus 1. No third-party web claims are used.

No source code, live document, slides, scheduler state, or experiment proof flag was modified by this note. No optimizer or cluster submission was run.
