# Mathematical review notes

## Fleet-first pricing and valid stopping

The active baseline minimizes `100000 * route weight + electricity + charging-start cost` in the continuous master, then minimizes integer fleet and charging cost in a saved-column MIP. These are different optimization problems. A large fixed cost can approximate an integer lexicographic preference when a suitable cost bound is proved, but does not by itself make the fractional route weight of a weighted LP a fleet-only lower bound.

An explicit fleet-only CG phase is a useful experiment. For the current uncapacitated covering formulation, write

\[
\min \sum_{r\in\mathcal R}\lambda_r,
\qquad A\lambda\ge\mathbf 1,\quad\lambda\ge0.
\]

For a nonnegative trip-dual vector \(\alpha\), let

\[
\rho=\min_{r\in\mathcal R}(1-a_r^T\alpha),
\qquad q=\max(1,1-\rho).
\]

Then \(a_r^T\alpha\le1-\rho\le q\) for every route, so \(\alpha/q\) is feasible for the full fleet-only dual. Thus

\[
L=\frac{\mathbf1^T\alpha}{q}
\]

is a fleet LP lower bound. This derivation also works with a rigorous lower bound on the pricing optimum in place of \(\rho\). It does **not** work with the reduced cost of a heuristic route as if that were a pricing lower bound. The distinction is central to early stopping: if a tolerance-safe integer lower bound from \(L\) matches a validated incumbent of \(K\) buses that is feasible in the same named route model, the fleet target is proved within this named route space without waiting for every fractional digit to converge. Numerical error margins must be explicit.

This is a standard dual-scaling idea, not a proposed novelty. Farley discusses bounds for early termination of large-column LPs; the unit-cost specialization follows directly from the dual-feasibility argument above. [Farley, 1990](https://pubsonline.informs.org/doi/10.1287/opre.38.5.922); [Desrosiers, Lübbecke, Desaulniers and Gauthier, Branch-and-Price, section 2.3 and Corollary 2.2](https://www.gerad.ca/papers/G-2024-36.pdf).

The formula above deliberately covers the uncapacitated covering model. Capacity rows, fleet caps, terminal-energy rows or other coupling constraints require their duals and right-hand sides in both pricing and the bound. A certificate for a discretized route space is not automatically a lower bound for a less restrictive continuous model.

## Reuse the fleet helper only after adapting its model contract

There is already useful code: `EventExpandedNetwork.min_reduced_cost_route` supports `fleet-only` and a route-count dual, while `certify_fleet_lp_bound.py` runs a separate fleet-pricing phase. This is an integration and validation task rather than a new solver from scratch.

However, at baseline commit `a29992196acb74d02b8c7891be4061718889999f`, the helper's `solve_fleet_master` uses equality rows (`A_eq`, lines 68–75). Its source guard checks a completed weighted-CG certificate and zero artificials (lines 214–217), while `_build_network` reconstructs basic battery/charging/reserve settings (lines 102–131). It does not implement the full current covering/capacity/terminal-policy contract. Its reported tolerance adjustment also differs from the dual-scaling proposal above.

Do not apply that helper unchanged to current covering results and label the output a covering fleet certificate. Add an explicit master-sense choice, model-identity guards and unsupported-feature rejection; price the same model that the bound names. This is a conditional reuse risk, not evidence that the already recorded production results are wrong. [Pinned helper](https://github.com/ndandnd/EVSP-DR/blob/a29992196acb74d02b8c7891be4061718889999f/src/certify_fleet_lp_bound.py#L61).

For a charging-cost comparison at a validated integer fleet cap \(K\), a subsequent charging-only CG model should include \(\sum_r\lambda_r\le K\), its dual in pricing, and the same physical/capacity/terminal constraints. Fixing the fleet at a *fractional* fleet-LP optimum is not a substitute for this integer-fleet comparison. Even a charging LP certificate does not prove the saved-pool integer charging optimum globally.

## Column dominance depends on the full master column

In the uncapacitated baseline, two routes with the same trip incidence and different costs can be represented by the cheaper route: they have identical constraint coefficients. This justifies incidence-based deduplication for that model.

With charger-capacity rows, different charging windows can have different master coefficients despite covering exactly the same trips. Keep the necessary schedule alternatives, or prove dominance using cost **and all** coupling coefficients. Terminal-energy requirements create a similar issue: a cheaper route with lower return energy may not dominate a more expensive route. V2G scenario injection profiles also belong to the column identity. This is why a baseline pool-compression rule cannot be copied blindly into either extension.

Likewise, caching one cheapest charging window is exact only for the objective and dual vector used to choose it. Capacity-aware caches must invalidate or re-evaluate when capacity duals change. Immutable physical/tariff breakpoints can be shared across iterations; the minimizing window generally cannot.

## Safe heuristic pricing and stabilization

A fast restricted search can add useful columns. It may use a sparse subgraph, a short candidate list, old duals or smoothed duals. Before adding a column as an improving one, recompute its reduced cost under the current original master duals. When the heuristic finds nothing, call the unrestricted exact oracle under those original duals before declaring a pricing certificate. An exact oracle under smoothed duals alone does not certify the current RMP solution.

Dual smoothing is well established; its value here would be fewer expensive pricing calls, not a new mathematical contribution. Evaluate it after removing obvious per-call overhead. [Pessoa, Sadykov, Uchoa and Vanderbeck, 2018](https://pubsonline.informs.org/doi/10.1287/ijoc.2017.0784).

## Integer pool quality is a separate objective

The existing fresh-versus-warm chain-3 k=8 comparison has the same LP endpoint but different proved saved-pool integer fleets, nine versus eight. It establishes a pool-complementarity issue, not a different full-model LP optimum. Saving only LP-attractive columns can leave the integer solution short of useful combinations.

Preserve validated predecessor incumbent routes and LP-support routes before selecting additional warm candidates. Compare the already implemented complementary selector against reduced-cost selection at equal budgets. True k-best DAG deviations or a bounded residual pricing/diving step can be later enrichment experiments. Report their effect on time to validated target fleet and charging cost, as well as the exact LP certificate. A larger or more diverse pool is not automatically better under a fixed MIP budget.

## Priority within the paper schedule

First remove provably redundant computation while holding the graph, objective, tolerances and column-selection rule fixed. Then test fleet-only bounds and controlled enrichment as separate algorithm treatments. Defer a full branch-and-price implementation, a new continuous-state labeling solver, or a full dynamic-discretization proof until the smaller experiments demonstrate a need that justifies the engineering and validation cost.

Large fixed-cost management and explicit lexicographic CG have existing precedents. The useful contribution would be their validated application and computational effect in this charging model, with an honest comparison against current baselines. [Desaulniers, Managing large fixed costs](https://www.sciencedirect.com/science/article/pii/S0305054805002066); [Bärmann, Müller and Weninger, Lexicographic column generation, published online 2026](https://link.springer.com/article/10.1007/s00186-025-00905-3).
