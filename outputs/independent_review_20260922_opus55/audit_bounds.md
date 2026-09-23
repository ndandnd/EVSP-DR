# Audit of Opus 5.5 F7 and the proposed fleet-cap rule

22 September 2026. Bounded local source/data audit; no solver, SSH, cluster submission, publication, or Git operation. **No new fleet lower bounds were computed or certified.**

**Verdict:** extending the existing reduced-cost bound audit is useful and may require no optimization. F7's three proposed numerical floors remain unverified; `Q ≈ 1040` is an assumption, not a route-cost certificate. Applying the helper indiscriminately to every endpoint is unsafe. The recommendation `fleet_cap = ceil(LP)` also needs a precise definition of the LP and its certificate.

## What is already established

This is an existing project method, not an unused new direction. [The earlier F3 audit](../independent_review_20260916/execution/f3/README.md#derivation-and-implementation) reports 52 audited uncertified endpoints in a frozen 102-case cohort, with per-instance route-cost envelopes bounded by 1044.008. Its CSV preserves the original data and adds numerical fleet floors. That cohort-wide envelope is not automatically an envelope for the later k33–36 cases or strict/capacity models. F3 explicitly distinguishes fleet bounds from fractional route weights and the fleet-only LP optimum.

The pure helper is [fleet_bound.py](/Users/nadan/Documents/projects/demandresponse/.codex-work/early-stop-20260917/src/fleet_bound.py:7). For the covering master, let `M = 100000`, route cost `c_r = M + e_r`, and `0 <= e_r <= Q` for **every route in the pricing model**. Let `z` be an optimal, artificial-free RMP objective and `delta` the exact global minimum reduced cost at that same solve's duals. Then

```
K = z / M
L = z + K * min(delta, 0)
integer_fleet_lower_bound = ceil((L - guard) / (M + Q))
```

`L` bounds the full weighted LP objective in that model. The conversion to an integer fleet floor additionally needs the upper cost envelope `Q`. The code uses a one-objective-unit guard and tiny ceiling slack; these are numerical precautions, not a proved floating-point error bound. The result is a numerical model certificate, not exact arithmetic, a continuous-physics certificate, or a pricing-convergence certificate. A fleet lower bound can be valid even while CG remains uncertified. Equality to `ceil(route_weight)` is the helper's early-stop/saturation condition, not what makes arbitrary route weight a lower bound.

For the audited baseline, `Q = power * horizon_hours * max_tariff * premium + start_fee * (q + 1)`, where `q` bounds the number of nonoverlapping trips on any route. See [the envelope implementation](/Users/nadan/Documents/projects/demandresponse/.codex-work/early-stop-20260917/src/fleet_bound.py:101). Verify the run's tariff, horizon, maximum station power, fee, trip times, charging-activity count, and nonnegative nonfleet costs. A maximum observed cost in the saved pool is insufficient. The historical value 1044.008 must not be rounded down to an assumed 1040 or carried into larger/different instances without proof.

## Exact iteration association in the three cited endpoints

The reviewed [endpoint collector](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260922/monitor_20260922T195842Z/operations/audit.py:19) combines `final_lp.objective` with `final.min_rc`. These are useful descriptive fields but are not automatically a matched pair for bound calculation. Source code stores `lp_obj`, `min_rc`, route weight and artificials together in each priced iteration; it may subsequently re-solve the enlarged pool without repricing it. See [paired record construction](/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/spatial_schedule_graphs/source_audit_a0e0bb/exact_pricer_expanded.py:2697) and [final re-solve](/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/spatial_schedule_graphs/source_audit_a0e0bb/exact_pricer_expanded.py:3082).

These are existing scalar observations, not new bound calculations:

| Case | Completed priced iteration | Its `final.lp_obj` | Its `final.min_rc` | Saved `final_lp` source |
|---|---:|---:|---:|---|
| w5_k33 | 226 | 3201404.3606701414 | -0.03184632719441627 | `final_pool_resolve`; objective 3201404.360670194 |
| w1_k35 | 51 | 3401539.076369124 | -36.0531837000212 | `last_good_iterate`; matching objective |
| w3_k35 | 180 | 3501465.8731186157 | -0.0539319463777268 | `last_good_iterate`; matching objective |

All three paired records report zero artificials. For w1_k35 and w3_k35 the retained matching trip duals are nonnegative; their sums agree with the respective objectives to about 2.0e-8 and 2.4e-8. For w5_k33 the saved duals belong to the later pool re-solve; do not use them as the priced dual vector even though objectives differ only in the eighth decimal place and the iteration number is still 226. Its paired scalar record can support the existing numerical approach through the successful LP solve's strong duality; an independently replayable certificate needs the matching dual vector or another pricing pass.

All source payloads are under `outputs/research_management_20260922/monitor_20260922T195842Z/operations/baseline/cases/`:

| Case | Relative payload | Verified SHA-256 |
|---|---|---|
| w5_k33 | `w5_k33/cg/661707_r0/cg.json` | `25bc97be2fca637fe613a744f233a5c7b303216ddcdd9d7264df218d35d1fd28` |
| w1_k35 | `w1_k35/cg/661622_r0/cg.json` | `bfb145cba8c6ca18145357d34c9b5766cc514b3dc0263c67cec8ac62959d64ae` |
| w3_k35 | `w3_k35/cg/661676_r0/cg.json` | `f4433a1afa1c61504606bf3934cc512a76afb5908a14b0260615479e79b7a3bf` |

The three input CSVs exist locally under `outputs/week_20260921/chain_extension_40/inputs/`, and their hashes match each payload's `instance_sha256`. The local flat tariff in the early-stop worktree also matches the recorded hash. Thus input availability does not block a solver-free per-instance envelope audit. This bounded audit has not performed that envelope calculation or verified the executed dirty source tree. The pinned archived pricer computes the global shortest path first and keeps it as batch element zero ([source](/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/spatial_schedule_graphs/source_audit_a0e0bb/event_pricer_network.py:984)); provenance of the actual execution must still support applying that source argument.

## Why “every endpoint, including strict runs” is too broad

The strict k16 payload has 2,453 completed pricing iterations, but its terminal exact reduced cost is null after `pricing_deadline`. Its last completed iteration pairs objective 900393.2305037788 with reduced cost -20.344798091493093; the final objective is a different value, 900393.1896050631. A future audit must use the completed iteration and validate its own strict-model envelope and native route validity. [Source payload](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/operations/strict_packed/cg/646675_r0/result.json).

The strict k19 payload has `iterations: []`, `terminal_exact_min_reduced_cost: null`, and `stop_reason: cg_wall_limit`. There is no completed pricing record to use in this formula. [Source payload](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260922/monitor_20260922T075504Z/operations/strict/k19/cg/result.json:33).

**Caller hazard:** [fleet_lower_bound line 185](/Users/nadan/Documents/projects/demandresponse/.codex-work/early-stop-20260917/src/fleet_bound.py:185) converts `min_rc=None` to zero, and its API defaults artificial mass to zero. A reporting wrapper must refuse missing pricing or missing artificial evidence before calling it. The returned `certified` boolean does not authenticate exact pricing, iteration identity, Q, model feasibility, or execution provenance. An absent terminal reduced cost cannot stand in for a nonnegative minimum reduced cost.

Capacity rows, fleet caps, global energy rows or other changed masters require the appropriate full dual objective and all reduced-cost terms; do not transplant the coverage-only derivation without checking them. Uniform-power assumptions must become maximum-power bounds for strict station-specific power. A capacity-omitted model floor, if valid, remains separate from solving or validating capacity-constrained schedules.

## The fleet-cap and interpretation corrections

- `ceil(LP)` is valid as a fleet lower-bound target if it means the optimum of a **certified fleet-only full LP**, with numerical rounding handled. A restricted-pool fleet LP is insufficient.
- In this project CG minimizes `100000 * route_weight + charging-related cost`. Its weighted objective, fractional route weight, and fleet-only LP optimum are different. Even at weighted-pricing convergence, simply rounding that solution's route weight is not a general proof of the minimum integer fleet. Use the validated Q conversion above or an independently proved fleet bound such as concurrency.
- A proved fleet floor is a sensible **heuristic trial cap**, not evidence that an integer solution at the cap exists. The algorithm needs an escalation/restart policy; an exhausted or timed-out dive cannot prove global infeasibility. The historical cap of eight should keep its actual provenance rather than be retrospectively relabelled as LP-derived.
- F7's statement that floors below GIRO's count are “likely a relaxed-physics artifact” is unsupported by a lower bound alone. A lower bound below a reference count does not demonstrate a feasible solution with fewer buses. A matching valid incumbent and consistent model are needed for that conclusion.
- Publishing a separate audited fleet floor would not turn an uncertified RMP objective or route weight into a certified LP endpoint. Retain the pricing status and report the new floor, derivation, scope, matching iterate and validated incumbent separately. It also does not prove the finite pool has an integrality gap.

**Recommended disposition:** accept F7/action 1 as a proposal to extend the already established offline audit. Reject the three speculative floors as present certified results and qualify its blanket scope. Accept a bound-based trial cap after replacing ambiguous `ceil(LP)` with a verified integer fleet lower bound. Preserve the original review unchanged; record these corrections alongside it.
