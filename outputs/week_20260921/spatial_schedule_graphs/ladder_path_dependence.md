# Ladder path dependence at k=40 — implementation audit, 21 September 2026

**Within the same final input and model, ladder order can change a time-limited restricted-master solution and its available integer pool. It cannot change the mathematical optimum of the complete final event-graph LP or integer model.** A successful complete pricing certificate establishes the former within the configured numerical tolerances. The actual implementation closes the potential inherited-timing loophole: it rebuilds inherited routes on the child graph before inserting them.

## Which runs really have the same final input?

All six include 40 numeric base duties, but preserved day variants give **three exact final CSV classes**. This audit rehashed each of the six local frozen input copies and checked the manifest cache owner/binding. Filenames may differ; bytes and model identity must agree.

| Chains | Trips | Preserved variants | Full input SHA-256 | Shared graph owner |
|---|---:|---|---|---|
| C1, C4 | 948 | 13316m, 13324muw | `904070ec8919dd11bf431eab62e1396ad4882d95f7d5917bcf83f34d1e20e607` | `w1_k40` |
| C2, C3, C6 | 947 | 13316uwt, 13324t | `3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd` | `w2_k40` |
| C5 | 946 | 13316uwt, 13324muw | `1c53d995701acc515431b296ff51445a8312d66621d6c380ea838dac5056e2fb` | `w5_k40` |

Thus compare C1 with C4, and C2 with C3/C6. C5 has no byte-identical peer. Cross-class comparisons are different instances. Do not change calendar variants mid-chain to force a common endpoint. Shared-cache scheduling is only a computational reuse; each chain retains its distinct inherited pool and separate CG/MIP stages.

## Inherited columns are reoptimized on the child graph

Audited source is pinned to `a0e0bb7681c8451e3cbbbfa06aef390026d9af4b`, archived verbatim under [source_audit_a0e0bb](source_audit_a0e0bb/). Line numbers below refer to that commit and those copies, not an unpinned working checkout.

1. [exact_pricer_expanded.py:562](source_audit_a0e0bb/exact_pricer_expanded.py#L562) validates the parent input hash and maps parent local trip indices through stable `Ordered_Trip_ID` into child indices. Lines 581–598 select only the child trip sequence, stable sequence, and old cost metadata. The parent's charging times/SOC trajectory are not passed into replay.
2. [exact_pricer_expanded.py:407](source_audit_a0e0bb/exact_pricer_expanded.py#L407) calls `child_network.fixed_sequence_record(child_sequence)`. A sequence absent from the child graph is rejected. The new record undergoes physical validation and charging-schedule identity checks. Lines 635–640 label provenance `inherited_event_pool_replayed_in_child_graph`; the old cost is informational.
3. [event_pricer_network.py:1134](source_audit_a0e0bb/event_pricer_network.py#L1134) implements this replay as dynamic programming for the **cheapest child-graph route realizing the prescribed trip order**. All relevant child arcs remain available, including when the optional complete replay index is used (lines 560–622). Returned actions and charging schedules come from child graph edges. Inheritance does not reoptimize the trip order itself.
4. [exact_pricer_expanded.py:2277](source_audit_a0e0bb/exact_pricer_expanded.py#L2277) invokes replay with the actual child network and inserts the rebuilt record, retaining the least-cost record per served-trip set. In this covering model, no station-capacity rows exist: same incidence and higher cost is dominated. Distinct trip-set columns and their discovery order still depend on prior stages. Unlimited configured inheritance is nevertheless subject to the overall CG wall budget.
5. [event_pricer_network.py:42](source_audit_a0e0bb/event_pricer_network.py#L42) constructs the child event times from its trip-station arrivals, deadlines, tariff boundaries and grid. Identical input/model yields the same graph universe. [exact_pricer_expanded.py:97](source_audit_a0e0bb/exact_pricer_expanded.py#L97) binds cache identity to source commit, input/reference/deadhead/tariff hashes and physics/discretization settings. Lines 126–153 validate that identity, the pickle SHA, object type and graph metrics. A differently named byte-identical CSV can safely use the same graph.

Physical replay can realize an abstract graph route with adjusted continuous charging amounts. This does **not** import a parent-only off-grid route or enlarge the certified objective: [event_pricer_network.py:803](source_audit_a0e0bb/event_pricer_network.py#L803) sets master cost to `recomputed_expanded_grid_cost` and explicitly records `continuous_cost_pricing_certified=False`. The certificate is for the conservative expanded-grid model, not the unrestricted continuous-time scheduling problem ([exact_pricer_expanded.py:1248](source_audit_a0e0bb/exact_pricer_expanded.py#L1248)).

## What must agree, and what can differ?

- **Complete graph LP optimum:** for identical graph, coverage sense, RHS, objective and route domain, the optimum weighted value is path independent. Here the route objective is fleet penalty 100000 plus electricity and 5 per charge start. Pricing stops with `certified_rc_optimal` only when a returned minimum reduced cost is at least `-rc_epsilon`; no-path termination is not certified ([exact_pricer_expanded.py:2745](source_audit_a0e0bb/exact_pricer_expanded.py#L2745)). The campaign epsilon is 0.0001. Compare certified values using appropriate pricing/LP numerical tolerance; do not demand bit equality or treat a tolerance certificate as a rational exact proof. Also require feasible coverage and negligible artificials.
- **Four-hour RMP values:** these can differ because different inherited columns and dual trajectories lead to different pools before timeout. An uncertified RMP objective is not a full-model lower bound. Even at the same full LP objective, alternative optimal lambda vectors and duals can differ.
- **Fractional route weight:** `sum(lambda)` is a separate statistic from the weighted LP objective. Equality of the weighted objective alone does not prove equality of route weight: charging cost can offset a change in this continuous quantity. An independently optimized fleet-only or lexicographic objective would support a different claim.
- **Integer outcomes:** the true optimum over the identical complete route universe is also path independent. An exact MIP optimum over each chain's finite pool can differ, because pricing certification only establishes LP optimality and need not include positive-reduced-cost columns needed for integer combinations. A time-limited pool MIP incumbent can differ even on the same pool; preserve its bound and budget as well as its best feasible fleet.

## Eventual comparison rule

Within each exact full40 input-hash group, report weighted LP objective, fractional route weight separately, minimum reduced cost, certificate flag, stop reason, artificials/feasibility, full accumulated CG time (including inherited replay/previous attempts as separately identifiable components), pool size, and finite-pool MIP incumbent plus bound/proof status and stage budgets. Retain graph preparation time separately from pricing time and the sum of predecessor-stage CG costs separately from final-stage time. A per-stage four-hour cap is not equal cumulative ladder work. Compare physical validation and GIRO target attainment independently from the LP/pool certificates.

This is an implementation/theory audit, not a new k40 result: no new job was submitted and no finished/certified k40 endpoint is claimed. The JSON evidence records exact hashes, cache bindings and scientific settings. Parent audits and the campaign launch receipt remain in `../chain_extension_40/`.
