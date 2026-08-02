# Goal-1 local pricing experiments: 2026-08-02

This note records the controlled experiments behind the current Goal-1
decision. The tested branch is `issue20-local-pricing-audit`. The test instance
is synthetic random 15-r04 (`337` trips, SHA-256 prefix `7f1445e1e5da`). GREEDY
constructs 17 feasible seed routes. A verified 15-trip reachability antichain
proves that every fractional route cover in the current pricing graph has route
weight at least 15, and MATCHING constructs a feasible 15-route cover. The
valid rediscovery target is therefore 17 to 15. A result below 15 would be a
correctness failure, not a success.

All runs below used the SciPy/HiGHS restricted master, flat prices, the full
1,560-minute station-to-trip wait, successor-boundary SOC targets, one process,
one BLAS thread, five 60-second pricing calls, 150 returned columns per call,
and no final MIP. Every call timed out and was non-exhaustive. None had a
label-cap eviction.

## Exact result bundle

The eight local pools and both JSON audits are published in GitHub release
[`results-goal1-local-5m-20260802`](https://github.com/ndandnd/EVSP-DR/releases/tag/results-goal1-local-5m-20260802).
After generating the deterministic random inputs, reproduce the local layout
with:

```bash
python -u src/generate_random_goal1_instances.py
mkdir -p src/results/local_goal1
gh release download results-goal1-local-5m-20260802 \
  --repo ndandnd/EVSP-DR \
  --pattern 'evsp_goal1_local_5m_portfolio_20260802.tar.gz*' \
  --dir /tmp/evsp_goal1_local_release
(cd /tmp/evsp_goal1_local_release && \
  shasum -a 256 -c evsp_goal1_local_5m_portfolio_20260802.tar.gz.sha256)
tar -xzf /tmp/evsp_goal1_local_release/evsp_goal1_local_5m_portfolio_20260802.tar.gz \
  -C src/results/local_goal1
```

The archive SHA-256 is
`617b2e8a1821faba60d108e51e648bdbad5328a4e6daca613c0f6b8af4643115`.

## Matched five-minute results

| Pricing configuration | Final LP objective | LP route weight | DP trip coverage | Mean / max trips | Labels expanded |
|---|---:|---:|---:|---:|---:|
| `time` | 1,701,363.153 | 17 | 61 / 337 | 4.351 / 10 | 50,972 |
| `reduced_cost` | 1,701,363.153 | 17 | 97 / 337 | 7.595 / 12 | 64,350 |
| `reduced_cost_bound` | 1,701,363.153 | 17 | 151 / 337 | 10.497 / 19 | 46,385 |
| bound + diversified output | 1,701,363.153 | 17 | 183 / 337 | 10.741 / 22 | 42,177 |
| bound + diversified, after station pass-through fix | 1,701,363.153 | 17 | 179 / 337 | 9.615 / 18 | 44,647 |
| bound + diversified + incidence-diverse dominance | 1,701,363.153 | 17 | 168 / 337 | 8.413 / 12 | 38,883 |
| start-fair bound + diversified + incidence-diverse dominance | 1,701,339.894 | 17 | 280 / 337 | 6.397 / 11 | 56,374 |
| start-fair bound + diversified + resource dominance | 1,701,363.153 | 17 | 319 / 337 | 9.084 / 19 | 62,314 |

Each pool contains 17 GREEDY seeds plus 750 accepted DP columns. The
start-fair queue maintains a bound-priority heap per first trip and pops the
groups round-robin. It sharply improves breadth. The incidence-diverse version
also found a cheaper realization worth 23.259 objective units, but its route
weight remained exactly 17. That is an energy/path-cost improvement of only
about `0.00023` bus fixed costs, not fleet progress.

The fair/resource run reaches 319 trips and routes as long as 19, yet still
does not move the master. Coverage, most-negative reduced cost, number of
negative completions, and maximum route length are therefore useful search
diagnostics but not adequate success metrics by themselves.

## The useful result is the column portfolio

Re-solving one restricted master over the union of all eight saved pools gives:

| Input | Unique trip incidences | DP trip coverage | LP objective | LP route weight | Artificial trips |
|---|---:|---:|---:|---:|---:|
| all eight five-minute pools | 5,669 | 336 / 337 | 1,687,059.989 | 16.857142857 | 0 |

The inputs contain 6,136 route records before cheapest-incidence
deduplication. No single pool or pair of pools lowers route weight. The unique
smallest improving combination is:

- bound + diversified output (`2a710ad`);
- start-fair + diversified + incidence-diverse dominance (`ecb8a31`);
- start-fair + diversified + resource dominance (`ecb8a31`).

That three-pool union has 2,198 unique incidences, objective 1,690,235.216, and
route weight 16.888888889. The all-pool LP uses fractional DP columns from
several search policies together with the surviving GREEDY seeds. This is the
first controlled DP-generated improvement in fleet-weight on random 15-r04.
It is still well above the certified target of 15.

The union audit recomputes every master cost, checks the saved instance, price,
and every pricing-action field recorded by the pools, and solves the union LP.
Older pools did not save every cost constant, so omitted constants come from
the current checkout and are reported as a provenance limitation. The audit
also does not replay the full time/SOC path of every saved route; route
feasibility is inherited from the corresponding runner output. Both limits are
explicit in the JSON report.

This result changes the next engineering step: a bounded queue portfolio that
retains complementary columns is justified; extending one queue from five
minutes to three or twelve hours is not.

## Reduced-cost and complementarity checks

The stored reduced cost of the first 150 audited DP columns agrees with

```text
route master cost - sum(trip coverage duals)
```

to maximum absolute error `1.164e-10`. Adding those individually negative
columns can leave the master objective unchanged because the seed master is
highly dual-degenerate: its HiGHS solution has only 17 positive trip duals, one
per disjoint GREEDY seed route. Another dual solution supports the same master
objective after a batch is added.

Known feasible duties show the same complementary-wave behavior. At one
stalled dual, 11 of 15 were negative; adding them did not move the LP. The
remaining four then became negative, and adding that second wave moved the
master to route weight 15. The current action set can realize all 15 duties,
but no exact duty incidence was present in the DP pool.

A stronger diagnostic avoids historical duties entirely. A deterministic
model-derived matching cover was built from only the active trip graph,
deadhead/time/energy data, current charging rules, and flat prices. It did not
read `VehicleTask` or any GIRO route assignment. The complete audit took about
eight seconds locally and returned 15 resource-feasible routes partitioning all
337 trips, with total cost 1,501,255.911. Starting from the 17 GREEDY seeds,
its currently negative unseen routes entered in waves of 9 and 6; the following
master solve had route weight 15. Starting from the fair/resource 767-column
pool required waves of 7, 5, and 3 before reaching 15. Rebuilding the saved
master from route metadata reproduced its objective and route weight exactly.
The instance and price hashes are verified. Auxiliary deadhead/reference files
and model constants are current-checkout inputs because historical pools did
not record hashes for all of them.

For a partitioning cover `C`, the sum of its route reduced costs equals the
cover cost minus the current master objective. Consequently, while the master
is more expensive than that cover, at least one cover route must be negative.
The audit therefore confirms the master, route costs, dual sign, and
reoptimization logic. It does not prove that the elementary DP can discover
the cover.

## Why the DP misses the useful routes

Exact prefix tracing found two separate mechanisms:

- A known 50-trip duty reached the prefix `{3, 7}`, but cross-incidence
  resource dominance discarded it in favor of `{1, 7}` with equal reduced
  cost, time, and recharge count and higher SOC. The retained prefix conflicted
  with a different duty. The optional incidence-diverse mode preserves equal-
  cost alternatives, but its extra labels reduce depth under a fixed wall time.
- A known 36-trip duty remained feasible, but after 60 seconds its depth-three
  label still had more than 31,000 labels ahead of it. Different global heaps
  starved different first-trip regions. The start-fair queue repairs breadth,
  but strict fairness spends time on many shallow groups and does not by itself
  assemble a fleet-improving bundle.

The search is also discretized. Successor-boundary SOC targets and zero-charge
station pass-through repaired demonstrated feasibility omissions, but current
pricing is not an exact continuous charging oracle. A timed-out or capped run
must never be reported as proof that no improving route exists.

## Dual-centering control

A maximin dual on the exact optimal face gave all 337 trips positive duals
(`eta = 2634.469684`, objective gap about `1e-9`, maximum route-constraint
violation `7.28e-11`). It did not solve the search problem:

- pure centered-dual pricing for 30 seconds expanded 3,800 labels and found no
  negative completion;
- a 90% sparse / 10% centered blend returned 150 columns with mean length
  13.28 and maximum 17, but covered only 38 trips and left route weight 17.

Centering spreads a 100,000 fixed bus cost across many modest duals, so a route
must become deep before it is negative. Treat stabilization as a later
portfolio component, not the first repair.

## Decision and next experiment

Do not launch a 30-minute, three-hour, or twelve-hour repetition of any single
queue from this table. The justified sequence is:

1. Reproduce the saved-pool union with `src/audit_goal1_column_pools.py`.
2. Reproduce the model-only matching-cover waves with
   `src/audit_matching_cover_pricing.py`.
3. Implement a time-budgeted pricing portfolio that combines bound/resource,
   start-fair/resource, and start-fair/incidence-diverse candidates before
   reoptimizing. Report component provenance and aggregate statistics.
4. Give that portfolio a five-minute r04 gate. Continue to 30 minutes only if
   route weight falls materially below 17 or keeps improving across iterations.
5. Keep MATCHING as the operational non-cheating initializer. It already
   supplies the certified 15-route solution on this instance. Use GREEDY only
   as the pricing-discovery control.
6. Reconstruct a verified coherent full-day trip set before making a 43-bus
   parity claim.
7. After flat-price Goal 1 is trusted, add delayed-start charging; immediate
   charging cannot support the temporal demand-response claim.

The union result is encouraging, but it is not convergence: 16.857 is still
above 15, all pricing calls timed out, and the DP has not certified the absence
of improving columns.
