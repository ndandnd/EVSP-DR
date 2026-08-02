# Goal-1 status and next decisions

This is the short project checkpoint for a fresh contributor. The maintained
branch is `issue20-local-pricing-audit`, based on `issue20`. Do not merge
`main`: it predates the maintained DP line and mixes obsolete generated files
with conflicting model/data changes.

## Bottom line

Goal 1 is no longer blocked by Gurobi or by the absence of a feasible warm
start. The free SciPy/HiGHS restricted master and the model-derived MATCHING
initializer produce all-real covers of the hard first-10 and first-15 trip sets
using 10 and 15 routes. Their peak-concurrency lower bounds are also 10 and 15,
so the fleet counts are certified for those two restricted instances.

This does **not** mean the pricing DP is healthy. The released GREEDY runs
started at 13/21 routes and did not improve, and known historical duties were
negative-reduced-cost columns that pricing failed to rediscover. Current fixes
make all 10/10 and 15/15 known duties representable, but a controlled GREEDY
test must still show that pricing can move a weak seed toward the certified
fleet count.

## What the released five-hour runs said

| Run | Initial/saved real columns | DP columns | Final evidence |
|---|---:|---:|---|
| hard 10 GREEDY | 13 seed routes | 1,050 | LP route weight stayed 13 |
| hard 15 GREEDY | 21 seed routes | 450 | LP route weight stayed 21 |
| hard 10 NO_CHEAT | none | 4,055 | only 113/329 trips had any real-column coverage |
| hard 15 NO_CHEAT | none | 3,894 | only 132/473 trips had any real-column coverage |

GREEDY and MATCHING are both non-cheating: neither reads GIRO's `VehicleTask`
assignment. GREEDY builds feasible routes sequentially and can use too many.
MATCHING builds a graph-derived path cover and independently resource-validates
every route. NO_CHEAT is an artificial-only pricing ablation, not the fairest
Goal-1 benchmark.

## Correctness and search repairs in this checkpoint

- a SciPy/HiGHS restricted-master LP removes Gurobi from column generation;
- the DP heap is explicit: `time`, `reduced_cost`, or
  `reduced_cost_bound`;
- the bound priority uses an optimistic weighted-interval suffix bound on
  positive trip duals;
- label-cap retention follows the selected priority and records every eviction;
- station-to-trip waiting can span the 1,560-minute horizon for flat-price
  rediscovery;
- successor-specific SOC boundary targets repair feasible charges lost by the
  old fixed 30-kWh grid;
- existing trip-incidence patterns are filtered before the K-best cutoff;
- only a strictly cheaper realization of an existing incidence pattern enters
  the current trip-cover-only master;
- negative depot completions are condensed online to the cheapest realization
  per trip set;
- pricing cannot report restricted reduced-cost optimality after a timeout or
  label-cap eviction;
- checkpoints include hashes, algorithm settings, Git state, and a deterministic
  dirty-worktree fingerprint;
- the runner records route weight and artificial use before adding each batch,
  then re-solves and records final LP metrics;
- local and Unicorn launchers default to SciPy, MATCHING, the repaired bound
  heap, full station waiting, and successor targets.

The current test suite passes (`79 passed, 4 subtests passed`) before the
reportable local benchmark. See
`GIRO_COLUMN_AUDIT.md` for exact known-duty evidence and
`LOCAL_BENCHMARK_RUNBOOK.md` for commands and interpretation.

## Structural lower bounds: do not ask for impossible results

Peak simultaneous service `P` is always a fractional route-weight lower bound:
one bus route can cover at most one trip active at the same instant. The hard
first-10 and first-15 instances have `P=10` and `P=15`; a request for LP route
weight below those values is therefore impossible.

The deterministic random suite is useful for rediscovery and scaling, but it
does not currently contain a below-`N` fleet target. A reachability-antichain
audit finds width exactly `N` for all five 10-bus, all five 15-bus, and all five
20-bus samples. For random 15-r04, peak concurrency is only 14, but an explicit
15-trip reachability antichain proves every fractional route cover has weight at
least 15. MATCHING supplies 15 routes, while GREEDY supplies 17. Its valid DP
test is therefore 17 to 15; a result below 15 is a correctness alarm.

Reproduce these certificates with `src/audit_structural_route_bound.py`. Its
JSON output includes the exact antichain trip IDs, graph edge counts,
transitive-closure check, and the proof scope for the configured restricted
pricing graph.

Do not use `LP_Obj / BUS_COST` as the route weight. Charging cost is also in the
objective. Read `LP_Route_Weight_Before_Add` and `Final_LP_Route_Weight`
directly. A restricted-master objective is an upper bound on the full column
master optimum until pricing is genuinely exhaustive; it is not an LP lower
bound merely because it came from an LP solve.

## Heap evidence so far

On the same repaired hard-10 MATCHING master for one second and 5,000 labels per
node:

| Heap | Accepted | Best reduced cost | Longest returned route | Mean trips |
|---|---:|---:|---:|---:|
| `time` | 88 | -100,112.036 | 4 | 2.47 |
| `reduced_cost` | 150 | -100,110.639 | 9 | 5.24 |
| `reduced_cost_bound` | 150 | -100,112.036 | 6 | 4.81 |

None moved that master because it was already at the certified fleet count.
The heap effect is real but dual-dependent; accepted-column count and most
negative reduced cost do not identify the best master progress.

A one-second random 15-r04 GREEDY probe also stayed at 17 routes, but the heaps
behaved very differently: `time` accepted 50 columns, `reduced_cost` 150, and
`reduced_cost_bound` 27; their best reduced costs were approximately -200k,
-300k, and -500k. That conflict is why the next experiment compares five-minute
reoptimized outcomes rather than crowning a heap from one pricing call.

## Next experiment and stop rules

1. Commit the exact code, then run random 15-r04 GREEDY for five matched minutes
   with each of the three heaps, sequentially on the Mac.
2. Rank by final LP route weight, time to the first drop below 17, time to 15,
   and final objective. Use throughput/best reduced cost only as diagnostics.
3. If one heap moves the master, confirm it and give only that configuration a
   30-minute run. Use 3h only after 30m shows continued useful movement.
4. If all heaps remain at 17 while adding many columns, do not buy more hours.
   Add multi-queue/diversified pricing or a duty-guided diagnostic and examine
   why useful long columns are crowded out.
5. Reconstruct and validate a coherent full-day trip set before claiming
   43-bus GIRO parity. The tracked 20/43 files mix weekday variants and are
   synthetic scaling inputs, not verified parity instances.
6. After flat-price Goal 1 is trustworthy, make charging start time a real
   decision. Full station waiting repairs rediscovery, but charging still starts
   immediately on arrival; temporal demand-response savings are not credible
   until delayed charging is modeled.

Preserve every long-running column pool outside Git and back up its whole result
directory. Never use `git clean` to manage generated instances or results.
