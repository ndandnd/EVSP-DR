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
make all 10/10 and 15/15 known duties representable. On synthetic random
15-r04, no individual five-minute search moved the 17-route seed's LP route
weight. Combining complementary columns from eight saved search variants did move it
to 16.857142857, while an independent model-derived MATCHING cover proves 15 is
attainable. Pricing has therefore shown its first controlled fleet-weight
progress, but it remains far from a reliable or exhaustive oracle.

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
- the DP heap is explicit: `time`, `reduced_cost`, `reduced_cost_bound`, or the
  optional first-trip round-robin `start_fair_bound`;
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
- optional diversified output mixes best-reduced-cost, longest, and rare-trip
  columns, and optional incidence-diverse dominance preserves equal-cost
  alternative histories;
- a zero-charge station pass-through preserves labels whose higher SOC makes a
  positive charge impossible, and restricted-wait dominance no longer assumes
  that higher SOC always has the same temporal charging options;
- pricing cannot report restricted reduced-cost optimality after a timeout or
  label-cap eviction;
- checkpoints include hashes, algorithm settings, Git state, and a deterministic
  dirty-worktree fingerprint;
- the runner records route weight and artificial use before adding each batch,
  then re-solves and records final LP metrics;
- local and Unicorn launchers default to SciPy, MATCHING, the repaired bound
  heap, full station waiting, and successor targets.

The complete checkpoint test suite passes (`113 passed, 6 subtests passed`). See
`GIRO_COLUMN_AUDIT.md` for exact known-duty evidence and
`GOAL1_LOCAL_RESULTS_20260802.md` for the matched local experiments and
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

## Five-minute pricing evidence

Every individual random 15-r04 run accepted 750 DP columns and stayed at route
weight 17. Trip coverage ranged from 61/337 for `time` to 319/337 for
`start_fair_bound` with ordinary resource dominance. The incidence-diverse fair
run reduced path/energy cost by 23.259 units but did not change route weight.
All calls timed out, none was exhaustive, and none had a label-cap eviction.

The important result comes from preserving complementary pools. The union of
all eight five-minute runs contains 5,669 cheapest unique trip incidences and
has LP route weight 16.857142857 with no artificial trips. No individual pool
or pair improves route weight. The smallest improving combination is the
bound/diversified pool plus both fair-queue dominance variants; it reaches
16.888888889. See `GOAL1_LOCAL_RESULTS_20260802.md` for the complete table and
the exact known-duty, matching-cover, and dual-centering controls.

## Next experiment and stop rules

1. Reproduce the saved-pool union and model-only matching-cover wave audits.
2. Implement a bounded pricing portfolio that combines bound/resource,
   fair/resource, and fair/incidence-diverse candidates before reoptimization.
3. Give that portfolio one five-minute random 15-r04 gate. Rank by final route
   weight and objective; use coverage, depth, and reduced cost as diagnostics.
4. Do not run a single queue for 30 minutes, 3 hours, or 12 hours. Allow a
   30-minute portfolio confirmation only after the five-minute gate shows
   material or continuing route-weight progress.
5. Keep MATCHING as the operational non-cheating initializer. Its model-derived
   15-route cover closes this instance's fleet-count question; GREEDY remains a
   deliberately weak pricing-discovery control.
6. Reconstruct and validate a coherent full-day trip set before claiming
   43-bus GIRO parity. The tracked 20/43 files mix weekday variants and are
   synthetic scaling inputs, not verified parity instances.
7. After flat-price Goal 1 is trustworthy, make charging start time a real
   decision. Full station waiting repairs rediscovery, but charging still starts
   immediately on arrival; temporal demand-response savings are not credible
   until delayed charging is modeled.

Preserve every long-running column pool outside Git and back up its whole result
directory. Never use `git clean` to manage generated instances or results.
