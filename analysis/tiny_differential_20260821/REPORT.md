# Tiny SOC-time four-way differential oracle

Date: 2026-08-21. All runs were local.

## Design

Seed: **20260821**.

The campaign generated 240 deterministic instances with:

- 8–14 trips;
- one or two charging stations;
- random 2–20 minute layover slack;
- trip energy 1–3 SOC steps;
- battery capacity 6–9 steps and reserve 0–2 steps;
- randomized deadhead energy and trip/station reachability.

The oracle explicitly traverses the acyclic SOC×time network and enumerates
every reachable trip-incidence route. A bitmask dynamic program then enumerates
all exact partitions and returns the minimum fleet. The full route set is also
used for the ground-truth LP.

Each case is compared against:

1. exact CG followed by an integer solve over its generated final pool;
2. corrected Phase-I/Phase-II Ryan–Foster branch-and-price;
3. the direct all-arc-integer arc-flow formulation.

## Agreement table

| comparison with exhaustive oracle | agreements | rate |
|---|---:|---:|
| exact-CG LP bound | 240 / 240 | 100% |
| branch-and-price LP bound | 240 / 240 | 100% |
| arc-flow LP bound | 240 / 240 | 100% |
| branch-and-price integer fleet | 240 / 240 | 100% |
| arc-flow integer fleet | 240 / 240 | 100% |
| exact-CG final-pool integer fleet | 206 / 240 | 85.83% |
| all four integer fleets | 206 / 240 | 85.83% |

There were **zero LP disagreements** and zero brute-force/B&P/arc-flow integer
disagreements.

## The 34 disagreements

Every disagreement has exactly one source: the exact-CG final pool lacks a
complementary integer partition even though CG has certified the correct LP.

- 33 original cases overestimated the fleet by one bus.
- 1 original case overestimated by two buses.
- After shrinking, every reproducer is a one-bus pool excess.
- No disagreement was averaged away; all 34 JSON reproducers are retained.

Greedy deletion produced trip- and station-irreducible reproducers:

| minimal trip count | reproducers |
|---:|---:|
| 7 | 6 |
| 8 | 7 |
| 9 | 8 |
| 10 | 9 |
| 11 | 4 |

Twenty-two minimal reproducers need one station; twelve need two.

The smallest example is
`disagreement_random_0215.json`: 7 trips, 1 station, 56 exhaustive routes,
34 CG-pool columns. The exhaustive, B&P, and arc-flow fleet is 2; the
CG-pool integer fleet is 3. All four LP bounds are exactly 2.

This is a pipeline finding, not an exact-pricing LP bug: reduced-cost
certification proves LP completeness, not integer-pool completeness.

## Mutation results

Each mutation used a targeted two-trip instance and changed the optimum in all
four methods:

| mutation | baseline fleet | mutated fleet | binding? |
|---|---:|---:|---|
| permit 58-minute trip gap instead of cap 57 | 2 | 1 | yes |
| permit SOC one step below reserve | 2 | 1 | yes |
| permit station-to-station transfer arc | 2 | 1 | yes |

Thus all three constraints are genuinely binding on at least one explicit
instance. Their mutation tests would detect an implementation that silently
stopped enforcing them.

## Agreement by generated trip count

| trips | cases | disagreements |
|---:|---:|---:|
| 8 | 38 | 0 |
| 9 | 30 | 1 |
| 10 | 29 | 2 |
| 11 | 42 | 4 |
| 12 | 41 | 9 |
| 13 | 33 | 9 |
| 14 | 27 | 9 |

The pool-composition failure becomes more common as the tiny route system grows.

## Artifacts

- `agreement.csv`: one row per generated case.
- `summary.json`: seed, aggregate counts, mutation outcomes, and every
  disagreement path.
- `disagreement_random_*.json`: original instance, original result, minimized
  instance, minimized result, and the irreducibility scope.

Producing commit:
`1c6258b806970b84d905dbad35ac5d7eca019ac1`.

- `summary.json` SHA256:
  `a8be712d2b30327c62419f4f5b446e1d7bd2792432720257f74561314840aabd`
- `agreement.csv` SHA256:
  `773787db54bdcca64b912262b9e3df682322a2cb2ec3e30d2bafa30f32d43acd`

No cluster jobs were submitted.
