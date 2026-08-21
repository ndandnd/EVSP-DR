# Tiny SOC-time differential and arbitrary-dual pricing oracle

Date: 2026-08-21. All execution was local.

## Campaign

- Seed: **20260821**
- Generated networks: **240**
- Trips per network: 8–14
- Dual vectors per network: **32**
- Total arbitrary-dual pricing comparisons: **7,680**

The exhaustive oracle propagates every reachable `(SOC-time node, trip
incidence)` state and retains the cheapest explicit path for that state.
Discarding a costlier prefix only when it reaches the identical node with the
identical trip incidence is exact: every continuation and every dual
contribution is then identical. The resulting route costs are independent of
the DP pricers being tested.

The dual suite contains exact zero-dual ties, signed perturbations of
`1e-8`, all-negative vectors, wide signed vectors, CG-scale vectors with
negative components, and quantized tie-heavy vectors. It is not restricted to
duals visited by column generation.

## Arbitrary-dual pricing result

| pricer | exhaustive agreements | rate | maximum absolute error |
|---|---:|---:|---:|
| exact-CG unconstrained DAG DP | 7,680 / 7,680 | 100% | 3.50e-10 |
| corrected constrained DAG DP, empty branch set | 7,680 / 7,680 | 100% | 3.50e-10 |

There were **zero pricing disagreements**. Of the 7,680 vectors, 7,403
contained at least one negative component and 1,246 produced multiple
exhaustive minimizers within `1e-7`.

Pricing-domain coverage:

| dimension | generated networks |
|---|---:|
| SOC step 0.5 | 79 |
| SOC step 1.0 | 87 |
| SOC step 2.0 | 74 |
| one station | 139 |
| two stations | 101 |
| immediate-only charging entry | 118 |
| delayed charging entry available | 122 |
| flat tariff | 120 |
| time-varying tariff | 120 |

Station reachability density ranged from 0.15 to 1.00.

## Four-way agreement

| comparison with exhaustive oracle | agreements | rate |
|---|---:|---:|
| exact-CG LP bound | 240 / 240 | 100% |
| branch-and-price LP bound | 240 / 240 | 100% |
| arc-flow LP bound | 240 / 240 | 100% |
| branch-and-price integer fleet | 240 / 240 | 100% |
| arc-flow integer fleet | 240 / 240 | 100% |
| exact-CG final-pool integer fleet | 216 / 240 | 90% |
| all four integer fleets | 216 / 240 | 90% |

Every one of the 24 four-way disagreements is an exact-CG final-pool
composition failure. Exhaustive enumeration, branch-and-price, and arc-flow
agree; exact CG certifies the same LP but its final pool needs one extra bus.
The original outcomes are:

- 19 cases: true fleet 2, pool fleet 3;
- 3 cases: true fleet 3, pool fleet 4;
- 1 case: true fleet 4, pool fleet 5;
- 1 case: true fleet 5, pool fleet 6.

Each disagreement file contains its original case and a greedy
trip/station-irreducible reproducer. Minimal trip counts are:

| trips | reproducers |
|---:|---:|
| 5 | 1 |
| 7 | 4 |
| 8 | 5 |
| 9 | 4 |
| 10 | 3 |
| 11 | 2 |
| 12 | 3 |
| 13 | 1 |
| 14 | 1 |

The smallest reproducer is
`disagreement_random_0175.json`: 5 trips, 2 stations, 20 exhaustive routes,
and 14 CG-pool columns. The true/B&P/arc-flow fleet is 2; the pool fleet is 3;
all LP bounds are 2.

This is not an LP-pricing error. Negative-reduced-cost exhaustion proves the
LP bound, but it does not guarantee that the final pool contains complementary
columns forming an optimal integer partition.

## Mutation result

All methods agreed before and after each targeted mutation:

| mutation | baseline fleet | mutated fleet | binding? |
|---|---:|---:|---|
| permit a 58-minute gap where the cap is 57 | 2 | 1 | yes |
| permit SOC one step below reserve | 2 | 1 | yes |
| permit a station-to-station arc | 2 | 1 | yes |

## Artifacts

- `agreement.csv`: one row per generated network, including domain dimensions,
  pricing sample count, and maximum pricing error.
- `summary.json`: aggregate agreement, coverage, mutations, and every
  disagreement path.
- `disagreement_random_*.json`: original and irreducible reproducing instance
  for every four-way disagreement.

Producing code commit:
`ebb8f5e3900c5b18a0f30f496908f3a775e87867`.

- `summary.json` SHA256:
  `3151cb6c4804e387dbad292c2389829ced059856e7f18a093124897fb3aedf6a`
- `agreement.csv` SHA256:
  `afc3f3f583c2d485db159705ddc233c443011138b5cd9912cfb3e83d77231c53`

No cluster jobs were submitted.
