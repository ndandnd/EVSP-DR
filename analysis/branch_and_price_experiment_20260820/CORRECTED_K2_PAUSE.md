# Corrected branch-and-price: bounded k2 pause report

Date: 2026-08-20.

This report contains only runs made after B0027 was corrected. Every node used:

1. exact constrained Phase I with zero-cost real routes;
2. infeasibility only from a strictly positive conservative artificial-mass
   dual bound;
3. artificial-free, fleet-only Phase II;
4. conservative Phase-II fleet bounds with integerized fleet pruning.

All runs were local. No cluster jobs were submitted.

## Validation gates

- Hash-bound G1: pass (`k02_s2`, Phase-II route weight `2.187500000`).
- Finite-cost Phase-I regression: pass.
- Strictly positive Phase-I dual-certificate regression: pass.
- G2 child-bound runtime assertion: pass.
- G3 exhaustive constrained pricing: pass, including 20 randomized multi-SOC
  seeds, interacting constraints, and combined/Phase-I/fleet objectives.
- G4 pair-integrality cross-check: pass.
- G5 exact-partition audit: pass.
- Durable checkpoint/resume: pass.
- Final module suite: 15/15 pass.

## Executed primary-grid results

`root fleet LP` and `global LB` are certified bounds for the 15 kWh / 10 minute,
300 kWh / 300 kW discretized route space. `best integer cost` is the stored cost
of a physically replayed fleet-optimality candidate; charging cost was not
separately optimized.

| cell | root fleet LP | best integer fleet | best integer cost | global LB | fleet proven? | nodes | open depth-capped | pricing solves | pricing wall s | wall s | stop |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| k02_s1 | 2.181818182 | 4 | 400099.936 | 2.181818135 | no | 7 | 0 | 1142 | 38.069 | 45.087 | 10x kill |
| k02_s2 | 2.187500000 | **3** | 300148.744 | **3.000000000** | **yes** | 15 | 0 | 433 | 5.186 | 6.855 | tree closed by integer fleet bound |
| k02_s3 | 2.274725275 | 7 | 700085.472 | 2.274725219 | no | 7 | 0 | 2682 | 126.963 | 149.330 | 10x kill |

No node was pruned as Phase-I infeasible in these runs. One explored node began
Phase I with artificial mass 3, priced missing real routes, reached mass 0, and
then entered Phase II; it was not falsely pruned.

## Kill criterion

The adopted criterion fired honestly on two cells:

- `k02_s1`: solved-node pricing slowdowns included 18.49x, 32.96x, and 49.04x
  relative to the root Phase-II pricing call.
- `k02_s3`: slowdowns included 18.50x, 35.15x, and 34.33x.

The experiment is therefore paused. No k3/k5 or corrected fine-grid matrix was
started after the criterion fired.

## Scientific conclusion

For `k02_s2` on the primary grid, the true discretized minimum fleet is
**3 buses**, not 4. The old RAW one-shot pool MIP proved 4 only for its fixed
pool, so this cell is direct evidence that complementary column composition was
the missing ingredient.

`k02_s1` and `k02_s3` remain unresolved by corrected branch-and-price. The
separate arc-flow oracle should decide them independently. No conclusion is
drawn about all three k2 replicates or the 1 kWh / 5 minute grid here.

## Artifact bindings

| cell | code commit | result SHA256 | node-ledger SHA256 |
|---|---|---|---|
| k02_s1 | `0ff69ea89f59df9c0d1ff929e39c41fd6e7472db` | `9ec17c2b1f4933b8e72fb75b394501941b26f0e7393d1786776538b1168a0257` | `649956d2d286aa1bc6598c55db51d100ff8bbb40201116a2d7d39beb77b24a4d` |
| k02_s2 | `af45ed80d6e93f768224c118296dade7fb4b31cb` | `8c7da24a6c0a6121ea013e5aa8e0eae73d2fee412d967085389b907b7bc70aec` | `6980f3fd912ac88273aa1a1b037098eeccef1c632513d3ad0e2f2254e5abade5` |
| k02_s3 | `0ff69ea89f59df9c0d1ff929e39c41fd6e7472db` | `c9dface96bca8f1d880b05c44e4cb80de2ee39e30bd8d4bdff3b639259c1467b` | `4bb5f28986d2cf3556df6c2684b990f1bf9d7460e140c97a8ebfa1323d3dc7ba` |

The root baseline manifest SHA256 recorded in each result binds the exact
instance, flat tariff, reference/deadhead files, grid, physics, and source
analysis CSV.
