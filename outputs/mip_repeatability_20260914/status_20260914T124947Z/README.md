# Repeating the original MIP allowance

27 completed repetitions on 9 selected inherited pools. 21 match the target fleet. These are repeated computations, not independent sampled datasets.

| Case | Target | Original buses | Repeat 1 | Repeat 2 | Repeat 3 | Repeat fleet-stage minutes | Earlier longer-search buses |
|---|---:|---:|---:|---:|---:|---:|---:|
| C2 k17 | 17 | 18 | 17 | 17 | 17 | 22.7–24.2 | 17 |
| C2 k18 | 18 | 19 | 18 | 18 | 18 | 8.6–9.2 | 18 |
| C2 k19 | 19 | 20 | 19 | 19 | 19 | 14.9–15.0 | 19 |
| C2 k20 | 20 | 21 | 20 | 20 | 20 | 17.8–20.1 | 20 |
| C3 k19 | 19 | 20 | 20 | 20 | 20 | 30.1–30.1 | 19 |
| C3 k20 | 20 | 21 | 20 | 20 | 20 | 20.4–21.5 | 20 |
| C3 k21 | 21 | 22 | 22 | 22 | 22 | 30.1–30.1 | 21 |
| C5 k19 | 19 | 20 | 19 | 19 | 19 | 16.6–18.0 | 19 |
| C6 k18 | 18 | 19 | 18 | 18 | 18 | 13.9–14.4 | 18 |

## Additional unresolved targets

| Case | Target | Original buses | Longer-search buses | Fleet bound | Fleet proved in saved pool | Fleet-stage minutes |
|---|---:|---:|---:|---:|---|---:|
| C1 k19 | 19 | 20 | 19 | 19 | Yes | 22.3 |
| C3 k22 | 22 | 23 | 22 | 22 | Yes | 16.1 |
| C3 k23 | 23 | 24 | 23 | 23 | Yes | 24.1 |

These searches use the original saved columns and up to three hours for fleet minimization, within a three-and-a-half-hour total budget. Fleet proof and charging-cost optimality are separate.

Each repeat has a one-hour total solver allowance, with up to 30 minutes for fleet minimization. The second stage minimizes charging costs with fleet no greater than the first-stage incumbent. Small runtime overruns while the solver stops are recorded rather than rounded into an exact deadline.

All three repeats agree on the fleet within each completed case. Seven selected pools reach their targets in all three repeats. C3 k19 and C3 k21 retain one extra bus in every repeat, with open bounds at the target. Earlier longer searches recovered both targets. The same pools therefore contain the needed routes; these misses are incomplete integer searches. The seven recoveries at the original allowance show that extra allocated time alone is not the cause of the original-versus-rerun difference. Hardware and parallel-search timing are not isolated. These observations do not estimate a population success probability.

Ordered pool hashes, physical input hashes, source CG hashes, execution commit, and initializer kind/count/cost agree with the registered comparison. The initializer summary check is not a hash of the selected initializer route indices. Repeats keep Gurobi 12.0.3, eight threads and default seed zero. All selected routes pass individual replay. Fleet proofs concern the saved pools; charging optimality, duplicate-trip removal and shared-capacity validation are separate. The baseline omits shared station capacity and a terminal-SOC floor.

[Exact results, hosts, timings, node counts and source hashes](results.csv). [Validation and remaining attempt states](validation.json).
