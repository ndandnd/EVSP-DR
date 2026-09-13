# Cumulative-budget controls: 13 September, 14:44 EDT

Nine fresh CG runs have pricing certificates. Seven fresh MIPs have finished: all six k=5 cases and chain6 k=8. Six match their GIRO targets. Chain5 k=5 instead proves six buses within its fresh pool, while the matched warm pool proves five.

| Case | CG allowance | Fresh CG convergence | Fresh integer fleet | Warm integer fleet |
|---|---:|---:|---|---|
| Chain 1, k=5 | 203.2 min | 20.6 min | 5; proved in pool | 5; proved in pool |
| Chain 2, k=5 | 34.1 min | 11.6 min | 5; proved in pool | 5; proved in pool |
| Chain 3, k=5 | 51.0 min | 5.1 min | 5; proved in pool | 5; proved in pool |
| Chain 4, k=5 | 20.2 min | 11.3 min | 5; proved in pool | 5; proved in pool |
| Chain 5, k=5 | 110.0 min | 8.0 min | 6; proved in pool | 5; proved in pool |
| Chain 6, k=5 | 15.0 min | 6.1 min | 5; proved in pool | 5; proved in pool |
| Chain 6, k=8 | 104.6 min | 12.8 min | 8; proved in pool | 8; proved in pool |

## The precise limitation in chain5, k=5

Fresh CG stops after481.16s (8.02min), well within its6601s (110.02min) accumulated allowance. It certifies the tested event graph at reduced-cost tolerance0.0001: final minimum reduced cost−6.44e−10, artificial coverage0, maximum row violation1.30e−14. Weighted LP objective500276.192 and total fractional route weight5.000000000000015 are different quantities; route weight is not asserted to be a fleet-only lower bound.

The fresh pool contains14,175 columns. Gurobi stage1 proves its integer fleet optimum6 in36.23s, with incumbent6 and bound6. The matched warm pool contains16,870 columns and proves5 in3.35s. Inputs, model settings and current MIP code are matched; individual-route replay and at-least-once trip coverage pass for both. Duplicate-trip removal is not validated for this pair, and this baseline has no shared station capacity or terminal-SOC floor.

Thus the fresh column pool lacks a five-bus integer cover, although another validated pool supplies one. This is a concrete pool-composition limitation, not an unresolved MIP time limit or unfinished negative-reduced-cost search in the tested graph. CG’s LP-optimality stopping condition does not require every route useful to an integer combination to have been generated. Longer MIP on this unchanged pool cannot obtain five buses. Further useful routes may have zero or positive reduced cost at the final dual; their actual reduced costs have not been measured here. This result alone does not establish whether each specific warm route is present in the fresh graph, or isolate all historical code/hardware effects.

The larger-budget continuation shares a certified endpoint by design; it does not invent additional columns after convergence. A future test of integer-oriented column generation, column transfer or branch-and-price would be a new treatment, not a relaunch of this completed control.

All nine CG certificates are restricted to the tested graph/tolerance. All17 completed MIP artifacts (7fresh,10warm) have finite-pool fleet proofs and individual-route physical replay. No new execution error, confirmed preemption or unsatisfiable dependency was observed. Snapshot timestamps describe a collection interval, not one instantaneous scheduler state.

Source: [comparison.csv](comparison.csv), [collection.json](collection.json), full collector snapshot `outputs/post_meeting_20260910/monitor/20260913T184244Z.json`. Historical code and hardware varied; timings remain a retrospective accumulated-budget comparison.
