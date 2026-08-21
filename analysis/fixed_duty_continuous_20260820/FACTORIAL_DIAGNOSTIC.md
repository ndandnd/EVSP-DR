# Fixed-duty 2×2×2 factorial and prefix-causal bound

Executed locally on 2026-08-20. No cluster jobs were submitted.

## Treatments

The same 15 distinct duties in the three k5 cells were evaluated under:

1. **Start treatment**
   - `zero_uncapped`: zero monetary charge-start cost, no event cap.
   - `zero_observed_event_cap`: zero monetary start cost and hard cap
     `N_d` equal to that duty's recorded GIRO recharge count. Counts come from
     `data/tariff_response/giro40_duty_manifest.csv`; the 15 values (3–16) are
     recorded in `event_caps.csv`.
2. **Terminal SOC**
   - `>= reserve`: depot-arrival SOC at least 0 kWh.
   - `>= initial`: depot-arrival SOC restored to the 240 kWh initial SOC.
3. **Tariff**
   - `peak12`: the existing synthetic single-peak instrument.
   - `two_peak_se3_synthetic`: the new mean-preserving synthetic morning/
     evening instrument documented in `TWO_PEAK_INSTRUMENT.md`.

Both start treatments deliberately set the monetary start cost to zero. The
event-cap arm isolates fragmentation control from a dollar-valued penalty.

All runs otherwise use the frozen 240 kWh battery, 240 kW constant-power
charger, zero reserve, full initial SOC, zone-centroid deadhead, and unlimited
chargers.

## Factorial results

Costs below are charging energy costs; start cost is zero. Each configuration
reports three distinct quantities on one matched feasible duty set:

1. **Uncapped timing value:** uncapped arrival-priority baseline minus
   uncapped optimized charging.
2. **Capped timing value:** capped arrival-priority baseline minus capped
   optimized charging, both subject to the same duty-specific `N_d`.
3. **Cost of the cap:** capped arrival-priority baseline minus uncapped
   arrival-priority baseline.

| tariff | terminal | duties | uncapped baseline → optimum | uncapped timing | capped baseline → optimum | capped timing | cap cost |
|---|---|---:|---:|---:|---:|---:|---:|
| peak12 | >= reserve | 15 | 516.586 → 508.791 | 7.795 (**1.509%**) | 530.367 → 523.940 | 6.428 (**1.212%**) | 13.782 |
| peak12 | >= initial | 12 | 537.536 → 497.125 | 40.412 (**7.518%**) | 546.458 → 506.046 | 40.412 (**7.395%**) | 8.922 |
| two-peak | >= reserve | 15 | 522.715 → 494.866 | 27.849 (**5.328%**) | 529.363 → 500.529 | 28.834 (**5.447%**) | 6.648 |
| two-peak | >= initial | 12 | 686.999 → 505.308 | 181.691 (**26.447%**) | 705.481 → 511.992 | 193.489 (**27.427%**) | 18.483 |

The capped arrival-priority baseline is solved directly by the same MILP with
`timing_mode="arrival"` and `max_charge_events=N_d`; it is not formed by
deleting events. Every included capped baseline and optimum is energy-, time-,
SOC-, trip-, and event-cap-feasible and passes the same exact replay.

Every feasible optimized and charge-on-arrival schedule replayed exactly.

The `>= initial` policy is physically infeasible for duties 13302, 13410, and
13304 under 240 kWh/240 kW and the restricted graph, regardless of tariff or
event cap. Initial-policy percentages therefore describe the same 12-duty
subset and must not be presented as 15-duty results.

## Prefix-causal bound

The earlier global-cheapest-price bound was loose because it allowed energy
needed early in a duty to be bought in a late terminal window. The
prefix-causal relaxation instead:

- uses service energy plus minimum deadhead energy at each duty prefix;
- creates continuous charging variables in every time-feasible
  station/tariff segment, each bounded by segment duration × 240 kW;
- requires cumulative purchased energy to cover every trip prefix and the
  named terminal target;
- relaxes station-option exclusivity, battery upper capacity, and charge-event
  sequencing.

It is therefore still a valid lower bound on optimized charging cost, but it
respects energy causality and charging power by time segment. One aggregate
bound is reported per tariff/terminal configuration on the matched set.

| tariff | terminal | duties | prefix lower cost | timing upper value | timing upper % | uncapped achieved / bound |
|---|---|---:|---:|---:|---:|---:|
| peak12 | >= reserve | 15 | 484.046 | 32.539 | **6.299%** | 24.0% |
| peak12 | >= initial | 12 | 460.101 | 77.435 | **14.406%** | 52.2% |
| two-peak | >= reserve | 15 | 482.490 | 40.225 | **7.695%** | **69.2%** |
| two-peak | >= initial | 12 | 475.353 | 211.646 | **30.807%** | 85.8% |

The prefix relaxation itself cannot restore duty 13302 to initial SOC, even
while allowing all alternative station windows simultaneously. Duties 13410
and 13304 pass the relaxation but fail the exact model because of the
constraints the bound intentionally drops.

## Interpretation

The tariff shape remains the largest lever. With the unresolved but minimally
committing `>= reserve` policy, replacing `peak12` by the realistic-shaped
two-peak instrument raises uncapped timing-only savings from 1.509% to
**5.328%** and capped timing-only savings from 1.212% to **5.447%**. The cap
itself raises the two-peak arrival-priority baseline by 6.648 model-currency
units (1.272% of the uncapped baseline). Timing value and cap cost are separate
effects and must not be collapsed into one fixed-denominator percentage.

Restoring initial SOC raises the measured response dramatically, to 25–26%
under the two-peak instrument, because it adds a large, deferrable terminal
energy purchase. It also makes 3 of 15 duties infeasible. This treatment is a
terminal-accounting sensitivity, not a defensible headline until the terminal
policy is frozen.

Under two-peak/`>= reserve`, the exact model captures much of the
prefix-causal opportunity: uncapped timing captures
`27.849 / 40.225 = 69.2%` of the single aggregate 7.695% bound.
Under `peak12`, considerably more of the causal opportunity remains blocked by
exact SOC, station-choice, and event constraints.

## Synthetic two-peak provenance

`two_peak_se3_synthetic_h26.csv` SHA-256:
`21ac43bdd006582ea4a8acacf5168897b3495e28807a7bd247c5079693c79a59`.

The first 24 hours have mean `0.0992`, matching the flat instrument. The curve
has an overnight trough, broad morning peak, and stronger evening peak;
hours 24–26 repeat hours 0–2. It is inspired by the intraday shape of Nordic
SE3 day-ahead prices but contains no observed series values. It is a
**synthetic instrument pending a frozen real series**, not a research result.
Full construction and source provenance are in `TWO_PEAK_INSTRUMENT.md`.

Post-change full repository test result:
`513 passed, 123 subtests passed in 159.20s`.

Real charging tapers above about 80% SOC, so constant power to full is mildly
optimistic. Charger capacity remains unlimited; these results make no
peak-shaving claim.
