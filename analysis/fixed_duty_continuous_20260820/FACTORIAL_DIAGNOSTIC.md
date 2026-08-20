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

Costs below are charging energy costs; start cost is zero. Within each
tariff/terminal cell, **both treatments use the uncapped charge-on-arrival
cost as their fixed denominator**. `Treatment arrival` is retained only to
show why treatment-specific denominators are not comparable.

| tariff | terminal | start treatment | duties | fixed denominator | treatment arrival | optimized | fixed-denominator saving % |
|---|---|---|---:|---:|---:|---:|---:|
| peak12 | >= reserve | uncapped | 15 | 516.586 | 516.586 | 508.791 | **1.509%** |
| peak12 | >= reserve | observed cap | 15 | 516.586 | 530.367 | 523.940 | **-1.424%** |
| peak12 | >= initial | uncapped | 12 | 537.536 | 537.536 | 497.125 | **7.518%** |
| peak12 | >= initial | observed cap | 12 | 537.536 | 546.458 | 506.046 | **5.858%** |
| two-peak | >= reserve | uncapped | 15 | 522.715 | 522.715 | 494.866 | **5.328%** |
| two-peak | >= reserve | observed cap | 15 | 522.715 | 529.363 | 500.529 | **4.244%** |
| two-peak | >= initial | uncapped | 12 | 686.999 | 686.999 | 505.308 | **26.447%** |
| two-peak | >= initial | observed cap | 12 | 686.999 | 705.481 | 511.992 | **25.474%** |

The capped optimum costs at least the uncapped optimum in all four
tariff/terminal cells and for every individually comparable duty (zero
violations at absolute tolerance `1e-6`). Consequently capped savings are no
greater than uncapped savings everywhere, as required by nesting.

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
respects energy causality and charging power by time segment. The same bound is
valid for both start treatments because it relaxes the event cap.

| tariff | terminal | start treatment | duties | actual saving | prefix upper bound |
|---|---|---|---:|---:|---:|
| peak12 | >= reserve | uncapped | 15 | 1.509% | **6.299%** |
| peak12 | >= reserve | observed cap | 15 | -1.424% | **6.299%** |
| peak12 | >= initial | uncapped | 12 | 7.518% | **14.406%** |
| peak12 | >= initial | observed cap | 12 | 5.858% | **14.406%** |
| two-peak | >= reserve | uncapped | 15 | 5.328% | **7.695%** |
| two-peak | >= reserve | observed cap | 15 | 4.244% | **7.695%** |
| two-peak | >= initial | uncapped | 12 | 26.447% | **30.807%** |
| two-peak | >= initial | observed cap | 12 | 25.474% | **30.807%** |

The prefix relaxation itself cannot restore duty 13302 to initial SOC, even
while allowing all alternative station windows simultaneously. Duties 13410
and 13304 pass the relaxation but fail the exact model because of the
constraints the bound intentionally drops.

## Interpretation

The tariff shape remains the largest lever. With the unresolved but minimally
committing `>= reserve` policy, replacing `peak12` by the realistic-shaped
two-peak instrument raises uncapped timing-only savings from 1.509% to
**5.328%**. Under the observed event cap, the fixed-denominator result rises
from -1.424% to **4.244%**. The previously reported capped 5.447% used the
capped arrival cost as its own denominator and is not comparable or quotable.
The corrected result shows that fragmentation control costs about 1.08
percentage points under the two-peak instrument.

Restoring initial SOC raises the measured response dramatically, to 25–26%
under the two-peak instrument, because it adds a large, deferrable terminal
energy purchase. It also makes 3 of 15 duties infeasible. This treatment is a
terminal-accounting sensitivity, not a defensible headline until the terminal
policy is frozen.

Under two-peak/`>= reserve`, the exact model captures much of the
prefix-causal opportunity (4.244–5.328% achieved versus a common 7.695% upper
bound).
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
