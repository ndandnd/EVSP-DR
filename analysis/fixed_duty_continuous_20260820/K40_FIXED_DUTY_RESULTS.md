# All-40 GIRO fixed-duty charging results

Executed locally on 2026-08-21. No cluster jobs were submitted.

## Scope

This analysis uses the complete 40-duty GIRO partition in
`data/tariff_response/frozen_instances/Practice_Custom_DutyUnion_k40_r2.csv`
(947 trips; SHA-256
`3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd`).
Trip sequences are fixed to the industrial schedule; no route-generation pool
or routing-tier result is used.

Each tariff/terminal configuration solves four arms independently:

1. uncapped charge-on-arrival baseline;
2. uncapped optimized timing;
3. capped charge-on-arrival baseline;
4. capped optimized timing.

The cap `N_d` is each duty's recorded GIRO recharge count. All four arms use
zero monetary start cost. A duty enters an aggregate only when all four arms
are feasible. Every duty remains in `k40_matched_duty_results.csv`, including
explicit infeasibility reasons.

## Matched aggregate quantities

| tariff | terminal | matched / 40 | uncapped baseline → optimum | uncapped timing | capped baseline → optimum | capped timing | cap cost |
|---|---|---:|---:|---:|---:|---:|---:|
| peak12 | >= reserve | 39 | 1204.168 → 1181.631 | 22.537 (**1.872%**) | 1242.367 → 1221.468 | 20.899 (**1.682%**) | 38.198 |
| peak12 | >= initial | 30 | 1248.602 → 1123.365 | 125.237 (**10.030%**) | 1295.980 → 1169.047 | 126.933 (**9.794%**) | 47.378 |
| two-peak | >= reserve | 39 | 1219.040 → 1155.491 | 63.549 (**5.213%**) | 1237.773 → 1173.955 | 63.819 (**5.156%**) | 18.733 |
| two-peak | >= initial | 30 | 1633.975 → 1134.543 | 499.432 (**30.565%**) | 1680.874 → 1151.622 | 529.252 (**31.487%**) | 46.899 |

These are three separate quantities: timing value within the uncapped model,
timing value within the capped model, and the cost imposed on the
arrival-priority baseline by the cap.

## Feasibility counts

| terminal | uncapped infeasible | infeasible under cap | matched |
|---|---:|---:|---:|
| >= reserve | 0 / 40 | 1 / 40 (13303) | 39 |
| >= initial | 7 / 40 | 10 / 40 | 30 |

The `>= initial` uncapped failures are 13302, 13304, 13312, 13319, 13325,
13409, and 13410. The capped arm additionally excludes 13303, 13321, and
13405.

Thus 17.5% of all duties cannot restore initial SOC even without an event cap,
close to the earlier 3/15 = 20% finding. That consistency is evidence that a
single-day horizon is too short for this terminal policy on a material share
of the real schedule; it is not evidence that restoring energy is inherently
the wrong policy. The capped 25% rate combines that horizon effect with the
observed event-count restriction.

Every included baseline and optimum is generated directly under its named
constraints and passes exact energy, time, SOC, trip, and cost replay. No
charging event is deleted post hoc.

## Load and charger-concurrency measurements

`charge_events` and `charge_starts` are equal here: each emitted positive,
contiguous event has exactly one physical start. `peak_kW` is the maximum
simultaneous system load across the matched duties. The final column is the
maximum simultaneous buses at one modeled station in one exact time interval.

| tariff | terminal | arm | duties | starts | kWh | global concurrency | peak kW | max same-station concurrency |
|---|---|---|---:|---:|---:|---:|---:|---:|
| peak12 | >= reserve | uncapped arrival | 39 | 371 | 12159.7 | 8 | 1920 | 2 |
| peak12 | >= reserve | uncapped optimized | 39 | 371 | 12159.7 | 13 | 3120 | 4 |
| peak12 | >= reserve | capped arrival | 39 | 314 | 12141.6 | 7 | 1680 | 3 |
| peak12 | >= reserve | capped optimized | 39 | 314 | 12144.8 | 12 | 2880 | 4 |
| peak12 | >= initial | uncapped arrival | 30 | 270 | 15636.8 | 12 | 2880 | 8 |
| peak12 | >= initial | uncapped optimized | 30 | 268 | 15636.8 | 18 | 4320 | 16 |
| peak12 | >= initial | capped arrival | 30 | 224 | 15634.0 | 11 | 2640 | 8 |
| peak12 | >= initial | capped optimized | 30 | 224 | 15634.0 | 20 | 4800 | 20 |
| two-peak | >= reserve | uncapped arrival | 39 | 331 | 12142.6 | 9 | 2160 | 3 |
| two-peak | >= reserve | uncapped optimized | 39 | 326 | 12154.5 | 24 | 5760 | 7 |
| two-peak | >= reserve | capped arrival | 39 | 282 | 12133.7 | 9 | 2160 | 3 |
| two-peak | >= reserve | capped optimized | 39 | 287 | 12148.4 | 22 | 5280 | 7 |
| two-peak | >= initial | uncapped arrival | 30 | 275 | 15628.4 | 9 | 2160 | 7 |
| two-peak | >= initial | uncapped optimized | 30 | 233 | 15632.4 | 30 | 7200 | 30 |
| two-peak | >= initial | capped arrival | 30 | 213 | 15625.4 | 8 | 1920 | 7 |
| two-peak | >= initial | capped optimized | 30 | 202 | 15632.4 | 30 | 7200 | 30 |

The complete station/time sweep is in
`k40_station_time_concurrency.csv` (5,343 positive-load intervals). Under the
minimally committing `>= reserve` policy, the two-peak optimized schedules
require up to 24 simultaneous buses system-wide (5.76 MW) and 7 at one station
(1.68 MW). Under `>= initial`, both two-peak optimized arms concentrate 30
buses at `PARX_1` simultaneously (7.2 MW).

These values measure how demanding the unlimited-charger solution is. They do
not establish peak-shaving value. Any charger limit below the reported
station/time concurrency changes the feasible set and requires reoptimization.

## Interpretation

For the complete industrial schedule under `>= reserve`, the synthetic
two-peak instrument produces a fixed-sequence timing value of **5.213%**
uncapped and **5.156%** under observed GIRO event caps. This is close to the
k5-only sensitivity but now covers the entire 40-duty, 947-trip schedule and
does not depend on unreliable large-scale routing pools.

The full-restoration sensitivity is much larger (30–31%) but applies to only
30 matched duties and drives severe depot concurrency. It remains a
terminal-policy and horizon sensitivity, not the primary result.

Both tariffs are synthetic instruments. A real-series Goal-2 claim requires a
frozen, hashed tariff with a documented selection rule. Real charging also
tapers above about 80% SOC, so constant power to full is mildly optimistic.
Chargers remain unlimited in this experiment.

Post-artifact full repository test result:
`513 passed, 123 subtests passed in 161.16s`.

## Machine-readable artifacts

- `k40_matched_duty_results.csv`: 160 per-duty/configuration rows, including
  all infeasible arms and per-arm replay/load/event fields.
- `k40_matched_aggregate.csv`: the three matched quantities and explicit
  infeasibility counts for four tariff/terminal configurations.
- `k40_solution_metrics.csv`: per-arm events, starts, energy, system peak, and
  station maxima.
- `k40_station_time_concurrency.csv`: exact positive-load intervals by station
  and arm.
- `k40_sweep_metadata.json`: input hashes, physics, matched-set rule, and
  producing commit.
