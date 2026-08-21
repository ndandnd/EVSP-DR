# All-40 GIRO fixed-duty charging experiment

Executed locally on 2026-08-21. No cluster jobs were submitted.

## Scope and fixed comparison population

The input is the complete 40-duty GIRO partition in
`data/tariff_response/frozen_instances/Practice_Custom_DutyUnion_k40_r2.csv`
(947 trips; SHA-256
`3508a11f73d1186ae87588656d65ea62812c6e222623ae85488eff26cafb35fd`).
Every industrial duty sequence is evaluated; no route-generation pool or
routing-tier result is used.

Per-duty output contains all 40 duties under three tariffs, two terminal
policies, and four arms (240 rows). To make every percentage comparable across
tariffs and terminal policies, aggregate rows use one fixed 30-duty
intersection: duties feasible under every tariff, both terminal policies, and
all four arms.

The four arms are solved independently:

1. uncapped charge-on-arrival baseline;
2. uncapped optimized timing;
3. capped charge-on-arrival baseline;
4. capped optimized timing.

The duty-specific cap `N_d` is the recorded GIRO recharge count. Both capped
arms are solved directly under the cap; no event is deleted post hoc. All arms
use zero monetary start cost.

## Flat control and synthetic-tariff results

| tariff | terminal | fixed duties | uncapped baseline → optimum | uncapped timing | capped baseline → optimum | capped timing | cap cost |
|---|---|---:|---:|---:|---:|---:|---:|
| **flat control** | >= reserve | 30 | 838.434 → 838.434 | 0.000 (**0.000%**) | 838.434 → 838.434 | 0.000 (**0.000%**) | 0.000 |
| **flat control** | >= initial | 30 | 1549.746 → 1549.746 | 0.000 (**0.000%**) | 1549.746 → 1549.746 | 0.000 (**0.000%**) | 0.000 |
| peak12 | >= reserve | 30 | 888.969 → 869.343 | 19.626 (**2.208%**) | 914.071 → 895.651 | 18.421 (**2.015%**) | 25.102 |
| peak12 | >= initial | 30 | 1248.602 → 1123.365 | 125.237 (**10.030%**) | 1295.980 → 1169.047 | 126.933 (**9.794%**) | 47.378 |
| two-peak | >= reserve | 30 | 841.034 → 801.335 | 39.699 (**4.720%**) | 852.666 → 812.123 | 40.542 (**4.755%**) | 11.632 |
| two-peak | >= initial | 30 | 1633.975 → 1134.543 | 499.432 (**30.565%**) | 1680.874 → 1151.622 | 529.252 (**31.487%**) | 46.899 |

The flat rows are exact to emitted precision: optimized and arrival-priority
costs agree, and cap cost is zero. This control shows that the machinery does
not manufacture timing savings when hourly prices carry no timing signal.

## Separately reported feasibility

Feasibility is tariff-invariant. These rows are not silently removed from
`k40_matched_duty_results.csv`.

| scope | terminal | treatment | infeasible | duties |
|---|---|---|---:|---|
| all 40 | >= reserve | uncapped | 0/40 (0%) | — |
| all 40 | >= reserve | observed event cap | 1/40 (2.5%) | 13303 |
| all 40 | >= initial | uncapped | 7/40 (17.5%) | 13302, 13304, 13312, 13319, 13325, 13409, 13410 |
| all 40 | >= initial | observed event cap | 10/40 (25%) | 13302, 13303, 13304, 13312, 13319, 13321, 13325, 13405, 13409, 13410 |
| prior k5 observation | >= initial | uncapped and capped | 3/15 (20%) | 13302, 13410, 13304 |

The all-40 uncapped full-restoration failure rate is 17.5%, close to the prior
20% k5 observation. That consistency is evidence that the single-day horizon
is too short for a full-restoration condition on a material share of duties;
it is not evidence that restoring energy is inherently the wrong policy. The
25% capped rate combines the horizon effect with the observed event-count
restriction.

Every included baseline and optimum passes exact energy, time, SOC, trip, cap,
and cost replay.

## Shared-service-day unconstrained charger demand

All concurrency below is aggregated on one shared service-day timeline across
the same fixed 30 duties. It is explicitly
**unconstrained charger demand, not a capacity-feasible fleet solution**.
`charge_events` equals `charge_starts`: every emitted positive contiguous
event has one physical start.

| tariff | terminal | optimized arm | starts | kWh | global concurrency | peak kW | max at one station-time |
|---|---|---|---:|---:|---:|---:|---:|
| flat | >= reserve | uncapped | 234 | 8452.0 | 6 | 1440 | 2 |
| flat | >= reserve | capped | 189 | 8452.0 | 7 | 1680 | 3 |
| flat | >= initial | uncapped | 267 | 15622.4 | 10 | 2400 | 8 |
| flat | >= initial | capped | 204 | 15622.4 | 15 | 3600 | 15 |
| peak12 | >= reserve | uncapped | 251 | 8480.4 | 10 | 2400 | 3 |
| peak12 | >= reserve | capped | 218 | 8467.5 | 9 | 2160 | 4 |
| peak12 | >= initial | uncapped | 268 | 15636.8 | 18 | 4320 | 16 |
| peak12 | >= initial | capped | 224 | 15634.0 | 20 | 4800 | 20 |
| two-peak | >= reserve | uncapped | 215 | 8477.2 | 20 | 4800 | 6 |
| two-peak | >= reserve | capped | 192 | 8471.1 | 18 | 4320 | 6 |
| two-peak | >= initial | uncapped | 233 | 15632.4 | 30 | 7200 | 30 |
| two-peak | >= initial | capped | 202 | 15632.4 | 30 | 7200 | 30 |

For the minimally committing two-peak/`>= reserve` case, the unconstrained
optimized schedules demand up to 18–20 simultaneous buses system-wide
(4.32–4.80 MW) and 6 at one station-time (1.44 MW). Under `>= initial`, both
two-peak optimized arms place 30 buses at `PARX_1` simultaneously (7.2 MW).

### Two-peak/`>= reserve` optimized demand over the day

| period | uncapped kWh | uncapped peak buses | capped kWh | capped peak buses |
|---|---:|---:|---:|---:|
| 00–05 overnight | 206.3 | 3 | 206.3 | 3 |
| 06–11 morning | 2052.6 | 6 | 2203.8 | 5 |
| 12–15 midday | 4319.7 | 12 | 4259.7 | 11 |
| 16–21 evening | 1001.9 | 4 | 944.9 | 4 |
| 22–26 late/extension | 896.8 | 20 | 856.4 | 18 |

The full hourly distribution for every baseline and optimized arm is in
`k40_hourly_unconstrained_demand.csv`; exact station/time intervals are in
`k40_station_time_concurrency.csv`.

These measurements quantify how demanding the unlimited-charger assumption
is. They do not establish peak-shaving value. Any charger limit below the
reported station/time demand changes the feasible set and requires
reoptimization.

## Interpretation and claim boundary

On the fixed common population, the realistic-shaped two-peak instrument gives
fixed-sequence timing value of **4.720% uncapped** and **4.755% capped** under
`>= reserve`. This is the strongest complete-schedule behavioural result in
the current branch, but both non-flat tariffs remain synthetic.

This is therefore a **full-schedule software and behavioural experiment, not
the final empirical savings result**. A final Goal-2 estimate requires:

1. a real, frozen, hashed SE3 price series selected by a documented rule; and
2. a decided terminal policy or a periodic/multi-day horizon that resolves the
   observed full-restoration infeasibility.

Real charging tapers above about 80% SOC, so constant power to full is mildly
optimistic. Chargers remain unlimited in this experiment.

## Machine-readable artifacts

- `k40_matched_duty_results.csv`: 240 per-duty/configuration rows covering all
  40 duties, including infeasible arms and per-arm event/load/replay fields.
- `k40_matched_aggregate.csv`: six tariff/terminal rows on the same fixed
  30-duty population.
- `k40_infeasibility_summary.csv`: separately labelled all-40 and prior-k5
  infeasibility rows.
- `k40_solution_metrics.csv`: 24 shared-service-day arm summaries.
- `k40_station_time_concurrency.csv`: exact positive-load intervals by station.
- `k40_hourly_unconstrained_demand.csv`: hourly load and concurrency
  distribution for every arm.
- `k40_sweep_metadata.json`: input hashes, physics, fixed-set rule, and
  producing commit.

Post-artifact full repository test result:
`513 passed, 123 subtests passed in 159.95s`.
