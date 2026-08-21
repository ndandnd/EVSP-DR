# Event-pricer realization and scaling report

Date: 2026-08-21. Branch: `cursor/event-based-pricer-2969`.
No cluster jobs were submitted.

## Outcome

The event approach passes both requested performance gates. It is not archived
as computationally non-scalable:

| cell | certified fleet LP | nodes | logical arcs | wall | peak RSS |
|---|---:|---:|---:|---:|---:|
| k02_s3 | 2.0000000000 | 5,077 | 7,810,190 | 557.18 s | 671.07 MiB |
| k05_s2 canary | 5.0000000000 | 8,784 | 22,161,911 | 1,013.33 s | 1,069.18 MiB |

These are certified lower bounds for the stated conservative expanded-grid
event models: 240 kWh battery, 240 kW charger, zero reserve, 2.5 kWh SOC,
instance-induced event times, and reachable 5-minute containment points. They
are not exact claims about an undiscretized real-world fleet problem.

The k02 gate is below both 8 GiB and 15 minutes. The k5 canary is below 24 GiB.
The canary is the same frozen k05_s2 instance whose prior explicit
representation used 26,814.64 MiB for 22,161,911 arcs. Packed lazy storage used
1,069.18 MiB, a 96.01% reduction, while preserving the same logical graph size.

## 1. Event realization contract

`realize_expanded_path(..., time_model="event")` accepts irregular charging
windows without changing their station, start, or end. It still enforces:

- exact route nodes and trip order;
- continuous arrival/departure timing with no event-mode arrival grace;
- continuous and expanded-grid SOC;
- reserve and battery capacity;
- charger-power limits over each irregular window;
- tariff-hour splitting, identities, and recomputed costs;
- unchanged master-column cost semantics.

The default remains the historical whole-block uniform contract. Event and
uniform realization schemas are distinct.

Strict replay results:

| cell | journal records | accepted | rejected | positive LP witnesses |
|---|---:|---:|---:|---:|
| k02_s3 | 15,882 | 15,882 | 0 | 2 |
| k05_s2 | 16,253 | 16,253 | 0 | 53 |

Every positive LP-support route is bound to its realization and charging-block
hash. The k5 LP support is fractional; “53 witnesses” does not mean 53 buses.

## 2. Uniform scientific identity

With `--time-model` omitted, the regression still byte-compares:

- column journal bytes and ordering;
- route hashes;
- reduced-cost sequence;
- complete `iters.csv`;
- selected scientific status fields and certification.

Operational fields explicitly excluded from scientific identity are `wall_s`,
`attempt_wall_s`, and `peak_rss_mb`. The new peak field is process peak RSS
reported in MiB.

## 3. Factorized event arcs

The event graph no longer stores one Python action dictionary per logical arc.
It stores packed target, cost, and recipe arrays and reconstructs physical
actions only for selected paths.

| cell | packed arc bytes | materialized Python arc objects |
|---|---:|---:|
| k02_s3 | 124,963,040 | 0 |
| k05_s2 | 354,590,576 | 0 |

An explicit representation remains available only as a small-network oracle.
Explicit and lazy modes match the logical arc count, minimum reduced cost,
trip sequence, route nodes, charging record, and fixed-sequence result in the
oracle tests.

## 4. Batch naming

The enrichment method is now named `sink_predecessor_route_batch`. It returns
the exact minimum-reduced-cost route plus at most one best prefix per sink
predecessor. It is explicitly documented as a heuristic and not as
k-shortest-path enumeration.

## SOC finding

Coarse SOC does not preserve the event result across k02:

- at 15 kWh, only k02_s2 reaches 2.0;
- at 10 kWh, k02_s2 is 2.090909 and k02_s3 is 2.081081;
- all three k02 cells reach 2.0 at 2.5 kWh.

The partial executed matrix is in `coarse_soc_evidence.csv`. This agrees with
the separate duty-13411 result: the current event formulation needs 2.5 kWh
for that duty. The memory fix therefore comes from arc representation, not
from discarding the required SOC resolution.

## Tariff identity correction

The first k5 replay rejected 52 otherwise physical late-horizon columns because
generation labeled last-price fallback blocks as requested hour 25 while replay
labeled the same price as source hour 24. One shared event tariff normalization
policy now fills identities through the full horizon. The corrected audit
accepts all 16,253 records; the pre-correction audit is retained in `artifacts/`.

## Validation

- focused API/identity/replay tests: 41 passed, 3 subtests;
- event realization and tariff tests: 21 passed, 3 subtests;
- Python compilation: passed;
- full repository suite: 526 passed and 126 subtests; two deterministic
  historical-artifact checks fail only because their embedded producer hashes
  correctly detect changes to `expanded_path_realization.py` and
  `run_exact_pool_mip.py`. The frozen membership scientific tables are
  byte-identical apart from those producer-code hashes, so they were not
  overwritten.

`results.csv` is the compact executed table. `artifacts/` contains statuses,
iteration logs, phase telemetry, compressed complete journals, run logs, and
strict physical audits. `SHA256SUMS` and `ARCHIVE_MANIFEST.csv` bind them.
