# 08:36 EDT results — 11 September 2026

Warm chain 5 now matches k=9 with nine buses. Both MIP stages are optimal within its saved pool; selected routes passed individual replay. Ten trips are overcovered, so duplicate removal and shared-capacity validation remain separate.

## Capacity pilot

| Case | Scheduler outcome | Scientific result |
|---|---|---|
| Duty 13406, combined capacity + depot speed (task 7) | CG COMPLETED after 8h34m; MIP COMPLETED | CG stopped at its time budget without a pricing certificate. Saved 35-column pool yields one bus, both MIP stages optimal; charging-related cost 56.952. All 14 trips covered exactly once and station-capacity sweep passes. Individual route feasibility is by event-model construction, not a separate full GIRO-physics replay. |
| Tasks 5, 9, 11, 13, 15 | CG TIMEOUT after 9h02m | No final result or infeasibility/optimality conclusion. These are not preemptions. |

The successful case proves the constrained model can produce a one-bus schedule for this duty; it does not measure the effect of lower depot speed in isolation. The other arms and charging locations still matter.

The current driver can exceed the total loop budget inside a long pricing call and does not preserve intermediate pools. Sol high is working on a bounded deadline/checkpoint fix with tests and immutable provenance. No blind repeat of the five timed-out cases was launched. Original attempts remain preserved.

A collector gap was also fixed: retry artifacts live under `results/<case>/`, using the pilot schema. The register now includes the successful CG and MIP artifacts under the retry campaign rather than silently missing them. The retained 08:34 snapshot predates this schema fix; the canonical 08:36 snapshot includes it.

Default MIP reliability remains 75 completed attempts and zero recorded preemptions. Warm-chain work remains active. See `evidence.json` and `capacity_accounting.txt` for sources and exact proof scopes.
