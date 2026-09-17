# Four requested results — 17 September, 08:04 EDT

The reporting gate is ready: 36 scientific endpoints, one censored startup timeout, zero control-integrity errors. This does not authorize new submissions. Numerical fleet bounds and individual route replay are not a separate duplicate-removal dispatch audit.

1. **F5, fresh k15 pools: 0/18 target hits.** Three-hour fleet searches found 16–19 buses; all pool bounds remain 15. These are three solver seeds over six saved fresh-CG pools, not 18 independent instances. The hypothesis that this budget recovers targets on at least four chains is REFUTED. Missing eight/fifteen-bus combinations versus unfinished search remains UNRESOLVED here.

2. **F2/F4, C5 k31: 31 buses, pool bound 30 after 12 h fleet search plus 30 min charging.** The target is matched; feasibility of 30 remains UNRESOLVED. Individual route replay passed; duplicate cleanup was not performed by this repeat. Extra time did not close the one-bus gap.

3. **F6, constrained k5 three-arm costs:**

| Tariff peak | Original invoice interval | Fixed-duty optimization | Fresh-CG routes, continuous charging |
|---|---:|---:|---:|
| 08:00 | 230.287–230.981 | 157.965 | 154.921 |
| 12:00 | 289.594–290.597 | Censored startup timeout | 195.214 (one duplicate trip) |
| 18:00 | 223.447–223.723 | 107.189 | 104.094 (one duplicate trip) |

240 kWh battery, 350 kW ceiling, 15% reserve, minimum three-minute active charging, zero start fee; common minimum aggregate ending energy 280.7833253 kWh, no shared station capacity. All available optimized schedules use five buses. Morning fresh-CG cost is 1.93% lower with no duplicated trips and more ending energy (289.44 versus 281.17 kWh): VERIFIED against this fixed-duty comparator. Evening difference is 2.89% with equal achieved aggregate ending energy, but dispatch validation remains pending. Noon comparison is UNRESOLVED. No global continuous charging optimum is claimed. Original invoice is bounded because the actual within-session power profile is unobserved.

4. **F2/F5, k32 seed variability:**

| Chain | Seed 0 buses | Seed 1 buses | Seed 2 buses | Population variance (buses squared) |
|---|---:|---:|---:|---:|
| C1 | 33 | 32 | 32 | 0.2222 |
| C3 | 33 | 32 | 32 | 0.2222 |
| C4 | 32 | 32 | 33 | 0.2222 |
| C5 | 32 | 32 | 35 | 2.0000 |

Three hours fleet search plus 30 minutes charging per seed. Same ordered pool, initialization and allowances verified within each chain. Variance uses divisor 3; sample variances (divisor 2) are 0.3333/0.3333/0.3333/3.0000. Solver search variability is VERIFIED; these four selected pools do not estimate population-wide success rates. C1/C3 matched results have pool fleet bound 32; C4/C5 bound 31 remains open. Separate duplicate-removal audits are pending.

## Additional monitored endpoints and failures

- Random-trip-group inheritance reaches all 364 trips: 21 buses versus GIRO 15, pool bound 15, no proof; CG wall-capped, route weight 15. This single order does not isolate duty grouping causally. Graph+CG+MIP recorded process time totals 104175.8 s (28.94 h); do not treat this as CG-only time.
- Frölunda k15: worker rejected CG output with one artificial variable after pricing stopped; final recorded route weight 15 is not a feasible 15-bus solution. The saved JSON says `certified`, but artificial mass remains 1.0. Earlier k1/2/3/5/8/10 results remain separate. No retry submitted.
- Job341164 (C2 k15 evening CG) timed out before recorded worker startup, empty stdout/no case directory, on jingjie-cpu-13. Job341166 has a scheduler TIMEOUT; its real-price scientific data remains internal and was not collected here.
- Six action3 arms still in replay, 43–50/125 complete shards; no final scientific comparison yet. Full-Partille CG remains dependent on graph preparation; MIP held.

Sources: `readiness.json`, focused snapshot `execution/monitor/20260917T120440Z/snapshot.json`, and `execution/monitor/20260917T120440Z/new_failures.json`. Frozen 128/102/67/35 Doc audit unchanged.
