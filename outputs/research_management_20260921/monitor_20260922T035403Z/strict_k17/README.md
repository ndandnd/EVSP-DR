# Strict packed k17 subgroup endpoint audit

Job **668432 completed**, but the 18E2 subgroup target of **10 buses was not attained**: final fleet **11**, saved-pool integer bound **10**, fleet proof unresolved. This case contains **277 trips / 10 reference GIRO duties** within C5's global prefix k17; it is not a 17-bus target experiment.

| Evidence | Audited endpoint |
|---|---|
| Scheduler | COMPLETED, exit 0:0; 21 September 20:02:51–21:03:16 cluster time; elapsed 1:00:25 |
| Resources | 8 CPUs, 24G, snavely-cpu-04; batch MaxRSS 1,073,960 KiB |
| Fleet stage | TIME_LIMIT; 11 buses / bound 10; gap 9.0909%; 1,800.120801 s |
| Charging stage | TIME_LIMIT; fleet cap ≤11 from unproved stage-one incumbent; 1,800.117828 s |
| Charging-related objective | 774.872; bound 380.338910643; gap 50.9159047% |
| Budget | 3,600 s nominal; two optimize-stage timers sum 3,600.238629 s; wrapper 3,620.977297 s |
| Saved pool / CG | 8,343 columns; CG pricing deadline, **uncertified**; restricted fractional weight 10.00000000000018 |
| Coverage | All 277 trips; 350 assignments: **73 extra assignments across 61 trips** |
| Shared station capacity | Omitted from optimization; diagnostic **fails** at 7880C (3 connections / 1 charger), JON_A (4 / 1) |
| Selected-route metadata | All 11 carry `valid_event_time_realized`; inherited 3,086 routes have recorded full replay |

Fleet and objective values were checked against the full two-stage Gurobi log. Selected pool rows independently reproduce coverage, duplicated-trip counts, cost and the half-open station-capacity sweep. Source/result/manifest/input/CG/pool hashes match; the execution source matches clean commit `35770aae2c08e7d5a356cc3b673e67608e5b1036`. All ten audit checks pass.

## Physics and proof limits

Battery/initial SOC 239.01 kWh, reserve 35.8515 kWh; PARX 60 kW, other sites 240 kW; 2.5 kWh SOC and 5-minute event grid; reserve-only terminal SOC; unlimited PARX capacity, no shared station-capacity enforcement. Master is set covering. CG uses 100000 × fractional route weight plus expanded-grid charging-related cost; MIP minimizes fleet then that charging-related cost. The reported 774.872 is not a separately repriced continuous-realization objective.

The JSON's `validated_incumbent=true` is limited: code requires covering validity, and requires the station-capacity sweep to pass **only when capacity is enabled**. Here capacity is disabled. Individual route feasibility is supplied by exact-event construction/realization metadata; this bounded audit did not run a new independent SOC/time replay. No exact-once dispatch, shared-capacity feasibility, full-GIRO validity, fleet optimum, charging optimum, or full-model lower bound is asserted. The uncertified restricted LP weight is not a certified full-model bound.

## Evidence

- [Compact endpoint JSON](verified_endpoint.json), [editable one-row table](verified_endpoint.csv), [reproducible audit](audit.py).
- [MIP result](sources/result.json), [complete Gurobi log](sources/result.json.gurobi.log), [command](sources/command.json), [completion receipt](sources/COMPLETE.json).
- [Collection and remote SHA-256 receipts](collection.json), including only 15 selected stage-one/stage-two pool rows; [scheduler](scheduler.txt), [clean pinned source receipt](source_commit_receipt.txt), [collector](collect.py).
- [Campaign manifest](sources/manifest.json), [CG completion/hash chain](sources/CG_COMPLETE.json), [execution source](sources/run_capacity_speed_event_cg.py).

Only this completed endpoint was queried. Eleven small files totaling 150,369 bytes plus selected-row/hash receipts were collected; no whole pool or large journal was downloaded. No queue-wide checks, submissions, source mutations, or document/Slides edits were performed.
