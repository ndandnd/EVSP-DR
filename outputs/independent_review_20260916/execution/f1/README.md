# F1 — serving each trip once while retaining empty driving

This audit converts the selected covering schedules into **service-assignment partitions**. Exactly one bus serves each trip; other buses that traversed the same trip now drive that leg empty. It preserves every vehicle movement, charging event, time and modeled energy amount. It does not shorten routes or optimize charging again.

## Finding tested

| F1 claim | Verdict | Scope |
|---|---|---|
| Duplicate coverage necessarily makes the reported fleet infeasible. | **REFUTED under the baseline model with empty driving permitted.** | Assigning each passenger trip to one occurrence yields exact service once, with the original vehicle trajectories retained. See replay results below. |
| Empty-driving conversion preserves fleet and physical feasibility. | **VERIFIED for the audited schedules under baseline physics**, as checked by all 128 fresh replays. | Battery 240 kWh, constant 240 kW charging, zero reserve, original restricted movement graph. This does not add GIRO's omitted constraints. |
| Charging costs become invalid merely because a covering has duplicate trips. | **REFUTED as a blanket claim.** | Conversion preserves the charging schedule, so both grid and continuous realized costs remain identical. All 3,341 selected routes' persisted charging blocks were independently recomputed against the hashed tariff and matched their saved continuous costs. |
| Conversion automatically provides exact-partition columns in the existing saved pool. | **REFUTED.** | A saved column still marks every visited service-trip node as covered. The new service assignment is an external dispatch overlay. Feeding the original columns unchanged into an equality master does not implement this overlay. |
| Covering and partitioning always have the same optimum, regardless of allowed route set. | **REFUTED without a closure assumption.** | Equality follows when the partition route family permits retaining any service leg as an empty traversal with unchanged cost. It need not hold for a finite pool lacking those relabeled alternatives or a route family that forbids them. |

The costs are feasible dispatch costs that include potentially unnecessary empty driving. They need not be the best achievable costs after changing vehicle movements. Deleting duplicate legs and recomputing charging is a different optimization experiment; this audit claims no resulting savings.

## Completed checks

| Check | Result |
|---|---:|
| Schedules replayed successfully | **128 / 128** |
| Selected physical routes | **3,341** |
| Passenger-service occurrences, exactly one per input trip in each schedule | **76,076** |
| Duplicate occurrences retained as empty drives | **14,450** |
| Schedules preserving fleet, movement, charging and costs | **128 / 128** |
| Routes left with no assigned passenger trip | **0** |

Counts sum across repeated/nested experiments. They are not counts of distinct trips in the underlying data. Restricted-graph construction and route replay took about **5.4 minutes** in total, with no optimization.

## Conversion and replay

For each trip, the occurrence on the lowest-index selected route is designated `service`; every remaining occurrence becomes `empty_drive_same_trip_path`. The ledger records original route index, node position, local and original GIRO trip ID, fixed start/end time, and energy. Repeated trips within one route are absent in the source cohort.

The original physical route object is retained unchanged. A fresh replay uses `validate_injected_route` from pinned MIP commit `871d057e1067411f09581e37d78f7c1ca43f68bb`: original graph arcs, travel times, trip times/energy, charging-window energy/power, battery capacity, reserve and 26-hour horizon. The first pass retained its original one-minute charging-arrival grace. A separate second pass set that grace to **zero**: **all 128 schedules and 3,341 routes still passed**, so this result does not depend on accepting arrivals after the recorded charging start. Module hashes are checked against the pin. A second pass validates persisted continuous charging blocks, their tariff/power provenance and continuous realized costs.

Input hashes match the original MIP physical-pool records and frozen source tables. Reference/deadhead/tariff files are separately hashed against those records. Original selected-route and charging hashes remain unchanged. The dispatch ledger is a new artifact; original MIP results are untouched.

**Shared station capacity was not validated**, and the baseline model does not impose the full GIRO charging/SOC rules. Since physical schedules are identical, this conversion preserves existing station occupancy rather than fixing possible overloads. It also creates no new LP or integer optimality proof.

## Why the route-set qualification matters

Let C be the available covering routes. For any selected covering, assign each trip to one selected route that visits it. If a route can retain its other trip legs as empty driving, this gives a dispatch partition with the **same fleet and cost**. Conversely, any such partition is a covering when one ignores the passenger-service labels. Thus covering and partitioning have equal optima for this physical route family closed under relabeling.

The existing finite-pool equality master has columns with fixed coverage vectors, and may lack the required relabeled vectors. Consequently its optimum can differ from the covering optimum even when the covering has a physically valid exact-service dispatch. Relabeling alone also does not prove anything about an unrestricted road network or omitted operational constraints.

## Strict charging-arrival sensitivity

`strict_arrival_summary.json` and `strict_arrival_replay_results.json` retain the zero-grace replay separately. All 128 schedules pass with zero charging-arrival grace; there are no failed routes. The original baseline replay is preserved.

## Artifacts

- `summary.json` and `per_case.csv`: verified counts and per-schedule outcomes.
- `replay_results.json` (also losslessly compressed as `.json.gz`): all route records and service/empty-driving dispatch ledgers; each result has its original path/hash and unchanged physical-schedule hash.
- `charging_validation.json`: all 3,341 route charging-block checks, recomputed costs, reference-data hashes and dependency module hashes.
- `replay_remote.py`, `verify_blocks_remote.py`: read-only remote audit code. No optimizer calls or expanded-network cache rebuilds.
- `run.py`, `requests.json`, `replay_progress.log`: frozen cohort, execution and progress record.
- `summarize.py`, `provenance.json`: derived summary and artifact hashes.

The cohort is the same 102 original and 26 longer-MIP selections in the F2 audit, frozen at `status_20260916T194843Z`; these are overlapping/nested experiments, not 128 independent data sets. Original and longer MIP results remain separately labeled.
