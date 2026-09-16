# P3 item 14 / F8 — Frölunda generalization pilot

**Submitted: job 341413.** One 48-hour job on the default partition, 8 CPUs / 64 GB, requeue enabled; `scaglione-compute-01` excluded. The scheduler receipt and native startup snapshot are saved here. The first full-duty ladder result is now available below; it is separate from the smoke tests.

| Reference duties | Service trips |
|---:|---:|
| 1 | 15 |
| 2 | 38 |
| 3 | 67 |
| 5 | 109 |
| 8 | 166 |
| 10 | 209 |
| 15 | 352 |

These are nested prefixes of a seeded random order of all **61 duties / 1,393 trips** in the Frölunda source. Seed **20260916** and duty order were frozen before feasibility testing. No rejected/easy duties were removed or reordered.

Each stage builds its event graph, runs up to **four hours of CG**, then runs a **one-hour two-stage MIP**: 30 minutes minimizing fleet, then charging cost subject to fleet ≤ the incumbent. The first CG starts with singleton routes; later stages retain all earlier columns and add new singleton routes. No GIRO routes are inserted. Graph construction has a separate six-hour cap and separate timing records. The overall 48-hour allocation may end before every stage finishes.

## First completed research endpoint

The unscreened first duty (**15 trips**) completed CG in **2.85 seconds / 30 iterations**, with a pricing certificate, zero artificials and fractional route weight **1**. Final weighted LP objective: **100,035.832**; last reduced cost: approximately **−4.4×10⁻¹¹**. The MIP found **one bus** and proved fleet **1** in its saved pool. The selected route covers each of the 15 trips exactly once and passed physical replay. Shared charging capacity was not checked.

Raw CG/MIP JSONs and hashes are retained in `first_endpoint/` and `status_after_submission.json`. A later verified **k2 / 38-trip** endpoint also uses **two buses**, covering every trip exactly once. Both routes pass an independent zero-arrival-grace replay. CG reports a certificate in **51.64 seconds / 110 iterations**. An overlapping pair of mandatory trips independently proves a fleet lower bound of two, matching the feasible dispatch; see `first_endpoint_k2/`. This does not prove charging optimality or compliance with omitted GIRO constraints. Later-stage results are outside this audit.

## Model and input assumptions

- **240 kWh battery, uniform 240 kW charging**, zero SOC reserve, no terminal SOC requirement, unlimited shared station capacity. This is an exploratory baseline, not full compliance with GIRO constraints.
- Set covering; objective **100,000 per bus + electricity + 5 per charging start**. Event representation uses 2.5 kWh / 5 min, 30 columns per iteration and reduced-cost tolerance 0.0001.
- Depot **KEX**; seven charging locations are taken from the workbook. The isolated CG/MIP code changes only depot/station configuration and the corresponding audit lookup.
- Service-trip energy is preserved from `FDL_VehicleDetails.xlsx`. Deadhead duration/distance use the **maximum across both directions and all available time intervals**; deadhead energy assumes **2 kWh/km**. This conservative conversion is declared, not a claim about actual Frölunda consumption. One source row with missing distance/duration was omitted and recorded; no shortest-path imputation was applied.
- The original constant tariff covered hours 0–24. We explicitly extended the same **0.0992/kWh** price to hour 25 for the 26-hour horizon. Original CSV, hash, extension reason and new hash are retained.

## Recorded-setting correction

The frozen manifest incorrectly labels the station-to-trip window as 220 minutes. Executed CG and native MIP replay explicitly use the 1,560-minute horizon. The independent input and selected-route checks used the stricter 220-minute window and passed. The original manifest and its launch hash are preserved; `manifest_erratum.json` records the correction and pinned code evidence. CG certificate scope uses the executed 1,560-minute setting.

## Validation before submission

All seven restricted input graphs passed depot-connectivity and input checks with the a stricter **220-minute station-to-trip window** than the **1,560-minute window actually used by CG and native MIP replay**. Optimizing charging continuously for the unchanged first full duty produced a physically validated feasible route; this did not alter selection.

A separate **one-trip** native event-CG smoke run obtained a pricing certificate. Native MIP found and proved one bus and passed physical replay. A second native CG run successfully resumed its saved checkpoint with matching input/code hashes. These validate plumbing and identity; they are not seven-stage research results.

CG code: `daac18d5598902c28baa6c43e58942930fc733bc` (base `a0e0bb76`). MIP code: `0c51b1fa491492212f5d875d1c082b476db0f1c3` (base `871d057e`). Their frozen bundles and diffs are included. `fdl_entry.py` points the clean CG checkout at the hashed Frölunda data directory, so Partille reference files are not overwritten in tracked code. MIP uses its explicit data-directory arguments.

## Progress and preemption

Remote root: `/home/nc437/ladder-lite/review_frolunda_20260916/`; large case outputs are under `/share/scaglione/nc437/evsp-dr/review_frolunda_20260916/`.

`collect.py` reports each stage's graph/CG/MIP completion, attempts, source hashes and separate scientific statuses. Source artifacts live in `cases/<case>/<cache|cg|mip>/<job>_r<restart>/`. Completed-stage receipts bind output hashes to the frozen manifest. Requeues skip validated completed stages and resume CG using copied, identity-checked native checkpoints. An interrupted MIP starts a new tree; no search-tree continuation is claimed.

The held historical campaigns are untouched. The one-job design deliberately runs stages sequentially; it is not an array throttled below the resource policy.

## Evidence files

`inputs.json`, `data/`, `prepare_inputs.py`: workbook provenance, selected duties and declared conversion. `input_validation.json`, `k1_fixed_smoke.json`: full-duty input/charging checks. `smoke_validation.json`, `resume_validation.json`: native pipeline tests. `manifest.json`, `jobs.json`, `submission.log`: frozen settings and scheduler receipt. `worker.py`, `worker.sub`, `fdl_entry.py`, `resume_entry.py`: execution and restart behavior.
