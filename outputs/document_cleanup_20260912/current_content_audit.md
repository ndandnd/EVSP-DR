# Current content audit — 12 September 2026

Read-only audit for the current EVSP–DR document/slide cleanup. The latest monitor snapshot is `2026-09-12T04:20:01Z` ([`20260912T041951Z.json`](../post_meeting_20260910/monitor/20260912T041951Z.json)); the overnight campaign snapshot is `2026-09-12T04:14:35Z` ([`cluster_snapshot.json`](../overnight_extension_20260912/cluster_snapshot.json)). The chain tables below are the register view verified at 11 September 23:51 EDT and should remain date-labelled ([`CURRENT_CHAIN_TABLES.md`](../research_register/CURRENT_CHAIN_TABLES.md)).

## Current status to put in the front matter

The 11–12 September overnight extension is a launch in progress, not a new scientific result. Its README records 174 research tasks: 37 CG + 37 dependent MIPs for six inherited chains, 40 CG + 40 MIPs for ten 32-duty decompositions, and 10 recombination CG + 10 MIPs. At the latest monitor timestamp, the six visible inherited-chain CG cases (`w1_k07`, `w2_k09`, `w3_k11`, `w4_k10`, `w5_k11`, `w6_k11`) are all `initializing`, with zero iterations, no final LP, and `certified_rc_optimal=false`; no overnight MIP result is present. Do not report a k11–15 result, a decomposition result, a certificate, or a target-attainment result from this campaign.

The campaign uses default-partition CPU jobs and excludes `scaglione-compute-01`. It is the baseline covering cohort: 240 kWh battery, 240 kW charging, event 2.5 kWh / 5 min, zero reserve, no return-SOC floor, no shared-station capacity, flat tariff, and route objective `100000 + electricity + 5 per charging start`. Bounded inheritance selects at most 512 predecessor sequences and replays them for at most 900 seconds on 8 workers. The accepted subset can depend on parallel completion; no duals, basis, or pricing certificate are inherited. These settings and limits are from [`overnight_extension_20260912/README.md`](../overnight_extension_20260912/README.md).

The latest scheduler view has 50 running and 227 pending records across the monitored queue. Scheduler state is operational evidence only; it is not a solver result or a validated scientific result.

## Warm versus fresh chain highlights

Chain order is C1–C6. A number is the integer fleet in the saved route pool; `—` means no completed result. The target is the row label.

| target k | warm / inherited C1–C6 | fresh / independent C1–C6 |
|---:|---|---|
| 2 | 2, 2, 2, 2, 3, 2 | 2, 2, 2, 2, 3, 2 |
| 3–4 | all target | all target |
| 5 | 5, 5, 5, 5, 5, 5 | 5, 5, 5, 5, 6, 5 |
| 6 | 6, 6, 6, 6, 6, 6 | 6, 7, 6, 6, 7, 6 |
| 7 | —, 7, 7, 7, 7, 7 | 8, 7, 8, 8, 8, 7 |
| 8 | —, 8, 8, 8, 8, 8 | 9, 9, 9, 9, 9, 8 |
| 9 | —, —, 9, 9, 9, 9 | 11, 11, 11, 10, 10, 10 |
| 10 | —, —, 10, —, 10, 10 | 11, 11, 11, 11, 11, 11 |
| 11 | —, —, —, —, —, — | 13, 12, 13, 13, 12, 12 |
| 12 | —, —, —, —, —, — | 14, 13, 14, 14, 13, 14 |
| 13 | —, —, —, —, —, — | 15, 15, 15, 15, 14, 15 |
| 14 | —, —, —, —, —, — | 17, 17, 16, 18, 16, 17 |
| 15 | —, —, —, —, —, — | 18, 17, 18, 19, 16, 19 |

The warm table has results only through selected k10 cells. C1 k7 timed out in initialization and blocks its descendants; C2 k9 exhausted its import budget with no final LP and its MIP export failed; C2 k10 remains initializing; C4 k10 timed out in initialization. Chain 3 k10 also contains validated routes from a fresh solver solution, so it should not be described as a pure GIRO-seeded continuation. The fresh grid has 84 completed cases and the values above are integer incumbents with differing MIP proof status. Both tables use set covering and baseline physics. Individual route replay passed where recorded, but these tables do not certify duplicate removal, shared-station capacity, unrestricted integer completeness, or charging optimality. Warm fleets are proved only within their saved pools.

## Capacity pilot: useful comparison, narrow claim

| case | baseline | PARX60 only | capacity only | capacity + PARX60 |
|---|---:|---:|---:|---:|
| k1 duty 13408 | 1 | 1 | 1 | 1 |
| k1 duty 13406 | 1 | 1 | 1 | 1 |
| k2, 23 trips | 2 | 2 | 3 | 3 |
| k3, 35 trips | 3 | 3 | 16 | 16 |

All 16 cells have MIP outcomes and the capacity-constrained selected schedules pass station sweeps, but six recovered CG cases are uncertified. The k3 capacity runs added only three columns in eight hours; one pricing call took 7.15 hours while its LP took 0.006 seconds. The value 16 is a saved-pool result, not a proved physical requirement. Baseline k2/k3 schedules violate the one-space station limit. This pilot keeps 240 kW opportunity charging, uses zero reserve, and does not impose the 65% return-SOC floor, nonlinear charging, or driver rules; the separate five-bus/350 kW aggregate return-energy tariff experiment is not combined with it.

Source: [`CURRENT_CHAIN_TABLES.md`](../research_register/CURRENT_CHAIN_TABLES.md), [`fresh_covering_complete84.csv`](../parallel_research_20260911/results_20260911T1134Z/fresh_covering_complete84.csv), and [`capacity_deadline5_completed/records.json`](../parallel_research_20260911/capacity_deadline5_completed/records.json).

## Current matched-return-energy charging comparison

The current comparison is the separate five-bus / 62-trip cohort at 240 kWh initial energy per bus and 350 kW charging. Both optimized arms enforce the verified common aggregate terminal-energy minimum of **280.7833253 kWh**, equal to the original GIRO replay total. There is no inferred 65% return-SOC floor. “Fixed duty” keeps the original ordered duties and reoptimizes charging; “joint” uses the immutable saved CG pool plus all fixed-duty terminal-frontier routes. Both use a fleet cap of five, no shared-station-capacity constraint, and a modeled 5-unit fee per charging start.

The chart [`charging_current.png`](charging_current.png) uses the **physical continuous-replay cost** for both optimized arms, with the original GIRO repriced interval shown as an errorbar. This keeps one physical-cost scale in the figure; the expanded-grid objective remains a separate table column because it is the exact finite-grid MIP objective. The original GIRO value remains an interval because within-window power is unobserved; it is not a continuous replay exact value. The physical cost is an audited replay accounting value, while continuous cost pricing is not certified as a new global optimum.

| tariff | original GIRO repriced interval | fixed expanded-grid objective | joint expanded-grid objective | fixed physical replay | joint physical replay | common terminal check |
|---|---:|---:|---:|---:|---:|---:|
| peak08 | 490.287010–490.981075 | 279.899879 | 261.569764 | 275.873182 | 257.687122 | grid 281.170001; replay 284.250003 |
| peak12 | 549.593536–550.596584 | 332.025525 | 332.025525 | 328.548226 | 328.358731 | grid 281.940001; replay 282.710002 |
| peak18 | 483.447291–483.723228 | 232.249934 | 217.505089 | 230.782136 | 215.756343 | grid 282.440001; replay 289.460004 |

All values are from the retry comparison records in the latest monitor: peak08 record SHA-256 `c5bd813c9f2c8f727150be77a95b3f80db2990a83b0fee6d6b0944bbdf87335d`, peak12 `d2f8ddfce55c5dfcc1680ba0c2204fd9305711f8c454667d8b0a220028fbc008`, and peak18 `dcff1c7807b643cfad4b0642d506bc30c98162a98ddbde4a43b2e567afb47037`. The shared instance hash is `b386f8a16958d25c857297ac4643bf6c73ae2114557c585446725cbd51c8b64d`; tariff hashes are peak08 `b461f12cf8ca41ba8a89102b32d46ea16b974566ea05849dd75776a7108f341e`, peak12 `8b231a2574fd4e3b4dc94873ad2d6515bfaba09e07afefbb1df43d5f775a8381`, and peak18 `d70c3c3f9c15b82f26e1262fb2067f9fcc5e41d63e9ffbe877d33b5fa16563a6`. The execution commit is `2424369f4b5c40198a22698b7a460d6aa8129169`. Source paths are the three `terminal_energy_fair_mip_retry_5cdb813_20260910/<tariff>/comparison.json` files recorded in the snapshot; method and validation scope are in [`terminal_energy/README.md`](../post_meeting_20260910/terminal_energy/README.md).

Each retry record reports stage-1 fleet 5 with `proven=true` and stage-2 MIP gap 0 in its saved pool. The fixed arm is optimal over its enumerated event-graph terminal frontiers; the joint arm is a finite-pool two-stage MIP result without a new full-model pricing certificate.

The earlier bars and table with fixed/joint physical values 267.335/242.553, 320.088/304.777, and 211.828/182.350 are the **unequal-return-energy** comparator and belong in the historical archive. Remove them from the current narrative. The prior k10 flat/peak08 equality check is also historical tariff-extension evidence and should remain date-labelled rather than being blended into this matched comparison.

## Measured import bottleneck in warm full-pool inheritance

[`import_time_current.png`](import_time_current.png) shows the measured inherited full-pool import/validation phase stacked against all other CG work. Values are minutes from the latest cluster snapshot; C3 k8, C3 k10, C5 k5, and C5 k6 agree with the current research export, and C5 k10 is a certified snapshot case included from the same phase fields.

| case | inherited full-pool import/validation (min) | other CG work (min) | status |
|---|---:|---:|---|
| C3 k8 | 133.19 | 3.88 | certified |
| C3 k10 | 282.59 | 9.98 | certified |
| C5 k5 | 59.12 | 3.01 | certified |
| C5 k6 | 91.08 | 10.73 | certified |
| C5 k10 | 330.42 | 11.49 | certified |

The source export is the current research Markdown snapshot (SHA-256 `9d47e145bfd6fdc99f1d3ddeeef236c480c8cb3ca6d54c6ea164dfaedeb05708`); phase values and C5 k10 come from [`cluster_snapshot.json`](../overnight_extension_20260912/cluster_snapshot.json), SHA-256 `281cfaa38eddefc60c70397be0cd3b19b24df29d6375a98ece29b7ab42edaed3`. The C3 cases are in `/home/nc437/ladder-lite/nested_warm_chain_p3_k2_10_20260909_8830a34`; the C5 cases are in `/home/nc437/ladder-lite/nested_warm_chain_p5_k2_10_20260910_ecb60c1`. These are phase timings, not causal speed claims; the imported predecessor pool is validated/replayed but does not carry duals, bases, or an inherited pricing certificate.

Chart provenance is recorded in [`figure_provenance.json`](figure_provenance.json) and [`generate_current_figures.py`](generate_current_figures.py): `charging_current.png` SHA-256 `ebb53bf730e154fcc4919f79bb1e9fe58584becc1dc5fd740eca9a55e69e4412`; `charging_current_slide.png` `fee2b32bfa3465d9a9c53b97ece1c8ddee11a225c03511c68aca71ed8f5639e1`; `import_time_current.png` `2ad4416113086f4916da7873407a42e43f44bb97da953e58938f12a9b6a53741`; and `import_time_current_slide.png` `568f174200a8b2c79dc432c08c0538c342c90a8c1780c2f575e91e2a2af3818c`. The slide variants use the same data at `figsize=(10.5, 4.3)`. All were rendered with the bundled Python executable and Matplotlib; the figures contain axes, ticks, and legends only, with no embedded title or caption.

## Proof and wording boundary

Use three separate labels in every table or figure:

1. A CG pricing certificate applies only to the named graph/model and stated reduced-cost tolerance. A running or time-limited RMP objective is not a full-model lower bound. For a bus-plus-charging objective, the certified monetary objective is not a fleet lower bound.
2. An integer MIP `OPTIMAL` status proves the supplied finite route pool, subject to the recorded validation scope. It does not prove unrestricted full-model integer optimality or charging optimality.
3. Physical route replay, duplicate/overcoverage checks, and shared-station capacity are separate fields. A scheduler `COMPLETED` state does not establish any of them.

Keep fractional route weight, weighted objective, fleet-only lower bound, and pool MIP proof as distinct quantities. This follows the register conventions in [`research_register/README.md`](../research_register/README.md).

## Obsolete claims to remove from the current narrative

- “The overnight extension produced results/certificates through k15.” It is running and has no completed overnight CG or MIP result in the latest snapshot.
- “All 174 overnight tasks completed,” or any use of launch counts as completed experiments.
- “Warm chains through k15 are complete.” The current verified warm table stops at the listed k10 cells; no warm k11–15 result is complete.
- “The fresh 84-case grid proves the unrestricted optimum or charging optimum.” It contains finite-pool integer incumbents; MIP proof status differs by case.
- “Matching the GIRO bus target enforces GIRO.” The green entries mean target fleet under the stated covering model; GIRO operational constraints are not all enforced.
- “The capacity pilot proves a 16-bus physical requirement.” Sixteen is a saved-pool outcome, with uncertified capacity CG cases and a narrow physics scope.
- “PARX60 caused a fleet or cost improvement.” The pilot rows show equal fleet values for the paired baseline/PARX60 cases; these small comparisons do not support a causal claim.
- “Tariff savings are verified operator invoice savings.” The tariffs and 5-unit start fees are modeled; terminal-energy credits and original cost intervals are accounting/model comparisons.
- “`OPTIMAL` means full-model optimal.” Use “optimal in the supplied pool” unless a separate full route-space pricing certificate and the matching proof scope are present.
- “Current runs enforce the full GIRO physics,” including the 65% return-SOC floor, nonlinear charging, driver rules, or shared-station capacity. The baseline campaign does not.

Date-stamp older plots and summaries under a historical archive section. Preserve their source path and proof scope rather than blending them into the current overnight status.

## Suggested seven-slide current deck outline

1. **Status and scope.** Timestamp `2026-09-12T04:20:01Z`; overnight extension running; six visible warm CG initializations; no new result claim. Add a small status panel and a proof legend.
2. **Baseline model and proof map.** One compact model-settings table (240/240, event 2.5/5, covering, flat tariff, zero reserve) beside a three-row map: CG certificate, finite-pool MIP proof, physical validation.
3. **Fresh k2–15 fleet matrix.** Heatmap/table of the 84 fresh incumbents, with target diagonal and the exact high-k rows (k11–15) shown as ranges/values. Caption that MIP proof status varies.
4. **Warm/inherited continuation.** The warm C1–C6 table through current k10 coverage, with callouts for C1 k7 initialization timeout, C2 k9 import exhaustion, and no warm k11–15 completion.
5. **Capacity pilot.** Four-column comparison table above plus one figure/callout for the k3 3 versus 16 saved-pool outcome. Put the station-sweep and missing-physics limitations directly on the slide.
6. **Charging response.** Use `charging_current.png`: original GIRO repriced interval plus fixed-duty and joint physical replay costs at peak08/12/18. Put the exact expanded-grid MIP objectives beside them in the companion table and keep the common 280.7833253 kWh terminal-energy floor, modeled-fee, and finite-pool caveats visible.
7. **Measured inheritance cost and archive.** Use `import_time_current.png` for the full-pool import bottleneck, then date-label the unequal-return-energy plots and k10 flat/peak08 equality under a historical archive section. Link each figure to its source artifact and retain the current-versus-historical boundary.
