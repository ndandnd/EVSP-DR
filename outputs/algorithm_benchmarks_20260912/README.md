# Astra measurements of proposed algorithm improvements

Date: 2026-09-12. This follows the [independent review](../algorithm_review_20260912/REVIEW.md). The earlier pass identified candidates and ran existing tests; it did not implement or measure those candidates. The research owner confirmed that none had since been deployed. This follow-up used Astra for two prototype implementations and an independent audit, with the parent implementing the master benchmark.

**There are now measured local gains, but no measured full-CG or cluster speedup.** The prototypes remain in this output directory. Production solver files, campaign settings and cluster jobs were unchanged. Their intended benefit is faster computation with the same optimization results. No new fleet or charging-cost saving is claimed.

## Quantified results

| Change and measured workload | Before | After | Interpretation |
|---|---:|---:|---|
| Indexed replay: 210 generated sequences on a 48-trip real-data subset, production SOC2.5; graph already available, cold action caches, index setup included | 0.558 s | 0.372 s | 1.50× faster for this batch |
| Same replay, index prepared and action/window caches warm | 0.566 s | 0.058 s | 9.75× for replay alone; excludes 0.311 s index preparation |
| Skip unused incidence matrix: synthetic covering master, 80 trip rows, 1,040 final columns, 14 solves | 78.10 ms | 58.09 ms | 1.34× for the master replay, including its model setup and audits |
| Capacity-window selector, 12 different synthetic windows; cold setup included | 24.107 ms | 15.893 ms | 1.52×; source-accounting blocker below |
| Capacity-window selector, 120 calls sharing 12 keys; cold setup included | 235.785 ms | 16.129 ms | 14.62× with 90% memo hits; source-accounting blocker below |

Ratios are baseline time divided by prototype time. Values are local medians from six alternating-order repetitions for replay/capacity and seven for the master. They are small-workload measurements on this Mac, not predictions for Unicorn. Full raw samples and input/source hashes are linked below.

Two negative results matter. A single cold capacity-window query slowed from 0.850 to 0.981 ms. Tiny explicit-graph replay also sometimes slowed. Setup and lookup overhead must be justified by the actual workload.

Replay's graph contained 4,225,616 arcs. Its one cold graph build took 88.471 s. Adding that observation to median replay timings gives approximately 89.029 versus 88.843 s, only 0.21% saved for build plus this one batch. Those totals are arithmetic combinations, not separately timed whole jobs. Repeated replay on an already available graph has a different cost balance. The historical 42,732-sequence import must not be divided by 9.75 to forecast its runtime.

The unchanged baseline profile supplies a useful ceiling: at k15, incidence construction accounts for a median 5.58% of each case's wall time. Removing **all** of it at zero replacement cost would save a median 16.98 minutes, corresponding to a 1.059× overall speedup. At k5 and k10, the median ceilings are 1.057× and 1.056×. These are conditional ceilings with the same iteration trajectory and all other costs fixed, not observed improvements. The profile uses per-case ratios rather than ratios of medians.

## Correctness findings

Replay passed **1,980 sequence-case comparisons**, checking exact complete records and action traces, including rejected sequences, tariff boundaries, tight SOC and alternative stations. It changes only which outgoing arcs the existing dynamic program scans. The capped historical pool importer and complete CG driver were not exercised.

Every paired master replay had identical LP objectives, variables, duals and artificials in the tested runs. Each solve passed independent restricted-pool reduced-cost, dual-sign and primal–dual-gap checks. Invalid routes remain rejected and cheaper same-incidence replacements work. No global pricing certificate is inferred from these fixed pools.

Capacity selection passed **3,600 per-arc comparisons**, with identical actions and zero returned-cost difference, plus boundary, configuration-change, deadline and tiny-path checks. However, an independent check uncovered an existing source bug: **event tariff-cost reconstruction uses the default charging power when the station has a different power**.

With a 60-kW station and a 240-kW default, a 60-kWh charge over minutes [61,121] is correctly priced across an hourly tariff change by the pricer, but the returned cost record allocates its energy as if charging finished at minute 76. The reconstructed cost is too low by 0.12; a second interval gives 4.8. The record claims expanded-grid cost, so this is not the intentional expanded/continuous distinction.

Both variable-tariff/60-kW reproductions fail the capacity runner's independent reduced-cost guard. The six flat-price or uniform 240-kW control cases pass. This establishes a source defect within the tested heterogeneous-power/variable-tariff scope; it does not establish that a registered production run encountered it. The prototype reproduces the defect because it preserves source behavior. **Capacity acceleration therefore remains blocked for production adoption until cost reconstruction is repaired and independently checked.**

The source-only [eight-case reproducer](capacity/reproduce_record_mismatch.py), [exact artifacts](capacity/record_mismatch.json) and [independent Astra audit](profile/AUDIT.md) preserve this finding. The research owner has been informed; no existing result was automatically invalidated.

## What to implement next

1. Fix station-specific power throughout event tariff-block construction, realized-energy allocation and block validation. Preserve each stop's true energy/time/power identity and test source/master reduced-cost agreement across tariff boundaries. Keep expanded-grid and continuous costs separate. The failing fixtures should become regression tests; this package does not contain that production fix.
2. Integrate omitted incidence construction for Gurobi, preserving SciPy paths, telemetry and validation in ordinary/final/diversification solves. It is a small, well-supported opportunity with a limited full-runtime ceiling.
3. Integrate indexed replay with cache/graph identity guards and full bounded-import comparisons. Benchmark the actual selected warm pool, including graph load, index preparation and validation.
4. Then integrate the capacity prefix/memo kernel, measure actual unique-key counts and memory on a saved hard-case dual, and run paired full-CG cases under identical budgets.

The literature-driven changes—fleet-first bounds, smoothing, sparse pricing, richer column generation and diving—remain proposed experiments. This follow-up did not implement or quantify them. For the conference schedule, advance the small verified changes and the accounting fix before attempting a broad algorithm rewrite.

## Evidence and reproduction

- [Replay prototype, full case table and scope](replay/REPORT.md); [raw replay results](replay/results.json)
- [Master prototype and paired solve results](master/REPORT.md); [raw master results](master/results.json)
- [Capacity timings and readiness blocker](capacity/REPORT.md); [independent validation](capacity/extra_validation.json)
- [Historical bottleneck shares and ceilings](profile/REPORT.md)
- [Independent prototype audit](profile/AUDIT.md)
- [Original work orders](../algorithm_review_20260912/WORK_ORDERS.md)

Baseline source: a29992196acb74d02b8c7891be4061718889999f. Capacity source: 253588e9b22d68fcbc67cb56bc3eb30cbb0e16b6. Scripts are isolated harnesses importing those source snapshots; source hashes and all generated artifacts are checked separately. Timing sections use a shared local lock and single-thread solver/BLAS settings. They do not eliminate all operating-system background load. Reproduction commands are in each subreport; reruns overwrite only their isolated result files.
