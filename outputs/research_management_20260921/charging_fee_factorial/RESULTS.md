# Controlled fee results

All 18 cells have validated five-bus witnesses serving the 62 trips exactly once; all nine fee-pair problem hashes match. 12 solver searches reached the configured 0.01% MIP gap; 6 stopped at 600 seconds. One raw witness required the documented numerical repair below. All original solver statuses, lower bounds and revised witness gaps are retained in cell_results.csv.

| Fixed trip assignment | Tariff peak | Starts, fee 0→5 | Electricity cost, fee 0→5 | Electricity increase | Saving at common fee 5 | Fee-5 gap |
|---|---:|---:|---:|---:|---:|---|
| Original | 08:00 | 46→36 | 174.524→189.636 | +15.112 | 34.888 | 0.000023% |
| Original | 12:00 | 48→34 | 243.829→268.192 | +24.363 | 45.637 | 0.000000% |
| Original | 18:00 | 51→32 | 159.378→189.575 | +30.196 | 64.804 | 1.8500% |
| Saved fee-0-derived | 08:00 | 42→34 | 158.706→174.488 | +15.782 | 24.218 | 0.7521% |
| Saved fee-0-derived | 12:00 | 47→35 | 234.295→255.642 | +21.348 | 38.652 | 2.9679% |
| Saved fee-0-derived | 18:00 | 50→32 | 163.149→199.573 | +36.424 | 53.576 | 1.4179% |
| Saved fee-5-derived | 08:00 | 43→30 | 160.962→183.715 | +22.753 | 42.247 | 2.0163% |
| Saved fee-5-derived | 12:00 | 46→27 | 233.839→262.469 | +28.630 | 66.370 | 0.0031% |
| Saved fee-5-derived | 18:00 | 48→26 | 146.330→183.062 | +36.732 | 73.268 | 3.8316% |

Observed starts fell in 9/9 paired incumbents. Lower and upper objective bounds establish strict separation of optimal start counts in 3/9 restricted models, with the 0.01 synthetic-unit numerical margin documented in README.md. This does not certify the displayed counts as unique.

Electricity cost E sums hourly tariff × charged kWh. The objective is F = E + fee × charging starts. “Saving at common fee 5” compares the two schedules under the same fee: (E0 + 5N0) − (E5 + 5N5). The increased electricity bill is reported separately. There is no fleet-cost term or currency conversion. Model objective gaps are not finite route-pool fleet gaps. The single-charge-per-gap, single-tariff-hour restriction and recovered station paths remain fixed.

For original/08:00/fee 5, raw replay missed one terminal floor by 0.00002816 kWh. A separate witness extends one existing charge by 6 milliseconds within its visit, tariff hour and available capacity, then recomputes downstream SOC, taper energy and costs. It passes the unchanged validator. Its feasible objective is 369.636334162, against the unchanged solver lower bound 369.636248039 (0.00002330% gap). The raw failed attempt remains untouched; this repaired witness is an upper bound, not a new solver optimality certificate. [Repair details](postprocess_repair/original_peak08_fee5/repair_report.json).

| Fixed trip assignment | Tariff peak | Fee-0 optimal starts, lower bound | Fee-5 optimal starts, upper bound | Strict reduction established |
|---|---:|---:|---:|---|
| Original | 08:00 | 40 | 39 | Yes |
| Original | 12:00 | 39 | 38 | Yes |
| Original | 18:00 | 37 | 38 | No |
| Saved fee-0-derived | 08:00 | 37 | 37 | No |
| Saved fee-0-derived | 12:00 | 37 | 39 | No |
| Saved fee-0-derived | 18:00 | 39 | 39 | No |
| Saved fee-5-derived | 08:00 | 34 | 34 | No |
| Saved fee-5-derived | 12:00 | 33 | 32 | Yes |
| Saved fee-5-derived | 18:00 | 31 | 33 | No |

[Editable paired table](paired_fee_results.csv) · [All endpoints and log locations](cell_results.csv) · [Figure caption](figure_caption.txt) · [Gurobi endpoint excerpts](log_excerpts.md)
