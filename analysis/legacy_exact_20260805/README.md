# Legacy and Exact-Pricer Comparison (2026-08-05)

This folder preserves a compact, reviewable record of the comparison between
the April 2026 legacy DP runs and the completed August 2026 exact-pricer runs.
It intentionally contains only derived analysis artifacts, not raw column
pools, checkpoints, Slurm logs, or solution journals.

## Contents

| File | Purpose |
| --- | --- |
| `practice_30_bus_g_300_charging_gantt.png` | Reconstructed charging schedule from the legacy 30-bus run. |
| `practice_43_bus_g_300_charging_gantt.png` | Reconstructed legacy 43-bus charging schedule, explicitly marked with its dummy-trip warning. |
| `charging_events.csv` | Long-form charging event data used by the two Gantt plots. |
| `legacy_vs_exact_convergence.png` | Log-scale restricted-master convergence comparison. |
| `timing_and_endpoint_summary.csv` | Endpoint and timing summary for three legacy traces and all 20 exact-pricer runs. |

## Findings and caveats

- No historical Gantt image was found in Git. The two images here were
  reconstructed from the preserved legacy solution files and matching route
  pools.
- The legacy 30-bus run selected 26 real routes and reached a 1.925% final
  MIP gap.
- The legacy 43-bus run selected 25 real routes plus two Big-M dummy trip
  cover variables (`q_6` and `q_147`). Its nominal objective must therefore
  not be interpreted as a fully real EVSP schedule.
- A legacy 20-bus solution file exists, but its matching saved route pool was
  not retained, so no reliable Gantt was generated for it.
- The convergence plot compares shapes only. The legacy and exact-pricer runs
  use different instances and model versions, so their objective levels are
  not directly comparable.
- The legacy CSVs report separate master and pricing times. The exact-run logs
  report total wall time, not a comparable master/pricing breakdown.
- The exact strict-partition MIPs were infeasible. The 46/48/58-bus results in
  the result release are cover-relaxation incumbents with trip overcoverage;
  they are useful diagnostics, not final schedules.

## Provenance and reproduction

The legacy source material remains local in:

- `src/Practice_20bus_g300_20260429_015514/`
- `src/Practice_30bus_g300_20260429_015514/`
- `src/Practice_43bus_g300_20260429_015514/`

The exact-run input is the GitHub release archive
`results-exact_big_mip_20260805T135556Z` on the `peel-and-price` branch. It is
kept as a release asset rather than committed to Git because it contains raw
experiment data.

Regenerate the artifacts after downloading that archive:

```bash
MPLCONFIGDIR=/private/tmp/mplconfig python3 src/plot_legacy_practice_gantt.py
MPLCONFIGDIR=/private/tmp/mplconfig python3 src/compare_legacy_vs_exact_pricing.py \
  --exact-archive /path/to/exact_big_mip_20260805T135556Z.tar.gz
```

The reusable generators are:

- `src/plot_legacy_practice_gantt.py`
- `src/compare_legacy_vs_exact_pricing.py`
