# RAW recovery and instance features

Analysis status: `not_estimable_no_auditable_raw_results`.

This audit is descriptive. Duty-union instances overlap and repeated cells share instances, so rows are not independent observations. No association is interpreted causally.

## Feature definitions

- `deadhead_density`: direct trip-to-trip deadhead edges divided by all temporally ordered trip pairs.
- `deadhead_energy_fraction`: direct known-duty deadhead kWh divided by service plus direct deadhead kWh.
- layover slack: calendar gap minus direct deadhead minutes for consecutive trips in the GIRO duty; missing direct legs are counted and excluded from the distribution.
- `station_reachability_fraction`: time-feasible trip-pair station bridges divided by all temporally ordered trip pairs; `station_only_bridge_fraction` excludes pairs already connected by a direct deadhead arc.
- grid fractions: fraction of GIRO duties feasible in the frozen fixed-duty expanded optimizer at each named 300/300 grid.

Distribution summaries across the 40 duty-union instances are in `feature_distribution.csv`.

## Recovery association unavailable

No auditable normalized RAW integer-result rows are tracked in this checkout. `records/RESULTS_LOG.csv` is header-only and no `mip_run_summary.csv` exists on any available git ref. Therefore no feature/recovery correlation or threshold claim is reported. Supply normalized RAW rows via `--raw-results`; prose and LP route weights are intentionally not recoded as integer recovery.
