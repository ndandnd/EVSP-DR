# GIRO zero charge-start fee audit

This note covers the separate five-duty, 62-trip, 240 kWh / 350 kW cohort. It does not describe the baseline-chain fee campaign in this directory, whose manifest uses 240 kW and flat prices.

## Existing evidence

The authoritative local table is `outputs/meeting_20260910/presentation_manager/charging_audit/charging_three_baselines.csv` (header and rows 2–10). The source hashes bind one instance (`b386f8a16958d25c857297ac4643bf6c73ae2114557c585446725cbd51c8b64d`) and tariffs:

| tariff | original starts | original energy cost interval | original charged kWh | fixed starts | joint starts |
|---|---:|---:|---:|---:|---:|
| peak08 | 52 | 230.2870096263–230.9810746874 | 1891.1133098 | 22 | 18 |
| peak12 | 52 | 289.5935355366–290.5965839308 | 1891.1133098 | 28 | 23 |
| peak18 | 52 | 223.4472905823–223.7232277427 | 1891.1133098 | 22 | 15 |

The existing optimized counts are fee-5 results: the raw records show modeled start fees of 110/140/110 for fixed duties and 90/115/75 for joint routes. Their terminal surplus is only 10.7900–13.1000 kWh (fixed peak12 is 7.7100 kWh), while original GIRO is 280.7833253 kWh. They are useful count evidence, but they are not an equal-terminal comparison. See `presentation_manager/charging_audit/README.md` lines 1–17 and `charging_shift/README.md` lines 18–26.

The original within-window power trace is unavailable. Therefore the energy cost is an interval; the uniform-power value is an explicitly unobserved assumption. Do not interpret the lower optimized energy as GIRO inefficiency.

## Fee-zero check

`src/compare_original_giro_charging.py` already accepts a nonnegative `--charge-start-cost` (lines 210–214 and 335–350). Its fee is applied once per recorded event (lines 267–288). Re-running the same instance and tariff hashes at fee 0 gives:

| tariff | starts | fee | total charging cost interval |
|---|---:|---:|---:|
| peak08 | 52 | 0 | 230.2870096263–230.9810746874 |
| peak12 | 52 | 0 | 289.5935355366–290.5965839308 |
| peak18 | 52 | 0 | 223.4472905823–223.7232277427 |

The charge events, total charged energy, terminal aggregate, and uncertainty interval are unchanged; only the modeled fee is removed.

## Launch contract

The isolated wrapper is on branch `codex/giro-zero-start-fee-20260913`, based at corrected commit `5cdb8138c29faef9d5bf949175cb1e815a0b4220`:

* `scripts/event_uniform_envelope/giro_zero_fee_campaign.py` creates fee-0 copies of the saved fee-5 source roots, preserving each original route cost as `source_cost` and recording source/destination fee, start count, and source hashes. It runs fresh fixed frontiers with a runtime fee override, then reprices the sibling frontier into the target fee and forms a common fee-normalized union before each joint MIP.
* `scripts/event_uniform_envelope/launch_giro_zero_fee.py` prints six frontier commands and six joint-MIP commands. Each joint command depends on both fee frontiers for its tariff. The frontier resources are default partition, 1 CPU, 24G, 2 hours; MIPs are Scaglione, 8 CPUs, 48G, 2 hours. `scaglione-compute-01` is excluded from every job; MIPs also exclude `scaglione-cpu-04`.
* `worker.sh` sets the shared Gurobi license and calls the wrapper. No jobs were submitted by this audit.

The common terminal row remains `sum(expanded_grid_terminal_soc_kwh * x) >= 280.7833253`, and the runner's continuous aggregate replay must also meet that target. The resulting claim is a finite saved-pool plus common-frontier-union comparison; changing the fee does not transfer a full-model pricing certificate.

Focused wrapper tests are in `tests/test_giro_zero_fee_campaign.py` and cover source immutability, fee repricing, original accounting repricing, and common-union normalization.
