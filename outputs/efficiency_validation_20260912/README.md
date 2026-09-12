# Efficiency validation — 12 September 2026

This campaign evaluates three opt-in implementation changes after correctness checks: omit redundant Gurobi incidence matrices, index fixed-sequence successor ranges, and use bounded charging-window prefixes/memoization. The capacity reference and optimized modes both include the station-specific charging-power accounting correction. The full-run speedup is unproven until the paired runs complete.

## Launch status

All nine allocations were submitted and their partition/node exclusions verified. At the initial observation they were pending scheduler priority; no production timing result was available. Baseline execution commit: `89c5ba3e8a66fd77ad8397e5ce63eae647a0a49a`. Capacity execution commit: `309d98d266ebaf6b7e99543a67f8f2be5736874a`. Both source branches are pushed to GitHub.

| Pair | Slurm job |
|---|---|
| d00_g0 | 964194 |
| d00_g1 | 964195 |
| w1_k08 | 964196 |
| w4_k11 | 964197 |
| w6_k12 | 964200 |
| w1_k08_repeat | 964201 |
| cap_k1 | 964202 |
| cap_k2 | 964203 |
| cap_k1_combined_peak12 | 964204 |

## Design and evidence

Nine independent allocations contain eighteen fresh-process treatments. Two fresh decomposition cases isolate the master change. Three previous-k warm cases compare master plus replay changes; a fourth warm allocation repeats w1_k08 in reverse order. Two flat capacity cases and one combined-power synthetic-peak12 case compare the capacity selector. Within each pair, execution commit, input, tariff, model settings and resource allocation match. Alternate treatment order is recorded. All cases are eligible concurrently on the default partition and exclude scaglione-compute-01.

Fresh pairs request 2 CPUs / 32 GB / 4 h 30 min with 7200 s per treatment. Warm pairs request 8 CPUs / 96 GB / 8 h with up to 10800 s separate source-authenticated graph-cache preparation and 7200 s per treatment. Capacity pairs request 1 CPU / 24 GB / 6 h 30 min with 10800 s per treatment. True prior-k parent artifacts are frozen and hashed. No historical job is changed. Each job uses a unique attempt directory, durable solver pools/checkpoints and watchdog records; automatic requeue is disabled for these paired validation allocations to avoid silently restarting only one arm.

`manifest.json` is the frozen experiment contract. `jobs.json` binds every submitted job to the manifest and records effective Slurm exclusions. `audit/` contains the audited source identities and launch justification. `local_validation.json` and `local_smoke/` contain the focused correctness evidence. The new source branches are `codex/baseline-efficiency-20260912` and `codex/capacity-efficiency-20260912`. Execution pins and deployment observations are recorded in `deployment.json`.

## What the charts should show

- Runtime and convergence versus elapsed time: expect a leftward shift if acceleration transfers to full solves; quantify paired end-to-end and phase timings separately. Report cache preparation separately and also show its contribution to time from scratch. Include completed-run coverage and censored cases; a timeout is not a speedup.
- Objective versus iteration and certified endpoints: these implementation changes preserve the intended model. Same certified endpoints are expected within tolerance. Parallel replay completion order can change insertion order and subsequent CG trajectories, so compare input/pool hashes and normalized accepted pools before attributing differences. An uncertified RMP value is not a full-model lower bound.
- Charging prices: the exogenous flat/peak08/peak12/peak18 curves are unchanged. Only the new combined capacity control here uses peak12; this is not a replacement tariff campaign. The pricing subproblem computes reduced costs and should produce equivalent answers under the two selectors.
- Charging costs: the station-power fix can correct how energy spans tariff periods at slower chargers. Historical affected numbers must be recomputed and explicitly superseded. A correction is not an optimization saving. Faster solving could improve a time-limited incumbent or certify a harder case; this campaign does not yet establish either result.

Keep scheduler state, CG certificate, finite-pool MIP proof, physical replay and GIRO target attainment separate. This campaign launches CG only. Generic heterogeneous-power MIP validation remains outside the verified implementation scope. Existing charts remain historical controls until comparable validated results justify new series.

## Collection

The pinned baseline script `scripts/efficiency_validation_20260912/campaign.py collect --root /home/nc437/ladder-lite/efficiency_validation_20260912` emits `evsp-efficiency-collection-v2`. It exposes preparation, live first-arm execution, not-yet-started arms, failures, both solver certificate schemas, input and pool hashes, order and flags. Scheduler status must be collected separately. Live artifact hashes are observations at collection time, not immutable final-artifact declarations. Cache identity is recorded after construction to avoid repeatedly hashing large caches during performance measurements.
