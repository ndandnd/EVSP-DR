# Provisional findings: cursor/arcflow-oracle-2969-3a99

These labels are branch-local. They are not authoritative bug or decision IDs.

## LOCAL-1

- Claim: The primary-grid discretized integer fleet is three buses for all
  three k2 replicates. The RAW pool-MIP excess is 1, 1, and 4 buses.
- Evidence path: `analysis/arcflow_oracle_20260820/REPORT.md`
- Producing commit SHA: `7641191a03a1ba908fdd6e56996f4c3edfe440cd`

## LOCAL-2

- Claim: On the fully expanded acyclic network, the arc-flow fleet LP equals
  the set-partitioning fleet LP on all nine primary cells; path-flow
  decomposition explains the equality.
- Evidence path: `analysis/arcflow_oracle_20260820/results.csv`
- Producing commit SHA: `7641191a03a1ba908fdd6e56996f4c3edfe440cd`

## LOCAL-3

- Claim: A service-integrality search result is accepted as an integer witness
  only when every returned arc is integral and physical route replay passes.
  Matching such a witness to a certified fleet lower bound proves the full
  all-arc-integer fleet optimum without treating the relaxation as exact.
- Evidence path: `analysis/arcflow_oracle_20260820/REPORT.md`
- Producing commit SHA: `3b0d485845adca6d69c9b7452a06b2fa0d8a2612`

## LOCAL-4

- Claim: Dual simplex did not return a k02_s1 arc-flow LP primal in 600
  seconds, while HiGHS interior point solved the same LP in 16.188 seconds.
- Evidence path: `analysis/arcflow_oracle_20260820/REPORT.md`
- Producing commit SHA: `4ac83d058b2d4d8befd2df4d9e86a6b0326bd33d`

## LOCAL-5

- Claim: The first direct all-arc-integer k2 solve exhausted the 47 GiB local
  VM; subsequent runs used a 30 GB process cap and the explicitly labelled
  witness-search relaxation described in LOCAL-3.
- Evidence path: `analysis/arcflow_oracle_20260820/REPORT.md`
- Producing commit SHA: `b09284f313cb409e7d370284d491f681ca0eaff6`

## LOCAL-6

- Claim: Explicit 300 kWh / 300 kW / reserve-0 CLI flags preserve the k02_s2
  legacy regression: fleet LP 2.1875, integer fleet 3, and deterministic
  scientific LP/MIP payloads byte-identical to the prior artifacts.
- Evidence path:
  `analysis/arcflow_oracle_20260820/PHYSICS_FLAGS_REGRESSION.md`
- Producing commit SHA: `2715b90b78f343b16d705031f47e4e2543cb463f`

## LOCAL-7

- Claim: At 240 kWh / 240 kW / reserve 0 with a 10 kWh / 10 minute
  commensurate grid, k02_s2 has fleet LP 2.4 and a fully integral,
  physically replayed three-bus witness; its discretized integer fleet
  optimum is therefore 3.
- Evidence path:
  `analysis/arcflow_oracle_20260820/PHYSICS_FLAGS_REGRESSION.md`
- Producing commit SHA: `2715b90b78f343b16d705031f47e4e2543cb463f`

## LOCAL-8

- Claim: On the public Utrecht qlink 8 timetable converted to EVSP-DR's stated
  160 kWh, constant-charge, 20 kWh / 8 minute model, the fleet LP is
  10.2272727273 and the proven integer fleet is 11. The published best solution
  uses 10 buses, a one-bus comparison gap, but objective scores are not
  comparable because charging and battery-degradation assumptions differ.
- Evidence path: `analysis/public_benchmark_evsp_20260821/README.md`
- Producing commit SHA: `7690f3e94add89da450afb8cc6c1bee77363ca3e`

## LOCAL-9

- Claim: The proprietary-data-free synthetic instance at seed 20260821 has
  exact fleet 2 on a 5 kWh / 5 minute grid and exact fleet 3 on a
  5 kWh / 10 minute grid, reproducing a one-bus time-discretization penalty.
- Evidence path: `analysis/public_synthetic_evsp_20260821/README.md`
- Producing commit SHA: `e2e99548d406b1ea3f19fd6d91b88607556b9559`

## LOCAL-10

- Claim: On the same public synthetic fine-grid model, an explicitly
  pair-limited pool is pool-optimal at 3 buses while the full model is
  2 buses, reproducing a controlled one-bus pool-composition gap. This is not
  labelled as a naturally generated RAW CG pool.
- Evidence path: `analysis/public_synthetic_evsp_20260821/README.md`
- Producing commit SHA: `e2e99548d406b1ea3f19fd6d91b88607556b9559`
