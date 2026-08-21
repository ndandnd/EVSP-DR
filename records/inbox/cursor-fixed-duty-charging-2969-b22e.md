# Provisional findings: cursor/fixed-duty-charging-2969-b22e

These are branch-local findings for curator review. Labels are provisional and
must not be cited as authoritative bug or decision IDs.

## LOCAL-1

- **Claim:** The event-based fixed-sequence optimizer passes G1–G5 locally.
  All 30 k2/k3/k5 duty occurrences are continuously feasible under both tested
  tariffs, and duty 13411 is feasible at the frozen 240 kWh/240 kW physics
  despite its lattice failures.
- **Evidence:** `analysis/fixed_duty_continuous_20260820/README.md`;
  `analysis/fixed_duty_continuous_20260820/duty_results.csv`
- **Producing commit SHA:** `f7fdf811d2ef9eb713e8a00258eb02e54f267e45`

## LOCAL-2

- **Claim:** A zero-energy station waypoint is ambiguous in the legacy
  station-string/charging-stop replay representation when a later visit uses
  the same station. Gap-indexed station visits and independent replay avoid
  greedily consuming the later event.
- **Evidence:** `src/fixed_duty_continuous_optimizer.py`;
  `tests/test_fixed_duty_continuous_optimizer.py`
- **Producing commit SHA:** `2f8b9b3cdcac730850bf2aee825e7171e429b03b`

## LOCAL-3

- **Claim:** The mean-preserving SE3-shaped two-peak curve is a synthetic
  software instrument, not observed data. Its first 24 hours average 0.0992;
  it has an overnight trough, broad morning peak, and stronger evening peak.
- **Evidence:**
  `analysis/fixed_duty_continuous_20260820/TWO_PEAK_INSTRUMENT.md`;
  `analysis/fixed_duty_continuous_20260820/two_peak_se3_synthetic_h26.csv`
- **Producing commit SHA:** `70eaf3641608fc0dc1d4444b222f7ebd0f3d2f54`

## LOCAL-4

- **Claim:** The continuous optimizer can impose a hard maximum number of
  physical charge events while retaining exact schedule replay. This is a
  diagnostic control and does not change the frozen production defaults.
- **Evidence:** `src/fixed_duty_continuous_optimizer.py`;
  `tests/test_fixed_duty_continuous_optimizer.py`
- **Producing commit SHA:** `cfb597962856bb2adb44be6e0e925a2b9a120221`

## LOCAL-5

- **Claim:** On matched feasible sets, the three relevant aggregate quantities
  are distinct. Under two-peak/`>= reserve`, uncapped timing value is 27.849
  (5.328%), capped timing value is 28.834 (5.447%), and cap cost is 6.648.
  The single aggregate prefix-causal timing bound is 40.225 (7.695%), so the
  uncapped optimizer achieves 69.2% of the bound. All capped arrival-priority
  baselines are solved directly under the event cap and pass physical replay;
  no events are deleted post hoc. Under `>= initial`, 3 of 15 duties are
  infeasible under the cap and are excluded from both matched arms.
- **Evidence:**
  `analysis/fixed_duty_continuous_20260820/FACTORIAL_DIAGNOSTIC.md`;
  `analysis/fixed_duty_continuous_20260820/factorial_summary.csv`;
  `analysis/fixed_duty_continuous_20260820/prefix_causal_bound_summary.csv`
- **Producing commit SHA:** `70c88f474e21058068112d3aad92baf9a58bbddb`
