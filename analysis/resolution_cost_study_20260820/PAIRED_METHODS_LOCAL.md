# Paired exact-CG versus direct arc-flow study

Date: 2026-08-20. All runs were local. No cluster jobs were submitted.

## Arms and proof scope

Every frozen scientific cell now has two method arms:

1. exact CG followed by a license-free binary integer master over its final
   column pool;
2. direct arc-flow LP and direct all-arc-integer MIP over the same imported
   SOC-by-time network.

The long output has one row per method arm. Both rows share a
`paired_cell_key`, instance hash, physics, and grid. LP equality is checked only
when both LP solves are optimal; a censored LP is never called unequal.

The local paired subset is selection 2 at k2/k3, both physics profiles, all
seven grids: 28 scientific pairs / 56 arm rows. Each solve had a hard
60-second parent-process budget. Existing exact-CG statuses were reused;
restricted pool MIPs and arc-flow runs were executed for this comparison.

## Local outcomes

| metric | exact CG / pool MIP | direct arc-flow |
|---|---:|---:|
| arm statuses | 28 | 28 |
| LP certificates / optimal LPs | 18 | 4 |
| proven integer fleets | 22 | 0 |
| wall-limit rows | 10 CG; pool MIP outcomes separate | 28 |
| process/memory failures | 0 | 0 |

Arc-flow's internal HiGHS time limit did not cap native model loading and
presolve. The parent runner now enforces budget+10 seconds and preserves the
already-published DAG and sparse-model size. One observed cell had
4,325,938 variables and 9,829,588 nonzeros before the LP call.

## Question 1: crossover

Within the 60-second local budget, direct all-arc-integer arc-flow is already
intractable at **k2 on the coarsest grids**, while CG certifies:

- 18 paired cells had a certified CG LP but no arc-flow integer proof;
- this includes k2 selection 2 at 15/10 and 10/10 under both 240/240 and
  300/300, plus finer k2 grids and the k3 anchors;
- no cell had a proven direct arc-flow integer result while CG was censored.

This is a budget-specific crossover, not a claim that arc-flow can never solve
k2. The independent longer oracle campaign found fully integral witnesses and
proved primary-grid fleets using certified LP lower bounds, but its requested
all-arc-integer solve exhausted a 47-GiB VM and its practical witness search used
service-arc integrality. The paired study deliberately measures the strict
all-arc formulation.

The full k2-k20 cluster plan is needed to place the production-budget
crossover, but the local direction is unambiguous: CG remains the tractable LP
method after direct arc-flow model/presolve cost becomes limiting.

## Question 2: LP equality

Four local paired LPs completed:

| profile/cell/grid | CG fleet LP | arc-flow fleet LP | difference |
|---|---:|---:|---:|
| 240 / k02_s2 / 15/10 | 2.333333333333 | 2.333333333333 | -4.4e-16 |
| 240 / k02_s2 / 10/10 | 2.400000000000 | 2.400000000000 | +4.4e-16 |
| 300 / k02_s2 / 15/10 | 2.187500000000 | 2.187500000000 | 0 |
| 300 / k02_s2 / 10/10 | 2.200000000000 | 2.200000000000 | 0 |

LP equality holds in every locally checked pair. The independent oracle also
found equality on all nine primary k2/k3/k5 cells, and the authoritative
constraint audit found all 36 route conditions network- or objective-encoded.
There is currently **no LP break locating a missing per-route constraint**.

## Question 3: k40 affordability

Target: 947 trips, 240/240, 1 kWh / 5 minutes.

| method | extrapolated size / time | affordable in 24 h? |
|---|---|---|
| exact CG | 497,780 fitted DAG nodes; 679,381 structural upper count; 1,969 certificate-hours | **No** |
| direct arc-flow | 457,204,079 variables; 2,580,521 constraints; 964,269,319 nonzeros; at least 26.1 GiB just for core sparse/arc arrays before solver copies | **No** |

The arc-flow variable model is
`trips^1.488 · (1/soc)^0.903 · (1/block)^1.049` (R² 0.99915).
The nonzero model has trip exponent 1.461 and comparable resolution exponents.
The target has about 50.9 times more arc variables than the largest local
training model. No local strict all-arc-integer run proved an integer fleet, so
an arc-flow integer-wall model is intentionally not fabricated.

Neither uniform exact-CG refinement nor direct all-arc MIP is affordable at
k40 under this local sanity evidence. That is the quantitative motivation for
an event-based representation.

## Cautions

- CG wall training spans only 29-71 trips and SOC steps 2.5-15 kWh; k40/1 kWh
  is a large extrapolation.
- Arc-flow size training uses only selection 2 at k2/k3. Peak RSS in censored
  rows is a pre-solver checkpoint, not the eventual HiGHS peak.
- Certified-only wall fitting is selected by tractability.
- The cluster k5-k20 rows remain necessary for production calibration.

## Artifacts

- `paired_local_k2_k3_s2/resolution_cost_long.csv`
  SHA256 `3d99073f6414c824610c2c8ef9b18de024920cb2b1dad94f099baa4bd75dca87`
- `paired_local_k2_k3_s2/resolution_cost_extrapolation.json`
  SHA256 `9bb8ca2b2e794333a95ab75b81fcd912b47584222a7b152d80824d339d2a95c5`
