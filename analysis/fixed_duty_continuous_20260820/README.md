# Fixed-duty continuous charging results — 2026-08-20

Executed locally only. No cluster jobs were submitted.

## Scope and conditions

- Fixed ordered GIRO trip sequences; only charging timing, energy, and modeled
  station interpositions vary. This is the fixed-sequence tier, not joint
  re-routing.
- Production physics: 240 kWh battery, 240 kW charger, zero reserve, full
  initial SOC, $5 per physical charge start, unlimited chargers, and
  zone-centroid deadhead.
- Reported runs use terminal policy `>= reserve`. The implementation also
  supports `free`, `>= start`, and priced terminal energy; this experiment does
  not resolve the policy choice.
- `grid_cost` is the independently reconstructed 10 kWh/10-minute lattice
  under the same 240/240 physics. A blank means that lattice has no path; it
  does not mean the continuous duty is infeasible. Costs include the $100,000
  fixed bus cost, so `saving_%` in `duty_results.csv` is a full-objective
  grid-relaxation percentage, not the timing-only percentage below.
- `flat` is the tracked flat tariff. `peak12` is a synthetic time-varying
  tariff and is a software demonstration, not a research result.

Real charging tapers above about 80% SOC. Constant power to 100% is mildly
optimistic, although it is the standard linear idealization in the E-VSP
literature.

Charger capacity is unlimited. Energy cost and load/concurrency are reported,
but no peak-shaving value is claimed. Across the nine duty unions, the largest
observed global concurrency was 3 buses (720 kW) under `flat` and 4 buses
(960 kW) under synthetic `peak12`; the largest same-station concurrency was 2.

## Validation gates

1. **G1 — pass.** On real duty 13412 (`k05_s1`) under the flat tariff, an
   independently implemented lattice-flow MILP returned exactly `100058.152`,
   equal to the existing dynamic program at 300 kWh/300 kW, 15 kWh SOC steps,
   and 10-minute blocks (difference `0.0`).
2. **G2 — pass.** On duty 13412 at the legacy physics, continuous cost was
   `100051.4202876032 <= 100058.152` under `flat` and
   `100047.76521847615 <= 100068.89508938692` under `peak12`. Across the frozen
   240/240 experiment, the continuous objective was also lower for all
   lattice-representable rows. Only 2 of 30 cell-duty occurrences were
   representable on the frozen 10 kWh/10-minute lattice.
3. **G3 — pass.** Duty 13411 is feasible at the frozen 240/240 physics:
   `100091.228831653` under `flat` and `100111.125192852` under `peak12`, with
   exact replay in both cases. It remains unrepresentable on the tested grid.
4. **G4 — pass.** All 60 duty/tariff schedules (30 duty occurrences × 2
   tariffs) replayed within battery capacity and reserve and reproduced the
   objective to absolute tolerance `1e-6`.
5. **G5 — pass.** Under `flat`, optimized timing and charge-on-arrival had
   aggregate charging costs `1650.7509150681087` and
   `1650.7509150684782`; the numerical difference is below `1e-6`. Delayed
   starts selected under the flat tariff are cost-degenerate ties, not an
   advantage.

Tests executed after the final model hardening:

```text
13 passed in 7.78s
51 passed, 4 subtests passed in 66.34s
512 passed, 123 subtests passed in 162.15s (full repository suite)
```

## Goal-2 answers

Under the synthetic `peak12` demonstration, allowing optimized delayed starts
instead of constraining the same fixed-sequence model to charge on arrival
reduced charging cost across the 15 distinct duties in the three k5 cells from
`1095.2946003174968` to `1088.7563748126413`, a saving of
`6.538225504855518` model-currency units or **0.5969376187%**. The three k5
cell percentages were 1.1757771%, 0.2776823%, and 0.1413158%. This is a
fixed-sequence, timing-only software demonstration conditional on `peak12`,
`>= reserve`, and the frozen physics. It is not re-routing value, and it is not
the observed-to-fixed research estimate because the comparator here is the
model's charge-on-arrival restriction rather than GIRO's recorded charging
windows.

The continuous model admits duty 13411 under the stricter 240 kWh battery even
though the recorded evidence found it unrepresentable on all five tested
lattices at 300 kWh. This shows that the 15 kWh/10-minute grid can exclude a
physically feasible fixed duty through transition alignment and SOC flooring;
grid infeasibility is therefore not evidence of physical infeasibility.

## Files

- `duty_results.csv`: requested per-duty/tariff table.
- `union_concurrency.csv`: simultaneous charging measured after independently
  optimizing every duty in each union.
