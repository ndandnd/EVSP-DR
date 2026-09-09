# Partille GIRO duty recovery gate — 2026-09-09

The first saved result at commit `369d3878` used conservative full-window
charger holding. The current v2 rerun uses early disconnect followed by the
documented idle draw. Both policies recover 42/42 fixed duties and 42/42
unrestricted duty trip sets. The capacity columns below report the v2 policy.

This experiment asks whether the current route-pricing representation can
express the supplied Partille duties after adding the documented battery and
charging physics. It is a deliberately bounded academic model gate, not a
claim that every GIRO operating rule has been reproduced.

## Result

All 42 literal Partille duty variants pass both checks:

| Check | Result |
|---|---:|
| Fixed GIRO trip sequence has a feasible single-vehicle path | 42/42 |
| Unrestricted maximum-cardinality pricing DP recovers every trip | 42/42 |
| Unrestricted route has the exact GIRO trip order | 42/42 |

The unrestricted result is stronger than replay alone: the DP receives the
duty's trip set but is not forced to follow the GIRO sequence. It maximizes the
number of recovered trips under the audit transition model. Because service
times impose the same chronology, every full recovery also reproduces the
GIRO order.

## Implemented physics

- 239.01 kWh usable energy for `18E2` (`133*`) and 236.44 kWh for `18E1`
  (`134*`), calibrated from the source SOC and energy records;
- a 15% physical SOC floor at every modeled movement boundary;
- initial SOC of 100%, with no hard 65% terminal requirement;
- 60 kW charging at `PARX`;
- the documented SOC-dependent opportunity-charging curve, including the
  271 kW 70–80% exception at `3127L` for `18E2`;
- vehicle-group-specific charging sites, 45-second opportunity setup for
  `18E2`, the three-minute minimum recharge duration, and 0.10 kW idle draw.

The curve integrator has a regression test using the source `13301` recharge
at `7880C`: 66.367096% to 83.250336% requires 10.249998 minutes of power plus
the documented 0.75-minute setup, matching the 11-minute recorded activity.

The 15% floor is represented in physical kWh. Equivalently, a zero-based
formulation can expose only the 85% energy above reserve, but the charging
curve must still be evaluated at physical SOC `(state + 0.15 C) / C`.

## Explicit model boundary

The transition gate uses the repository's static, symmetric reference-place
deadheads.
It removes the prior 57-minute trip-link and 61-minute trip-to-charge pruning
limits, but it still does not implement directed time-dependent source DHD
intervals. It also omits shared charger counts, `JON_A`/`2190L` platform
blocking, `4808` FIFO movement, complete interlining preferences, and crew
rules. A 42/42 result therefore establishes single-vehicle representability,
not full operational feasibility.

For the feasibility gate, charging starts after arrival and setup and the bus
disconnects after reaching full SOC or after the minimum recharge duration.
The documented idle draw is deducted between disconnect and the downstream
departure. This fixed early-charge policy is a restriction of the experiment;
later tariff work should optimize charging placement inside each feasible gap.

## Bounded k=2 probe

Four deterministic same-vehicle-group pairs were tested: the shortest and
longest pairs for each Partille vehicle type. Weekday variants of the same
base duty are never paired. The probe peels one maximum-cardinality pricing
route, then prices the exact residual once.

| Cell | Duties | Trips | Greedy two-route recovery | Route sizes | Trips left | Known-duty ports fit | Greedy ports fit |
|---|---|---:|---|---|---:|---|---|
| `e2_short` | 13323 + 13311 | 22 | yes | 15 + 7 | 0 | yes | yes |
| `e2_long` | 13303 + 13302 | 104 | no | 59 + 41 | 4 | yes | yes |
| `e1_short` | 13408 + 13401 | 23 | yes | 12 + 11 | 0 | no | no |
| `e1_long` | 13409 + 13407 | 34 | no | 17 + 16 | 1 | yes | no |

Port checks use conservative half-open occupancy from setup start through
disconnect and the documented counts. They do not check platform blocking or
FIFO. The failures are not impossibility proofs: this is one greedy peel, not
full column generation or branch-and-price. They are useful next-CG targets.

## Reproduction

```bash
python3 src/audit_giro_duty_recovery.py \
  --out analysis/giro_recovery_20260909/k1_results.json \
  --pair-out analysis/giro_recovery_20260909/k2_plan.json \
  --input-dir analysis/giro_recovery_20260909/inputs
```

The 42 `k1` and four `k2` input CSVs are persisted under `inputs/`; result rows
record their SHA-256 identities. No cluster job was submitted by this audit.
