# Capacity and PARX-speed pilot implementation

## Outcome

The pilot is implemented on branch `codex/capacity-speed-pilot-20260910`.
The executable covering-CG code is commit
`7d38efdd39857438c4a6e30b43b09e973ce51086`; later branch commits contain
submission-tool and manifest-only changes. The immutable cluster checkout is
detached at the executable commit and requires a clean tracked worktree.

The experiment uses four matched arms: 240 kW/no shared capacity; 240 kW plus
documented shared capacity; PARX 60 kW/no shared capacity; and PARX 60 kW plus
documented shared capacity. Every other setting is fixed: 240 kWh battery,
240 kW non-PARX power, full initial SOC, zero reserve, no additional terminal
SOC target, 2.5 kWh SOC grid, five-minute event lattice, flat tariff, and set
covering in both LP and MIP.

## Capacity and pricing implementation

Finite inventories are 2190L=1, 4808=1, 3127L=2, 7880C=1, and JON_A=1.
PARX is unlimited. The capacity LP contains all finite-site one-minute rows
before the first solve. Charging intervals use half-open overlap semantics.
The exact event shortest path subtracts the capacity-row duals as well as trip
duals. Because these duals can change the best connection time, the pricer
evaluates tariff and one-minute capacity breakpoints instead of penalizing only
the tariff-cheapest stored window. Explicit graphs retain one transition per
station when multiple stations reach the same trip/SOC state.

The same capacity rows are imposed in the binary covering MIP. Selected
schedules receive a separate continuous-time sweep by station, and duplicate
trip service is reported separately because covering permits it.

The exact stopping certificate applies only to this conservative event/SOC
graph. A run is labeled epsilon-optimal only when the exact shortest-path
reduced cost is at least `-rc_eps` and LP artificials are zero. The driver
retains its Gurobi LP log, network construction time, and per-iteration LP and
pricing times plus rows, columns, and nonzeros.

## Evidence boundaries

The documented inventory and PARX 60 kW come from
`outputs/meeting_20260910/GIRO_EMAIL_ATTACHMENT_AUDIT.md` and
`outputs/meeting_20260910/GIRO_EMAIL_CONFIRMED_ASSUMPTIONS.md` in the shared
workspace. Prior nonlinear k2/k3 results under
`outputs/meeting_20260910/giro_k23_newphysics` are not duplicates: they use a
different charge-physics model and a 15% SOC floor. This controlled pilot does
not impose a 65% end-of-duty floor; 65% is a recharge target in the source
material. It also excludes the future SOC-dependent opportunity-charge curve.

This is a one-column-per-iteration explicit reference CG, not the production
batched runner. Only comparisons among the four matched arms isolate capacity
and PARX speed. Differences from earlier production results may also reflect
the driver and batching change.

The bounded cohort contains duties 13408 and 13406 plus the aligned E1-short
k2/k3 sets. Duty 13406 was added because its recorded GIRO schedule charges at
PARX; the optimized route is still allowed to avoid PARX. These cohorts do not
guarantee that every finite station is active. Each result therefore records
actual charging stations and capacity use before any interpretation.

## Verification and launch

The targeted suite passes 59 tests, including station-specific event duration
and replay, capacity-dual timing changes, capacity-constrained master behavior,
half-open physical overlap, and covering duplicate-service reporting. A local
duty-13408 smoke run reached zero artificials and an exact terminal reduced
cost within tolerance; its MIP solved optimally and passed physical replay.

The immutable cluster root is
`/home/nc437/ladder-lite/capacity_speed_pilot_20260910_v2_7d38efd`.
CG array `772080` contains 16 tasks on `default_partition`, concurrency two,
24 GB and 90 minutes per task. Dependent MIP array `772082` contains the same
16 cells on `scaglione`, concurrency two, four CPUs, 16 GB and 30 minutes per
task, excluding `scaglione-compute-01`. At the saved snapshot, CG tasks 0 and
1 were running and the remainder were queued behind the array throttle; every
MIP task was waiting on its matched CG dependency.
