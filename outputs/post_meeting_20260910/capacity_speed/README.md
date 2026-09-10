# Charger-capacity × PARX-speed pilot

This is a controlled 2×2 pilot on the exact conservative event/SOC graph.
Every arm uses a 240 kWh battery, starts at 240 kWh, uses a 0 kWh reserve,
and has no terminal target beyond the same reserve. Non-PARX stations remain
at 240 kW. The speed factor changes only PARX from 240 kW to the documented
60 kW. The capacity factor adds the documented finite inventories 2190L=1,
4808=1, 3127L=2, 7880C=1, JON_A=1; PARX remains unlimited.

All four arms use set covering (`>= 1`) in both CG and the final MIP. The MIP
output audits duplicate trip service separately. The capacity arms put every
one-minute finite-site row in the restricted LP
before pricing, pass its dual into the event shortest path, and impose the
same rows in the pool MIP. The exact stopping test is therefore a full LP
certificate for this conservative discretized graph when the run reports
`certified_rc_optimal=true`, zero artificials, and a terminal minimum reduced
cost at least `-rc_eps`. This is an epsilon-optimal certificate, not a literal
claim that a floating-point value is nonnegative. It is not a certificate for continuous SOC or the
separate nonlinear GIRO charge curve.

The selected MIP routes are independently swept as half-open continuous
connection intervals. Existing nonlinear new-physics results use a 15% SOC
floor and model recharge targets; they are separate evidence. A 65% recharge
target is not treated here as a hard end-of-duty SOC constraint.

The first pilot is deliberately bounded: two GIRO duty cohorts (including
duty 13406, whose recorded GIRO duty charges at PARX) and one aligned k2/k3
short cohort, crossed with the four arms. CG uses the default partition
with array concurrency two. MIP runs depend on their matched CG cells and use
Scaglione, excluding `scaglione-compute-01`, also at concurrency two.
This cohort does not by itself exercise every documented station inventory;
the result reports which capacity rows and stations are actually active.

This is an explicit-event reference CG driver with one exact shortest-path
column added per iteration. It is not the production batched CG runner.
Comparisons among its four matched arms isolate the two configured factors;
comparisons with earlier production runs also mix driver and batching changes.
Each CG result retains its Gurobi LP log and reports network-build time plus
per-iteration LP time, pricing time, row count, column count, and nonzeros.
