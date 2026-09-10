# Charger-capacity × PARX-speed pilot

This is a controlled 2×2 pilot on the exact conservative event/SOC graph.
Every arm uses a 240 kWh battery, starts at 240 kWh, uses a 0 kWh reserve,
and has no terminal target beyond the same reserve. Non-PARX stations remain
at 240 kW. The speed factor changes only PARX from 240 kW to the documented
60 kW. The capacity factor adds the documented finite inventories 2190L=1,
4808=1, 3127L=2, 7880C=1, JON_A=1; PARX remains unlimited.

The capacity arms put every one-minute finite-site row in the restricted LP
before pricing, pass its dual into the event shortest path, and impose the
same rows in the pool MIP. The exact stopping test is therefore a full LP
certificate for this conservative discretized graph when the run reports
`certified_rc_optimal=true`, zero artificials, and a nonnegative terminal
minimum reduced cost. It is not a certificate for continuous SOC or the
separate nonlinear GIRO charge curve.

The selected MIP routes are independently swept as half-open continuous
connection intervals. Existing nonlinear new-physics results use a 15% SOC
floor and model recharge targets; they are separate evidence. A 65% recharge
target is not treated here as a hard end-of-duty SOC constraint.

The first pilot is deliberately bounded: one GIRO duty cohort and one aligned
k2/k3 short cohort, crossed with the four arms. CG uses the default partition
with array concurrency two. MIP runs depend on their matched CG cells and use
Scaglione, excluding `scaglione-compute-01`, also at concurrency two.
