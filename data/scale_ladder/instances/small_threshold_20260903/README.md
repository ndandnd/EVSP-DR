# Small exact-recovery threshold cohort (2026-09-03)

This directory freezes 50 GIRO duty-union instances before solver outcomes are
observed.  Targets are `k = 2, 5, 8, 9, 10`.  At each target there are six
fixed-seed probability samples and four predeclared structural samples:
trip-light, trip-heavy, service-energy-heavy, and tight scheduled inter-trip
gaps.  The trip-light row is an objective test of the proposed “easier GIRO
duties” direction; it is not selected after looking at solver performance.

Each of the 42 source GIRO duty orders has an independently optimized,
replay-validated continuous charging schedule under the current frozen
idealized physics: 240 kWh battery, 240 kW at every charging site, zero reserve,
free terminal SOC, flat tariff, and a $5 charging-start cost.  Therefore each
selected union has a physical `k`-route upper bound.  This does not prove that
the known routes are representable on the event lattice and does not prove
optimality.  Known routes are not injected into raw column generation.

`input_plan.json`, `selection_manifest.csv`, `candidate_universe.csv`, and the
per-duty certificate registry preserve the design, identities, features, and
caveats needed to audit the experiment.  Do not use this idealized cohort for
claims about savings relative to GIRO's recorded charging schedule: that
comparison requires a separately frozen charger-power, charger-capacity, and
terminal-SOC contract.
