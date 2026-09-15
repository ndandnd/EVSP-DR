# Next question3 experiment: full CG with fair ending energy

Design only,15September2026. No implementation, validation or production launch claimed. Existing zero-fee baseline-chain runs use flat prices and no ending-energy constraint; they do not answer this question. Existing peak08/12/18 comparisons reprice saved pools and add fixed-duty frontiers; they likewise do not run the required full CG loop.

## Minimal decisive comparison

Start with the existing62-trip, five-duty input and its three frozen tariff files. Retain240kWh initial/battery energy,350kW charging, zero charge-start fee and the same aggregate terminal-energy requirement280.7833253kWh in both treatments. This isolates routing flexibility in that cohort; it is not the240kW chain baseline or a full GIRO-capacity model. Do not change these conditions while attributing a benefit to routing.

Compare the existing fixed-duty event optimizer with a fresh singleton-initialized CG + two-stage MIP. An explicitly separate GIRO-seeded arm can test whether enriching a known feasible duty solution finds savings. Freeze the same graph representation, trip coverage semantics, fleet cap, tariff and terminal definition. Reuse existing graph construction only with exact identities. Report grid objectives and physically replayed costs separately, including final energy and selected trip assignments. Original repriced GIRO is a third comparator with an interval because its within-event power trace is unavailable.

## Necessary algorithm support

The inspected currenta0e0bb interfaces price trip and route-count duals but expose no aggregate terminal-energy dual. The baseline pool MIP also deduplicates by cheapest trip incidence. File hashes and matching interface lines are in interface_audit.json. This is an audit of these execution files, not a claim that no other experimental branch has relevant code.

For columns r with trip-incidence a_ir, conservative terminal energy e_r and objective c_r:

min sum_r c_r x_r; sum_r a_ir x_r >=1; sum_r e_r x_r >=E.

If an optional fleet cap sum_r x_r <=k is imposed, let its dual be gamma<=0. With trip duals pi_i>=0 and terminal dual beta>=0, pricing must minimize

reduced_cost_r = c_r - sum_i pi_i a_ir - beta e_r - gamma.

Put the terminal contribution on the appropriate sink transition, using the same conservative post-return energy that the master stores. Propagate it through the exact-best-route calculation, extra-column selection, reduced-cost recomputation, column persistence, restart identities and certification. The pricing certificate must cover these duals; an older certificate cannot be transferred.

Deduplication must preserve terminal-energy tradeoffs. For otherwise identical master incidence, routeA can discard routeB only when A costs no more and supplies at least as much terminal energy (plus any other modeled row dominance). Keeping only the cheapest route can remove an essential high-energy column. Trip-only signatures are insufficient. Recheck replay/inheritance/journaling and final-MIP readers, not just the LP constructor.

Initialization needs an explicit feasible phase-I treatment for trip and terminal-energy rows, and for a fleet cap when used. Do not silently seed the fresh arm with GIRO columns. Do not use artificial energy or trip coverage as a physical schedule.

## Proof and validation gates

1. Tiny enumerated event graphs: explicit full-route LP/MIP versus CG, including a nonzero terminal dual. Check reduced-cost signs and that a more expensive high-energy route survives deduplication.
2. Zero terminal dual must reproduce the old priced objective and exact-best route; changed terminal target must invalidate incompatible cached optimization/restart metadata.
3. Native licensedk1 fixture and the complete known five-duty witness: independently replay energy, coverage, cost and end energy. Then matched fixed-duty/frontier checks on the same conditions.
4. Native budgeted pilot before the three tariff cases. Default partition; exclude scaglione-compute-01; preserve per-restart results and source hashes. Record a concrete runtime/memory basis before choosing production limits.

Main experiment retains the standard weighted-cost CG then two-stage MIP, which can find improved incumbents but does not alone prove charging optimality for the full model. A separate charging-only CG with fleet<=k and the same energy row can supply an appropriate full-model LP lower bound. If that certified lower bound closes against the fixed-duty integer objective within tolerance, it supports optimality within the declared graph. A tied finite-pool incumbent without this certificate does not prove GIRO duties globally optimal.
