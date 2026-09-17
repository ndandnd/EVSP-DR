# Independent review of the time-only fleet certificates

**Verified, with an important upper-bound qualification.** The numerical lower-bound certificates are sound for all 102 audited inputs. They support a stronger fleet statement than comparing the weighted CG solution's route weight with GIRO's fleet. They do not automatically certify the weighted charging objective or event-lattice feasibility of GIRO duties.

| Check | Independent result |
|---|---|
| Exact input/master/reference hashes | All 102 instance inputs match; frozen master and both reference files match |
| Graph construction | All 612 graph hashes rebuilt independently; shortest-path closure used Dijkstra rather than the generator's Floyd–Warshall |
| Matching, vertex cover and path partition | All 612 certificates verified, including equal matching/cover sizes |
| Closure antichain | All 306 checked against full DAG reachability, not just direct adjacency |
| Group lower bounds versus GIRO duties | Equal in both groups for every case: 102/102 |
| Mixed lower bound | Equals k in 93 cases; equals k−1 in the same nine cases |
| F4 upper-bound source | All 42 saved optimized-duty result hashes and physical-replay flags verified; each of the 102 inputs is exactly a union of those complete duties with matching times, locations and energy data |
| Computation | 27.2 seconds locally; no solver calls or cluster submissions |

## Why this is a valid fleet lower bound

For consecutive service trips on any production route, the time between them must accommodate the production deadhead path, possibly through charging stations, plus nonnegative waiting and charging. The reference shortest-path closure is no longer than that path: it permits all reference locations as intermediate stops and uses unrounded durations, while the production travel arcs round upward. Ignoring battery, charging, depot and maximum-wait restrictions only enlarges the route set. Consequently every production route maps to a path in this relaxed time DAG.

A certified antichain contains A trips such that no path can serve two of them. Assigning dual weight one to those trips and zero elsewhere gives a fleet-cover lower bound of A, including fractional covering solutions. With group segregation, a route belongs to exactly one group, so the two antichain bounds add. This argument does not depend on the electricity-cost coefficient, saved CG columns, or a pricing stopping tolerance.

The matching and equal-size vertex cover also certify a minimum DAG path partition. The explicit antichain is the part that makes the fractional route-cover bound directly checkable. Reference triangle closure alone should not be assumed to imply transitivity of a trip graph in every conceivable dataset; this review checked actual reachability. All 306 closure graphs here are also transitive. The direct-matrix graphs are not used for the production lower-bound claim: a charger detour can be faster than an available direct arc.

## What can be concluded now

1. **Separated baseline model:** every feasible integer or fractional solution needs at least k buses/total route weight. This remains a valid lower bound if reserve, battery or depot-rate constraints are tightened.
2. **Continuous baseline charging model:** F4 supplies a physical k-bus witness, retaining GIRO's trips and reoptimizing charging, with homogeneous 240 kWh, uniform 240 kW, no reserve, free ending SOC and no shared capacity. Thus its minimum fleet is exactly k in all 102 separated instances. The same reasoning proves minimum fleet k in the 93 mixed cases whose time lower bound is k.
3. **Nine mixed cases:** the current integer fleet interval is [k−1, k]. The time-only lower bound does not produce a physically feasible k−1 electric schedule. Do not call the one-unit difference an achieved integer fleet saving.
4. **Production event-lattice model:** F4 explicitly did not establish that its continuous schedules lie on the 2.5 kWh/5-minute lattice. The lower bound k is valid, but an event-representable k-bus upper witness is still needed before declaring its minimum fleet k. The full segregation experiment can supply such witnesses. Existing appropriate event-model integer solutions could also close individual cases after group compatibility is checked.
5. None of these fleet arguments proves GIRO's charging schedule or weighted objective optimal. A physical k-bus witness plus fleet lower bound says nothing about the cheapest charging pattern among k-bus schedules.

The fixed-duty membership/hash check validates the saved F4 witnesses and links them to every chain input; this review did not rerun their solver or repeat their full continuous energy replay. Their numerical physical-replay results are retained in F4. The separate saved-positive-LP-route adjacency check is owned by the other reviewer and should be cited alongside this result when available.

Reproduce: `python3 independent_audit.py` from this directory or invoke its absolute path. `independent_audit.json` records source hashes and each verified graph. No production scripts or experiment settings were edited.
