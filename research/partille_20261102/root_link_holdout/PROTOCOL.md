# Does recombining trip connections recover eight buses on the other five chains?

Registered on 2 November 2026 Pacific, before the first held-out run. Chain 2 was the development case and is excluded. This is a frozen-method comparison on five existing instances. Their earlier fleet counts are known. The new recombination and same-root control outcomes have not been inspected.

## Cases and input boundary

Include C1, C3, C4, C5 and C6, each at k=8, from the original v15 fresh PARX60 arm. The [input manifest](cases.json) fixes exactly one completed source attempt and one byte-hashed root prefix per case. C4 uses the completed restart. Use only the first N columns named by the original root summary, before the later dive, even if pricing stopped at its time limit. Record the pricing certificate separately. Each historical master used trip covering. Each new dispatch must serve every passenger trip exactly once.

Both arms use the same 240 kWh, initially full battery, zero reserve and no positive terminal energy requirement. PARX is 60 kW with unlimited plugs. Opportunity stations are 240 kW, one plug each except two at 3127L. Use the same station-minute capacities, 2.5 kWh by 5-minute event model, flat source prices, five-unit charging start fee and original travel inputs. No arm reads GIRO passenger assignments, a witness schedule or another chain's output.

## Two independent runs on each root

* **Recombine consecutive trips.** Use the algorithm developed on C2. Count observed consecutive passenger-trip pairs across all root records. Process trips in order of start time, breaking ties by canonical input row. Search successors by descending observed frequency, then row. Maximum bipartite matching in this graph gives a minimum path cover. Verify the matching and its equal-size vertex cover. If eight or fewer paths result, retime each once, in order of its first trip. Reserve occupied charger minutes as each duty is accepted. Use one priced candidate per path and the existing 10,000 penalty on occupied finite-plug minutes. Allow 180 s for this one retiming order. No restarts, row permutations or tuning after seeing held-out outcomes. Report prefix/matching and replay time in addition to retiming.
* **Select original whole duties.** Build a binary feasibility MIP using the exact same root prefix. Each trip must be served once, fleet at most eight, with the original terminal-energy and all occupied finite station-minute rows. No start, new duties or secondary objective. Allow Gurobi 180 s with two threads and seed zero, stopping on the first feasible selection. Distinguish proved finite-pool infeasibility, timeout without a selection, and a selected schedule rejected by replay. The two time limits apply to different operations. Report total elapsed time for each method.

## Outcomes and next decision

The primary result is the number of independently replayed eight-bus dispatches out of **five predefined cases per arm**. Check trip identities, chronological transitions, energy, charging blocks, terminal requirement, modeled costs and plug occupancy. Establish the timetable's concurrent-trip floor before claiming fleet optimality. Report any verified smaller fleet separately and investigate the assumed target. Keep every miss and failed or missing case in the denominator. Retry only a documented technical interruption using identical code/settings, preserving all attempts and timing. A genuine first-order charging conflict or absence of a feasible solution within the budget ends the case.

Report the paired table, the discordant pairs, cause of each miss, incremental time, and graph plus root CG time. Count actual preceding attempts separately, including preemption. Do not count pricing or LP time again inside CG. Raw historical post-dive covering incumbents are descriptive only. Quote cost differences only between replay-valid equal fleets under the same physics and distinguish grid cost from continuous replay cost. Do not infer cost optimality from feasibility or population success rates from five chains.

Consider a separately registered k10 study only after all five rows have endpoints, at least three repairs achieve verified eight and every claimed hit passes the independent replay. That is a resource decision. The existing scientific release gates for fresh CG, paired MIPs and sparse sequential runs are separate.
