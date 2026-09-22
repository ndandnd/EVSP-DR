# Slide 1 — A certified fractional solution can still miss an integer fleet

- Fresh C1: route weight **8**, spread across **99 fractional routes**; the finite pool proves that **9 whole routes** are required.
- A separate diagnostic adds eight physically replayed witness routes: the augmented pool proves **8**. All eight routes have positive reduced cost at the saved fresh duals (0.103–53.086).
- LP pricing seeks a cheaper fractional solution. A route can fail that test yet complete a useful integer combination.

Suggested visual: “99 fractional routes, total weight 8” → “fresh pool: integer fleet 9”; below it, “+8 complementary witness routes” → “augmented pool: integer fleet 8.” Label the added routes as a **diagnostic witness**, not an input to the new algorithm.

Speaker note: C1 has 194 trips and 39,940 fresh columns. Its weighted LP value is 800,383.688; the witness costs 800,479.960. Under covering, the gap is **96.2720 = 72.1421 selected reduced costs + 24.1299 duplicate-coverage dual value**. Two trips are duplicated. This shows why these particular columns are not attractive to pricing at the final LP duals; it does not rule out all near-zero enrichment or earlier generation.

Sources: [saved duals](/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_sources/c1_k08/fresh_cg.json), [fresh 9/9 proof](/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c1_k08/control/391804_r0/gurobi.log:156), [augmented 8/8 proof](/Users/nadan/Documents/projects/demandresponse/outputs/week_20260921/evidence/k8_witness/c1_k08/augmented/391805_r0/gurobi.log:112), [recomputed identity](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/integer_pricing_explanation/verified_numbers.json).

# Slide 2 — Fix a route, reprice its completion, then try alternatives

1. Start from the frozen fresh pool; impose target fleet cap **K=8**.
2. Fix a promising fractional route to **xᵣ=1**. Re-solve the covering LP: remaining coverage and bus slots change its dual prices.
3. Generate feasible routes with **reduced cost cᵣ − Σᵢaᵢᵣπᵢ − μ < −10⁻⁴**. Here cᵣ includes bus penalty and charging; μ≤0 is the fleet-cap dual.
4. Dive deeper, or release a fixing and try another route. Preserve generated columns. Pass the enlarged pool and any self-discovered cover to the final integer MIP.

Suggested visual: `fractional master → fix one route → new duals → priced routes → deeper dive / alternative` with a second arrow from all generated routes to `final pool MIP + own-dive start`.

Speaker note: No known witness routes are supplied. Fixing does not delete covered trips or prohibit overlap. Three candidate orderings favor largest LP value, longest route, then lowest cost per served trip across restarts. At most three alternatives per level, two restarts and 40 nodes; this is heuristic search, not exhaustive branching. Even an uncertified node can guide the dive if artificials are zero. C1 actually backtracked at depths four and five before obtaining its eight-route cover.

Sources: [fixings](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:280), [pricing](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:417), [search](/Users/nadan/Documents/projects/demandresponse/.codex-work/integer-columns-20260921/src/diving_pricing_pilot.py:791), [C1 trace](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/integer_pricing_explanation/verified_numbers.json).

# Slide 3 — Paired k8 test: 7/8 target hits versus 0/8 controls

| Frozen case | Treatment hits across two seeds | Control hits across two seeds |
|---|---:|---:|
| C1 | 2/2 | 0/2 |
| C3 | 2/2 | 0/2 |
| C4 | 1/2 | 0/2 |
| C5 | 2/2 | 0/2 |

- All seven successful treatments finish at **8 buses, bound 8 in their augmented pools**; their own eight-route starts are accepted.
- Controls all finish at 9; five prove 9 in the unchanged pool. The C4 miss remains 9/bound 8, not an infeasibility result.
- Nominal matched budget: **3,600 s = dive subprocess wall + final MIP solver time**; graph caches are existing prerequisites and MIP setup/replay is measured separately.

Footer: Four selected cases × two seeds; historical 240 kWh/240 kW, zero-reserve covering model. Route replay passes; duplicate service generally remains and shared charger capacity is not imposed. Finite-pool proofs only.

Speaker note: Successful treatments take 14.14–35.41 minutes actual elapsed (817.53–2,043.62 charged seconds). The failed treatment takes 61.57 minutes elapsed. Controls and the miss exceed nominal charged 3,600 s by 0.68–4.70 s due to termination overrun; do not call this a hard one-hour end-to-end test. Both arms use the same case/seed, fresh input pool, physics and eight-thread MIP settings, but treatment adds both columns and its own start. No independent ablation separates those effects. These are eight paired runs on four instances, not eight independent instances or a general success-rate estimate.

Source: [complete paired proof table](/Users/nadan/Documents/projects/demandresponse/outputs/research_followup_20260921/integer_pricing_explanation/proof_links.md), [verified audit](/Users/nadan/Documents/projects/demandresponse/outputs/research_management_20260921/monitor_20260921T235438Z/integer_audit/README.md).
